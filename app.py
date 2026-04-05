from __future__ import annotations

import configparser
from pathlib import Path

import streamlit as st

from legal_agent import LegalRAGStore, get_default_config
from legal_agent.config import LLMSettings
from legal_agent.live_eval import evaluate_live_turn
from legal_agent.mode_compare import compare_retrieval_modes
from legal_agent.workflow import LegalRAGAgent


st.set_page_config(page_title="法律 RAG 实时评测面板", page_icon="⚖️", layout="wide")

APP_DIR = Path(__file__).resolve().parent
CONFIG_INI_PATH = APP_DIR / "config.ini"
LIVE_POLL_SECONDS = 2


@st.cache_resource(show_spinner=False)
def get_store() -> LegalRAGStore:
    return LegalRAGStore(get_default_config())


def load_llm_settings_from_ini() -> LLMSettings:
    parser = configparser.ConfigParser()
    if CONFIG_INI_PATH.exists():
        parser.read(CONFIG_INI_PATH, encoding="utf-8")

    return LLMSettings(
        base_url=parser.get("llm", "base_url", fallback="").strip(),
        api_key=parser.get("llm", "api_key", fallback="").strip(),
        model=parser.get("llm", "model", fallback="").strip(),
        temperature=parser.getfloat("llm", "temperature", fallback=0.1),
        max_tokens=parser.getint("llm", "max_tokens", fallback=700),
        retrieval_mode=parser.get("llm", "retrieval_mode", fallback="llm_retrieval").strip() or "llm_retrieval",
        answer_profile="quality",
    )
def history_entry_payload(entry) -> dict[str, object]:
    return {
        "question": entry.question,
        "answer": entry.answer,
        "thinking": entry.thinking,
        "retrieved_chunks": entry.retrieved_chunks,
        "citations": entry.citations,
        "conversation_scope": entry.conversation_scope,
        "scope_reason": entry.scope_reason,
        "retrieval_mode": entry.retrieval_mode,
        "effective_question": entry.effective_question,
        "llm_error": entry.llm_error,
    }


def rebuild_knowledge_base() -> str:
    store = get_store()
    stats = store.rebuild()
    return f"重建完成：{stats.documents} 个文档，{stats.chunks} 个 chunks。"


def backfill_pending_live_evaluations(limit: int = 100) -> dict[str, int]:
    store = get_store()
    pending_entries = store.list_pending_history_entries(limit=limit)
    processed = 0
    for entry in pending_entries:
        evaluation = evaluate_live_turn(history_entry_payload(entry))
        store.save_live_evaluation(entry.id, evaluation)
        processed += 1
    return {
        "processed": processed,
        "pending": store.count_pending_history_entries(),
    }


def sync_live_update_token(store: LegalRAGStore) -> str:
    token = store.get_live_update_token()
    st.session_state["live_update_token"] = token
    return token


@st.fragment(run_every=f"{LIVE_POLL_SECONDS}s")
def render_live_update_watcher(enabled: bool) -> None:
    if not enabled:
        return
    store = get_store()
    latest_token = store.get_live_update_token()
    last_seen_token = st.session_state.get("live_update_token", "")
    if not last_seen_token:
        st.session_state["live_update_token"] = latest_token
        return
    if latest_token != last_seen_token:
        st.session_state["live_update_token"] = latest_token
        st.rerun()


def render_sidebar() -> tuple[bool, LLMSettings]:
    store = get_store()
    config = get_default_config()
    stats = store.get_stats()
    live_summary = store.get_live_evaluation_summary(limit=200)
    ini_settings = load_llm_settings_from_ini()

    with st.sidebar:
        st.title("实时评测控制台")
        st.caption("主问答窗口在 PySide6；这里展示真实问答的实时评测结果。")
        st.write(f"文档数：`{stats.documents}`")
        st.write(f"Chunk 数：`{stats.chunks}`")
        st.write(f"SQLite：`{config.sqlite_path}`")
        st.write(f"向量索引：`{config.faiss_path}`")
        st.write(f"已评测：`{live_summary['evaluated']}`")
        st.write(f"待评测：`{live_summary['pending']}`")
        st.write(f"评测中：`{live_summary.get('processing', 0)}`")
        total = live_summary["evaluated"] + live_summary["pending"] + live_summary.get("processing", 0)
        st.write(f"评测覆盖率：`{(live_summary['evaluated'] / max(total, 1)) * 100:.1f}%`")
        st.write(f"当前模型：`{ini_settings.model or '-'}`")
        st.write(f"当前模式：`{ini_settings.retrieval_mode}`")

        auto_refresh = st.checkbox("监听新记录后刷新", value=True)
        st.caption(f"检测间隔：{LIVE_POLL_SECONDS}s；仅检测到新记录或评测状态变化时刷新。")

        if st.button("重建索引", width="stretch"):
            with st.spinner("正在重建知识库索引..."):
                message = rebuild_knowledge_base()
            st.success(message)

        if st.button("补评历史问答", width="stretch"):
            with st.spinner("正在补评未完成的历史问答..."):
                result = backfill_pending_live_evaluations(limit=100)
            st.success(f"已处理 {result['processed']} 条，剩余待评测 {result['pending']} 条。")
            st.rerun()

    return auto_refresh, ini_settings


def render_summary_cards(live_summary: dict[str, float | int]) -> None:
    total = int(live_summary["evaluated"]) + int(live_summary["pending"]) + int(live_summary.get("processing", 0))
    coverage = int(live_summary["evaluated"]) / max(total, 1)
    cols = st.columns(6)
    cols[0].metric("已评测", f"{int(live_summary['evaluated'])}")
    cols[1].metric("待评测", f"{int(live_summary['pending'])}")
    cols[2].metric("评测中", f"{int(live_summary.get('processing', 0))}")
    cols[3].metric("覆盖率", f"{coverage * 100:.1f}%")
    cols[4].metric("整体得分", f"{float(live_summary['overall_score']) * 100:.1f}%")
    cols[5].metric("通过率", f"{float(live_summary['pass_rate']) * 100:.1f}%")

    subcols = st.columns(3)
    subcols[0].caption(
        f"引用关联：{float(live_summary['citation_link_score']) * 100:.1f}%  |  "
        f"关键词重叠：{float(live_summary['question_answer_overlap_score']) * 100:.1f}%"
    )
    subcols[1].caption(f"候选关联：{float(live_summary['retrieval_support_score']) * 100:.1f}%")
    subcols[2].caption(f"答案长度：{float(live_summary['answer_length_score']) * 100:.1f}%")


def render_recent_evaluations(show_details: bool) -> None:
    store = get_store()
    recent_rows = store.list_recent_live_evaluations(limit=25)
    if not recent_rows:
        st.info("当前还没有实时评测记录。先在 PySide6 里发一轮问题，或点击“补评历史问答”。")
        return

    table_rows = []
    for row in recent_rows:
        table_rows.append(
            {
                "turn": row["turn_id"],
                "scope": row["conversation_scope"],
                "mode": row["retrieval_mode"],
                "status": row["status"],
                "score": f"{float(row['overall_score']) * 100:.1f}%",
                "overlap": f"{float(row['question_answer_overlap_score']) * 100:.1f}%",
                "citation": f"{float(row['citation_link_score']) * 100:.1f}%",
                "retrieval": f"{float(row['retrieval_support_score']) * 100:.1f}%",
                "issues": row["issue_count"],
                "updated": row["updated_at"],
            }
        )

    st.dataframe(table_rows, use_container_width=True, hide_index=True)

    if show_details:
        st.subheader("最近问答详情")
        for row in recent_rows[:8]:
            with st.expander(f"turn {row['turn_id']} · {row['conversation_scope']} · {float(row['overall_score']) * 100:.1f}%", expanded=False):
                st.write(f"问题：{row['question']}")
                st.write(f"答案：{row['answer']}")
                st.caption(
                    " | ".join(
                        [
                            f"模式：{row['retrieval_mode']}",
                            f"候选证据：{len(row['retrieved_chunks'])}",
                            f"引用证据：{len(row['citations'])}",
                            f"问题-答案重叠：{float(row['question_answer_overlap_score']) * 100:.1f}%",
                            f"引用关联：{float(row['citation_link_score']) * 100:.1f}%",
                            f"综合得分：{float(row['overall_score']) * 100:.1f}%",
                        ]
                    )
                )
                if row["summary"]:
                    st.info(row["summary"])
                if row["issues"]:
                    for issue in row["issues"]:
                        st.warning(issue)


def render_mode_compare_section(ini_settings: LLMSettings) -> None:
    store = get_store()
    config = get_default_config()
    st.subheader("同题模式对比")
    st.caption("输入同一个法律问题，分别运行 `hybrid` 和 `llm_retrieval`，同时比较 token 消耗与实时评测分。")

    default_question = st.session_state.get("mode_compare_question", "")
    with st.form("mode_compare_form", clear_on_submit=False):
        question = st.text_area(
            "待对比问题",
            value=default_question,
            height=140,
            placeholder="例如：我方当事人按期交货后，对方以质量问题拒付尾款并单方解除合同，应如何应对？",
        )
        top_k = st.number_input(
            "候选上限",
            min_value=1,
            max_value=12,
            value=int(config.final_top_k),
            step=1,
        )
        submitted = st.form_submit_button("开始对比")

    if not submitted and "mode_compare_report" not in st.session_state:
        return

    if submitted:
        question = " ".join(question.split()).strip()
        if not question:
            st.error("请先输入一个法律问题。")
            return
        st.session_state["mode_compare_question"] = question
        agent = LegalRAGAgent(store=store, config=config)
        try:
            with st.spinner("正在分别运行 hybrid 与 llm_retrieval，并统计 token 与效果指标..."):
                report = compare_retrieval_modes(
                    agent=agent,
                    question=question,
                    llm_settings=ini_settings,
                    report_dir=APP_DIR / "eval" / "reports",
                    top_k=int(top_k),
                )
        except Exception as exc:
            st.error(str(exc))
            return
        st.session_state["mode_compare_report"] = report
        st.success(f"对比完成，报告已保存到 `{report.get('report_path', '-')}`")

    report = st.session_state.get("mode_compare_report")
    if not report:
        return

    comparison = report["comparison"]
    mode_results = report["mode_results"]

    summary_cols = st.columns(6)
    summary_cols[0].metric("token 差值", f"{comparison['token_delta_total']:+d}")
    summary_cols[1].metric("score 差值", f"{comparison['score_delta']:+.4f}")
    summary_cols[2].metric("赢家(效果)", comparison["winner_by_score"])
    summary_cols[3].metric("赢家(token)", comparison["winner_by_tokens"])
    summary_cols[4].metric("赢家(耗时)", comparison["winner_by_latency"])
    summary_cols[5].metric("总耗时差", f"{comparison['elapsed_delta_seconds']:+.3f}s")

    rows = []
    for mode in ("hybrid", "llm_retrieval"):
        item = mode_results[mode]
        rows.append(
            {
                "mode": mode,
                "prompt_tokens": int(item["token_usage"]["prompt_tokens"]),
                "completion_tokens": int(item["token_usage"]["completion_tokens"]),
                "total_tokens": int(item["token_usage"]["total_tokens"]),
                "llm_calls": int(item["token_usage"]["llm_calls"]),
                "elapsed_seconds": float(item["elapsed_seconds"]),
                "overall_score": float(item["overall_score"]),
                "overlap": float(item["question_answer_overlap_score"]),
                "retrieval_support": float(item["retrieval_support_score"]),
                "citation_link": float(item["citation_link_score"]),
                "answer_length": float(item["answer_length_score"]),
                "retrieved_chunks": int(item["retrieved_chunk_count"]),
                "citations": int(item["citation_count"]),
                "pass": bool(item["pass"]),
            }
        )

    st.dataframe(rows, use_container_width=True, hide_index=True)

    for mode in ("hybrid", "llm_retrieval"):
        item = mode_results[mode]
        with st.expander(f"{mode} 详情 · 得分 {float(item['overall_score']) * 100:.1f}% · token {int(item['token_usage']['total_tokens'])}", expanded=False):
            detail_cols = st.columns(4)
            detail_cols[0].metric("prompt tokens", f"{int(item['token_usage']['prompt_tokens'])}")
            detail_cols[1].metric("completion tokens", f"{int(item['token_usage']['completion_tokens'])}")
            detail_cols[2].metric("总 token", f"{int(item['token_usage']['total_tokens'])}")
            detail_cols[3].metric("耗时", f"{float(item['elapsed_seconds']):.2f}s")
            st.caption(
                " | ".join(
                    [
                        f"整体得分：{float(item['overall_score']) * 100:.1f}%",
                        f"关键词重叠：{float(item['question_answer_overlap_score']) * 100:.1f}%",
                        f"引用关联：{float(item['citation_link_score']) * 100:.1f}%",
                        f"候选关联：{float(item['retrieval_support_score']) * 100:.1f}%",
                        f"通过：{'是' if item['pass'] else '否'}",
                    ]
                )
            )
            if item.get("summary"):
                st.info(item["summary"])
            if item.get("issues"):
                for issue in item["issues"]:
                    st.warning(issue)
            st.markdown("**回答**")
            st.markdown(item["answer"] or "_空_")
            if item.get("retrieved_chunks"):
                st.markdown("**候选证据**")
                for idx, chunk in enumerate(item["retrieved_chunks"][:5], start=1):
                    source_name = chunk.get("source_name", "")
                    text = str(chunk.get("text", "")).strip()
                    st.write(f"[{idx}] {source_name}")
                    st.caption(text[:300])


def main() -> None:
    auto_refresh, ini_settings = render_sidebar()
    store = get_store()
    live_summary = store.get_live_evaluation_summary(limit=200)
    sync_live_update_token(store)

    st.title("法律 RAG 实时评测面板")
    st.caption("页面直接读取 PySide6 产生的真实问答记录；后台轮询 SQLite 变更标识，仅在出现新记录或评测状态变化时刷新。")
    render_live_update_watcher(auto_refresh)

    render_summary_cards(live_summary)
    render_mode_compare_section(ini_settings)
    st.subheader("最新实时评测")
    render_recent_evaluations(show_details=True)


if __name__ == "__main__":
    main()
