# -*- coding: utf-8 -*-
from __future__ import annotations

import configparser
import json
import os
from pathlib import Path
import subprocess
import sys

import streamlit as st

from legal_agent import LegalRAGStore, get_default_config
from legal_agent.config import LLMSettings


st.set_page_config(page_title="法律 RAG 评测面板", page_icon="⚖️", layout="wide")

APP_DIR = Path(__file__).resolve().parent
CONFIG_INI_PATH = APP_DIR / "config.ini"
HARNESS_SCRIPT_PATH = APP_DIR / "run_legal_rag_harness.py"
HARNESS_REPORT_PATH = APP_DIR / "eval" / "reports" / "legal_rag_harness_latest.json"
HARNESS_BENCHMARK_PATH = APP_DIR / "eval" / "legal_qa_benchmark.json"
HARNESS_MODES = ("hybrid", "llm_retrieval")


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


def build_harness_env(llm_settings: LLMSettings) -> dict[str, str]:
    env = dict(os.environ)
    env["RAG_LLM_BASE_URL"] = llm_settings.base_url
    env["RAG_LLM_API_KEY"] = llm_settings.api_key
    env["RAG_LLM_MODEL"] = llm_settings.model
    env["RAG_LLM_TEMPERATURE"] = str(llm_settings.temperature)
    env["RAG_LLM_MAX_TOKENS"] = str(llm_settings.max_tokens)
    env["USE_TORCH"] = "1"
    env["USE_TF"] = "0"
    return env


def rebuild_knowledge_base() -> str:
    store = get_store()
    stats = store.rebuild()
    return f"重建完成：{stats.documents} 个文档，{stats.chunks} 个 chunks。"


def run_harness(modes: list[str], llm_settings: LLMSettings) -> dict:
    if not HARNESS_SCRIPT_PATH.exists():
        raise FileNotFoundError(f"Harness script not found: {HARNESS_SCRIPT_PATH}")
    if not HARNESS_BENCHMARK_PATH.exists():
        raise FileNotFoundError(f"Harness benchmark not found: {HARNESS_BENCHMARK_PATH}")

    command = [
        sys.executable,
        str(HARNESS_SCRIPT_PATH),
        "--benchmark",
        str(HARNESS_BENCHMARK_PATH),
        "--modes",
        *modes,
        "--output",
        str(HARNESS_REPORT_PATH),
        "--details",
    ]
    completed = subprocess.run(
        command,
        cwd=str(APP_DIR),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=3600,
        env=build_harness_env(llm_settings),
    )
    return {
        "command": command,
        "exit_code": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
        "report_path": str(HARNESS_REPORT_PATH),
    }


def load_harness_report() -> dict | None:
    if not HARNESS_REPORT_PATH.exists():
        return None
    return json.loads(HARNESS_REPORT_PATH.read_text(encoding="utf-8"))


def format_rate(value: float | int | None) -> str:
    try:
        return f"{float(value) * 100:.1f}%"
    except (TypeError, ValueError):
        return "-"


def load_benchmark_text() -> str:
    if not HARNESS_BENCHMARK_PATH.exists():
        raise FileNotFoundError(f"Benchmark file not found: {HARNESS_BENCHMARK_PATH}")
    return HARNESS_BENCHMARK_PATH.read_text(encoding="utf-8")


def save_benchmark_text(text: str) -> None:
    parsed = json.loads(text)
    if not isinstance(parsed, list):
        raise ValueError("Benchmark JSON 顶层必须是数组。")
    HARNESS_BENCHMARK_PATH.write_text(
        json.dumps(parsed, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def render_sidebar() -> tuple[LLMSettings, bool]:
    config = get_default_config()
    store = get_store()
    stats = store.get_stats()
    ini_settings = load_llm_settings_from_ini()

    with st.sidebar:
        st.title("评测控制台")
        st.caption("主问答窗口已迁移到 PySide6；这里仅保留 benchmark 编辑、索引重建和评测结果。")
        st.write(f"知识库文档数：`{stats.documents}`")
        st.write(f"知识库 chunks：`{stats.chunks}`")
        st.write(f"SQLite：`{config.sqlite_path}`")
        st.write(f"向量索引：`{config.faiss_path}`")
        st.write(f"Benchmark：`{HARNESS_BENCHMARK_PATH}`")
        st.write(f"报告：`{HARNESS_REPORT_PATH}`")

        if st.button("重建索引", width="stretch"):
            with st.spinner("正在重建知识库索引..."):
                message = rebuild_knowledge_base()
            st.success(message)

        st.subheader("评测模式")
        selected_modes: list[str] = []
        for mode in HARNESS_MODES:
            if st.checkbox(mode, value=True, key=f"mode_{mode}"):
                selected_modes.append(mode)
        show_case_details = st.checkbox("显示失败样例", value=True)

        st.subheader("LLM 配置")
        retrieval_mode = st.selectbox(
            "默认问答模式",
            options=["llm_retrieval", "hybrid"],
            index=0 if ini_settings.retrieval_mode == "llm_retrieval" else 1,
            disabled=True,
            help="评测时按选择的 mode 覆盖；这里仅展示当前默认配置。",
        )
        base_url = st.text_input("Base URL", value=ini_settings.base_url)
        api_key = st.text_input("API Key", value=ini_settings.api_key, type="password")
        model = st.text_input("Model", value=ini_settings.model)
        current_settings = LLMSettings(
            base_url=base_url,
            api_key=api_key,
            model=model,
            temperature=ini_settings.temperature,
            max_tokens=ini_settings.max_tokens,
            retrieval_mode=retrieval_mode,
            answer_profile="quality",
        )
        if current_settings.disabled_reason:
            st.warning(current_settings.disabled_reason)

        if st.button("运行回归评测", width="stretch"):
            if not selected_modes:
                st.error("至少选择一个评测模式。")
            elif current_settings.disabled_reason:
                st.error(f"无法运行评测：{current_settings.disabled_reason}")
            else:
                with st.spinner("正在运行回归评测..."):
                    result = run_harness(selected_modes, current_settings)
                st.session_state.harness_last_run = result
                if result["exit_code"] == 0:
                    st.success("评测完成，结果已刷新。")
                else:
                    st.error(f"评测失败，退出码：{result['exit_code']}")

    return current_settings, show_case_details


def render_benchmark_editor() -> None:
    st.subheader("Benchmark 编辑")
    st.caption("这里的内容就是评测固定文本。修改后保存，再运行回归评测。")
    st.caption(f"文件位置：`{HARNESS_BENCHMARK_PATH}`")

    if "benchmark_editor_text" not in st.session_state:
        st.session_state.benchmark_editor_text = load_benchmark_text()

    editor_cols = st.columns([1, 1, 4])
    with editor_cols[0]:
        if st.button("重新加载 Benchmark", width="stretch"):
            st.session_state.benchmark_editor_text = load_benchmark_text()
            st.rerun()
    with editor_cols[1]:
        if st.button("保存 Benchmark", width="stretch"):
            try:
                save_benchmark_text(st.session_state.benchmark_editor_text)
            except Exception as exc:
                st.error(f"保存失败：{exc}")
            else:
                st.success("Benchmark 已保存。")
    st.text_area(
        "legal_qa_benchmark.json",
        key="benchmark_editor_text",
        height=360,
    )


def render_harness_dashboard(show_case_details: bool) -> None:
    st.subheader("评测结果")
    report = load_harness_report()
    if report is None:
        st.info("还没有评测报告。先运行一次回归评测。")
        return

    st.caption(f"报告文件：`{HARNESS_REPORT_PATH}`")
    requested_modes = report.get("requested_modes") or []
    if requested_modes:
        st.write("模式对比：`" + "` / `".join(requested_modes) + "`")

    last_run = st.session_state.get("harness_last_run")
    if last_run:
        with st.expander("最近一次评测日志", expanded=False):
            st.code(last_run.get("stdout", "") or "(无输出)", language="text")
            if last_run.get("stderr"):
                st.code(last_run["stderr"], language="text")

    skipped_modes = report.get("skipped_modes") or []
    for item in skipped_modes:
        st.warning(f"已跳过 {item.get('mode', '')}：{item.get('reason', '')}")

    for mode_report in report.get("modes", []):
        mode_name = str(mode_report.get("mode", ""))
        metrics = mode_report.get("metrics", {}) or {}
        with st.expander(f"{mode_name} 指标", expanded=True):
            metric_cols = st.columns(6)
            metric_cols[0].metric("检索命中率", format_rate(metrics.get("retrieval_hit_rate")))
            metric_cols[1].metric("证据筛选命中率", format_rate(metrics.get("evidence_selection_hit_rate")))
            metric_cols[2].metric("引用准确率", format_rate(metrics.get("citation_hit_rate")))
            metric_cols[3].metric("答案合规率", format_rate(metrics.get("answer_pass_rate")))
            metric_cols[4].metric("平均耗时", f"{float(metrics.get('avg_latency_ms', 0.0)):.0f} ms")
            metric_cols[5].metric("综合通过率", format_rate(metrics.get("overall_pass_rate")))

            summary_cols = st.columns(3)
            summary_cols[0].caption(f"检索覆盖：{format_rate(metrics.get('retrieval_reference_coverage'))}")
            summary_cols[1].caption(
                f"证据筛选覆盖：{format_rate(metrics.get('evidence_selection_reference_coverage'))}"
            )
            summary_cols[2].caption(f"引用覆盖：{format_rate(metrics.get('citation_reference_coverage'))}")

            case_results = mode_report.get("cases", []) or []
            failed_cases = [case for case in case_results if not case.get("case_pass")]
            st.caption(f"样本数：{len(case_results)}，失败样例：{len(failed_cases)}")

            if show_case_details and failed_cases:
                for case in failed_cases:
                    with st.expander(case.get("id", "未命名样例"), expanded=False):
                        st.write(f"问题：{case.get('question', '')}")
                        for reason in case.get("fail_reasons", []):
                            st.write(f"- {reason}")
                        preview = case.get("result_preview", {}) or {}
                        answer_preview = str(preview.get("answer", "") or "").strip()
                        if answer_preview:
                            st.code(answer_preview, language="text")


def main() -> None:
    _, show_case_details = render_sidebar()
    st.title("法律 RAG 评测面板")
    st.caption("主问答窗口请通过 PySide6 桌面版启动；这里仅负责 benchmark 编辑、索引重建和评测结果展示。")
    render_benchmark_editor()
    render_harness_dashboard(show_case_details)


if __name__ == "__main__":
    main()
