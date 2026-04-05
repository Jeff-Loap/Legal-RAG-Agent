from __future__ import annotations

import json
import time
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Any

from .config import LLMSettings
from .live_eval import evaluate_live_turn
from .workflow import LegalRAGAgent


def _new_usage_tracker() -> dict[str, int]:
    return {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "llm_calls": 0,
    }


def _summarize_mode_result(
    *,
    mode: str,
    question: str,
    result: dict[str, Any],
    elapsed_seconds: float,
) -> dict[str, Any]:
    evaluation = evaluate_live_turn(result)
    token_usage = result.get("token_usage") or _new_usage_tracker()
    summary = {
        "mode": mode,
        "question": question,
        "answer": result.get("answer", ""),
        "citations": result.get("citations", []),
        "retrieved_chunks": result.get("retrieved_chunks", []),
        "thinking": result.get("thinking", ""),
        "conversation_scope": result.get("conversation_scope", ""),
        "scope_reason": result.get("scope_reason", ""),
        "retrieval_mode": result.get("retrieval_mode", mode),
        "llm_used": bool(result.get("llm_used")),
        "llm_error": result.get("llm_error", ""),
        "token_usage": token_usage,
        "elapsed_seconds": round(elapsed_seconds, 3),
        "tokens_per_second": round(
            float(token_usage.get("total_tokens", 0)) / max(elapsed_seconds, 0.001),
            3,
        ),
    }
    summary.update(
        {
            "overall_score": evaluation["overall_score"],
            "question_answer_overlap_score": evaluation["question_answer_overlap_score"],
            "retrieval_support_score": evaluation["retrieval_support_score"],
            "citation_link_score": evaluation["citation_link_score"],
            "answer_length_score": evaluation["answer_length_score"],
            "markdown_signal_score": evaluation["markdown_signal_score"],
            "issue_count": evaluation["issue_count"],
            "issues": evaluation["issues"],
            "pass": evaluation["pass"],
            "summary": evaluation["summary"],
            "retrieved_chunk_count": evaluation["retrieved_chunk_count"],
            "citation_count": evaluation["citation_count"],
            "linked_citation_count": evaluation["linked_citation_count"],
            "linked_chunk_count": evaluation["linked_chunk_count"],
            "evaluation": evaluation,
        }
    )
    return summary


def _mode_winner(left_mode: str, left_value: float, right_mode: str, right_value: float, higher_is_better: bool) -> str:
    if left_value == right_value:
        return "tie"
    if higher_is_better:
        return left_mode if left_value > right_value else right_mode
    return left_mode if left_value < right_value else right_mode


def compare_retrieval_modes(
    agent: LegalRAGAgent,
    question: str,
    llm_settings: LLMSettings,
    *,
    report_dir: Path | None = None,
    session_id: str = "",
    top_k: int | None = None,
) -> dict[str, Any]:
    text = " ".join(question.split()).strip()
    if not text:
        raise ValueError("Question is empty.")
    if not llm_settings.enabled:
        raise ValueError(llm_settings.disabled_reason or "LLM 配置无效。")

    mode_results: dict[str, dict[str, Any]] = {}
    for mode in ("hybrid", "llm_retrieval"):
        tracker = _new_usage_tracker()
        mode_settings = replace(llm_settings, retrieval_mode=mode)
        started_at = time.perf_counter()
        result = agent.ask(
            question=text,
            session_id=session_id,
            llm_settings=mode_settings,
            top_k=top_k,
            usage_tracker=tracker,
        )
        elapsed = time.perf_counter() - started_at
        mode_results[mode] = _summarize_mode_result(
            mode=mode,
            question=text,
            result=result,
            elapsed_seconds=elapsed,
        )

    hybrid = mode_results["hybrid"]
    llm_retrieval = mode_results["llm_retrieval"]
    comparison = {
        "token_delta_prompt": int(hybrid["token_usage"]["prompt_tokens"]) - int(llm_retrieval["token_usage"]["prompt_tokens"]),
        "token_delta_completion": int(hybrid["token_usage"]["completion_tokens"]) - int(llm_retrieval["token_usage"]["completion_tokens"]),
        "token_delta_total": int(hybrid["token_usage"]["total_tokens"]) - int(llm_retrieval["token_usage"]["total_tokens"]),
        "elapsed_delta_seconds": round(float(hybrid["elapsed_seconds"]) - float(llm_retrieval["elapsed_seconds"]), 3),
        "score_delta": round(float(hybrid["overall_score"]) - float(llm_retrieval["overall_score"]), 4),
        "winner_by_score": _mode_winner(
            "hybrid",
            float(hybrid["overall_score"]),
            "llm_retrieval",
            float(llm_retrieval["overall_score"]),
            higher_is_better=True,
        ),
        "winner_by_tokens": _mode_winner(
            "hybrid",
            float(hybrid["token_usage"]["total_tokens"]),
            "llm_retrieval",
            float(llm_retrieval["token_usage"]["total_tokens"]),
            higher_is_better=False,
        ),
        "winner_by_latency": _mode_winner(
            "hybrid",
            float(hybrid["elapsed_seconds"]),
            "llm_retrieval",
            float(llm_retrieval["elapsed_seconds"]),
            higher_is_better=False,
        ),
    }

    report = {
        "question": text,
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "mode_results": mode_results,
        "comparison": comparison,
    }

    if report_dir is not None:
        report_dir.mkdir(parents=True, exist_ok=True)
        report_path = report_dir / "mode_compare_latest.json"
        report["report_path"] = str(report_path)
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    return report
