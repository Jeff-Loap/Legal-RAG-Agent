# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import configparser
import json
import os
import re
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from legal_agent import LegalRAGAgent, LegalRAGStore, get_default_config
from legal_agent.config import LLMSettings


DEFAULT_BENCHMARK_PATH = Path(__file__).resolve().parent / "eval" / "legal_qa_benchmark.json"
DEFAULT_OUTPUT_PATH = Path(__file__).resolve().parent / "eval" / "reports" / "legal_rag_harness_latest.json"
DEFAULT_CONFIG_INI_PATH = Path(__file__).resolve().parent / "config.ini"
DEFAULT_MODES = (
    "hybrid:quality",
    "llm_retrieval:quality",
)


@dataclass(frozen=True)
class HarnessMode:
    retrieval_mode: str
    answer_profile: str

    @property
    def name(self) -> str:
        return f"{self.retrieval_mode}:{self.answer_profile}"

    @property
    def requires_llm(self) -> bool:
        return self.retrieval_mode == "llm_retrieval"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="运行法律 RAG 回归评测 harness。")
    parser.add_argument("--benchmark", default=str(DEFAULT_BENCHMARK_PATH), help="法律 QA benchmark JSON 文件路径。")
    parser.add_argument("--config-ini", default=str(DEFAULT_CONFIG_INI_PATH), help="LLM 配置 ini 文件路径。")
    parser.add_argument(
        "--modes",
        nargs="*",
        default=list(DEFAULT_MODES),
        help="评测模式，格式为 retrieval_mode[:answer_profile]，例如 hybrid 或 hybrid:quality。",
    )
    parser.add_argument("--top-k", type=int, default=None, help="覆盖 ask() 的 top_k；默认使用系统配置。")
    parser.add_argument("--details", action="store_true", help="输出每道题的详细评测结果。")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT_PATH), help="评测结果 JSON 输出路径。")
    parser.add_argument(
        "--skip-unavailable-modes",
        action="store_true",
        default=True,
        help="当模式要求 LLM 但当前未配置时，跳过该模式。",
    )
    parser.add_argument(
        "--fail-below-overall-pass-rate",
        type=float,
        default=None,
        help="若 overall_pass_rate 低于该值则返回非 0 退出码。",
    )
    parser.add_argument(
        "--fail-below-retrieval-hit-rate",
        type=float,
        default=None,
        help="若 retrieval_hit_rate 低于该值则返回非 0 退出码。",
    )
    parser.add_argument(
        "--fail-below-citation-hit-rate",
        type=float,
        default=None,
        help="若 citation_hit_rate 低于该值则返回非 0 退出码。",
    )
    return parser.parse_args()


def parse_mode_specs(mode_specs: list[str]) -> list[HarnessMode]:
    modes: list[HarnessMode] = []
    seen: set[str] = set()
    for raw_spec in mode_specs:
        spec = str(raw_spec).strip()
        if not spec:
            continue
        if ":" in spec:
            retrieval_mode, answer_profile = [item.strip() for item in spec.split(":", 1)]
        else:
            retrieval_mode, answer_profile = spec, "quality"
        if retrieval_mode not in {"hybrid", "llm_retrieval"}:
            raise ValueError(f"Unsupported retrieval mode: {retrieval_mode}")
        if answer_profile != "quality":
            raise ValueError(f"Unsupported answer profile: {answer_profile}. Only quality is supported.")
        mode = HarnessMode(retrieval_mode=retrieval_mode, answer_profile=answer_profile)
        if mode.name in seen:
            continue
        seen.add(mode.name)
        modes.append(mode)
    if not modes:
        raise ValueError("No valid harness modes were provided.")
    return modes


def load_benchmark(path: Path) -> list[dict[str, Any]]:
    items = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(items, list) or not items:
        raise ValueError(f"Benchmark file must be a non-empty list: {path}")
    validated: list[dict[str, Any]] = []
    for index, item in enumerate(items, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"Benchmark item #{index} is not a JSON object.")
        case_id = str(item.get("id", "")).strip()
        question = str(item.get("question", "")).strip()
        if not case_id:
            raise ValueError(f"Benchmark item #{index} is missing id.")
        if not question:
            raise ValueError(f"Benchmark item {case_id} is missing question.")
        validated.append(item)
    return validated


def load_llm_settings_from_ini(path: Path) -> LLMSettings:
    parser = configparser.ConfigParser()
    if path.exists():
        parser.read(path, encoding="utf-8")
    base_url = os.getenv("RAG_LLM_BASE_URL", parser.get("llm", "base_url", fallback="")).strip()
    api_key = os.getenv("RAG_LLM_API_KEY", parser.get("llm", "api_key", fallback="")).strip()
    model = os.getenv("RAG_LLM_MODEL", parser.get("llm", "model", fallback="")).strip()
    return LLMSettings(
        base_url=base_url,
        api_key=api_key,
        model=model,
        temperature=float(os.getenv("RAG_LLM_TEMPERATURE", parser.get("llm", "temperature", fallback="0.1"))),
        max_tokens=int(os.getenv("RAG_LLM_MAX_TOKENS", parser.get("llm", "max_tokens", fallback="700"))),
        retrieval_mode=parser.get("llm", "retrieval_mode", fallback="llm_retrieval").strip() or "llm_retrieval",
        answer_profile=parser.get("llm", "answer_profile", fallback="quality").strip() or "quality",
    )


def build_mode_settings(base_settings: LLMSettings, mode: HarnessMode) -> LLMSettings:
    return LLMSettings(
        base_url=base_settings.base_url,
        api_key=base_settings.api_key,
        model=base_settings.model,
        temperature=base_settings.temperature,
        max_tokens=base_settings.max_tokens,
        retrieval_mode=mode.retrieval_mode,
        answer_profile=mode.answer_profile,
    )


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", "", str(text or "")).lower()


def text_contains(text: str, needle: str) -> bool:
    return bool(needle) and normalize_text(needle) in normalize_text(text)


def ensure_variant_groups(value: Any) -> list[list[str]]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise ValueError("must_include_any must be a list.")
    groups: list[list[str]] = []
    for item in value:
        if isinstance(item, str):
            groups.append([item])
            continue
        if not isinstance(item, list) or not item:
            raise ValueError("Each must_include_any item must be a string or non-empty list.")
        variants = [str(variant).strip() for variant in item if str(variant).strip()]
        if not variants:
            raise ValueError("must_include_any contains an empty variant group.")
        groups.append(variants)
    return groups


def ensure_string_list(value: Any, field_name: str) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise ValueError(f"{field_name} must be a list.")
    return [str(item).strip() for item in value if str(item).strip()]


def get_expected_references(case: dict[str, Any]) -> list[dict[str, str]]:
    payload = case.get("expected_references")
    if payload is None:
        payload = case.get("expected", [])
    if not isinstance(payload, list):
        raise ValueError(f"Case {case.get('id')} expected_references must be a list.")
    references: list[dict[str, str]] = []
    for item in payload:
        if not isinstance(item, dict):
            raise ValueError(f"Case {case.get('id')} contains an invalid expected reference.")
        references.append(
            {
                "source_name": str(item.get("source_name", "")).strip(),
                "article_anchor": str(item.get("article_anchor", "")).strip(),
            }
        )
    return references


def chunk_matches_reference(chunk: dict[str, Any], expected: dict[str, str]) -> bool:
    actual_source = str(chunk.get("source_name", "")).strip()
    actual_article = str((chunk.get("metadata", {}) or {}).get("article_anchor", "")).strip()
    expected_source = expected.get("source_name", "")
    expected_article = expected.get("article_anchor", "")
    if expected_source and actual_source != expected_source:
        return False
    if expected_article and actual_article != expected_article:
        return False
    return True


def resolve_cited_chunks(result: dict[str, Any]) -> list[dict[str, Any]]:
    retrieved_chunks = result.get("retrieved_chunks") or []
    chunk_map = {
        str(chunk.get("chunk_id", "")).strip(): chunk
        for chunk in retrieved_chunks
        if str(chunk.get("chunk_id", "")).strip()
    }
    cited_chunks: list[dict[str, Any]] = []
    for citation in result.get("citations") or []:
        chunk_id = str((citation or {}).get("chunk_id", "")).strip()
        if chunk_id and chunk_id in chunk_map:
            cited_chunks.append(chunk_map[chunk_id])
    return cited_chunks


def match_reference_set(
    chunks: list[dict[str, Any]],
    expected_references: list[dict[str, str]],
) -> tuple[int, list[dict[str, str]], list[dict[str, str]]]:
    matched: list[dict[str, str]] = []
    missing: list[dict[str, str]] = []
    for expected in expected_references:
        if any(chunk_matches_reference(chunk, expected) for chunk in chunks):
            matched.append(expected)
        else:
            missing.append(expected)
    return len(matched), matched, missing


def build_candidate_sizes(agent: LegalRAGAgent, top_k: int | None) -> list[int]:
    target_top_k = max(int(top_k or agent.config.final_top_k), 1)
    sizes: list[int] = []
    for size in (
        target_top_k,
        max(target_top_k * 2, 8),
        max(target_top_k * 3, 12),
    ):
        if size not in sizes:
            sizes.append(size)
    return sizes


def collect_raw_candidate_chunks(
    agent: LegalRAGAgent,
    question: str,
    settings: LLMSettings,
    top_k: int | None,
) -> list[dict[str, Any]]:
    recent_conversation, memory_hits, effective_question = agent._prepare_context_layers(question, "")
    retrieval_plan = agent._plan_retrieval_strategy(
        question=question,
        effective_question=effective_question,
        recent_conversation=recent_conversation,
        memory_hits=memory_hits,
        llm_settings=settings,
    )
    query_variants = retrieval_plan.get("query_variants", []) or [question.strip()]
    domains = retrieval_plan.get("domains", []) or []
    ordered_chunks: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for query_variant in query_variants:
        for candidate_k in build_candidate_sizes(agent, top_k):
            candidate_chunks = agent._retrieve_candidate_chunks_for_mode(
                query_variant,
                retrieval_mode=settings.retrieval_mode,
                top_k=candidate_k,
            )
            candidate_chunks = agent._prioritize_domain_chunks(candidate_chunks, domains)
            for chunk in candidate_chunks:
                chunk_id = str(chunk.get("chunk_id", "")).strip()
                if not chunk_id or chunk_id in seen_ids:
                    continue
                seen_ids.add(chunk_id)
                ordered_chunks.append(chunk)
    return ordered_chunks


def evaluate_answer_constraints(answer: str, case: dict[str, Any]) -> dict[str, Any]:
    checks = case.get("answer_checks", {}) or {}
    if not isinstance(checks, dict):
        raise ValueError(f"Case {case.get('id')} answer_checks must be an object.")

    must_include_all = ensure_string_list(checks.get("must_include_all"), "must_include_all")
    must_include_any = ensure_variant_groups(checks.get("must_include_any"))
    must_exclude_all = ensure_string_list(checks.get("must_exclude_all"), "must_exclude_all")

    missing_include_all = [term for term in must_include_all if not text_contains(answer, term)]
    missing_include_any = [group for group in must_include_any if not any(text_contains(answer, variant) for variant in group)]
    violation_excludes = [term for term in must_exclude_all if text_contains(answer, term)]

    return {
        "passed": not missing_include_all and not missing_include_any and not violation_excludes,
        "missing_include_all": missing_include_all,
        "missing_include_any_groups": missing_include_any,
        "violation_excludes": violation_excludes,
    }


def evaluate_case(
    case: dict[str, Any],
    result: dict[str, Any],
    elapsed_ms: int,
    raw_candidate_chunks: list[dict[str, Any]],
) -> dict[str, Any]:
    expected_scope = str(case.get("expected_scope", "legal")).strip() or "legal"
    actual_scope = str(result.get("conversation_scope", "") or "").strip() or "unknown"
    expected_references = get_expected_references(case)
    retrieved_chunks = result.get("retrieved_chunks") or []
    cited_chunks = resolve_cited_chunks(result)

    candidate_match_count, candidate_matched, candidate_missing = match_reference_set(
        raw_candidate_chunks,
        expected_references,
    )
    retrieval_match_count, retrieval_matched, retrieval_missing = match_reference_set(retrieved_chunks, expected_references)
    citation_match_count, citation_matched, citation_missing = match_reference_set(cited_chunks, expected_references)
    answer_evaluation = evaluate_answer_constraints(str(result.get("answer", "")), case)

    require_retrieval_hit = bool(case.get("require_retrieval_hit", bool(expected_references)))
    require_citation_hit = bool(case.get("require_citation_hit", False))
    scope_pass = actual_scope == expected_scope
    candidate_retrieval_hit = candidate_match_count == len(expected_references) if expected_references else True
    evidence_selection_hit = retrieval_match_count == len(expected_references) if expected_references else True
    citation_hit = citation_match_count == len(expected_references) if expected_references else True
    retrieval_pass = evidence_selection_hit if require_retrieval_hit else True
    citation_pass = citation_hit if require_citation_hit else True
    answer_pass = bool(answer_evaluation["passed"])
    case_pass = scope_pass and retrieval_pass and citation_pass and answer_pass

    fail_reasons: list[str] = []
    if not scope_pass:
        fail_reasons.append(f"scope expected={expected_scope} actual={actual_scope}")
    if not retrieval_pass:
        fail_reasons.append(f"retrieval missing={json.dumps(retrieval_missing, ensure_ascii=False)}")
    if not citation_pass:
        fail_reasons.append(f"citation missing={json.dumps(citation_missing, ensure_ascii=False)}")
    if not answer_pass:
        if answer_evaluation["missing_include_all"]:
            fail_reasons.append(
                f"answer missing_include_all={json.dumps(answer_evaluation['missing_include_all'], ensure_ascii=False)}"
            )
        if answer_evaluation["missing_include_any_groups"]:
            fail_reasons.append(
                "answer missing_include_any_groups="
                + json.dumps(answer_evaluation["missing_include_any_groups"], ensure_ascii=False)
            )
        if answer_evaluation["violation_excludes"]:
            fail_reasons.append(
                f"answer violation_excludes={json.dumps(answer_evaluation['violation_excludes'], ensure_ascii=False)}"
            )

    reference_target = max(len(expected_references), 1)
    return {
        "id": str(case.get("id", "")).strip(),
        "question": str(case.get("question", "")).strip(),
        "elapsed_ms": elapsed_ms,
        "scope_pass": scope_pass,
        "actual_scope": actual_scope,
        "expected_scope": expected_scope,
        "candidate_retrieval_hit": candidate_retrieval_hit,
        "evidence_selection_hit": evidence_selection_hit,
        "citation_hit": citation_hit,
        "retrieval_pass": retrieval_pass,
        "citation_pass": citation_pass,
        "answer_pass": answer_pass,
        "case_pass": case_pass,
        "candidate_match_count": candidate_match_count,
        "retrieval_match_count": retrieval_match_count,
        "citation_match_count": citation_match_count,
        "expected_reference_count": len(expected_references),
        "candidate_match_rate": candidate_match_count / reference_target,
        "retrieval_match_rate": retrieval_match_count / reference_target,
        "citation_match_rate": citation_match_count / reference_target,
        "candidate_matched": candidate_matched,
        "candidate_missing": candidate_missing,
        "retrieval_matched": retrieval_matched,
        "retrieval_missing": retrieval_missing,
        "citation_matched": citation_matched,
        "citation_missing": citation_missing,
        "raw_candidate_count": len(raw_candidate_chunks),
        "retrieved_chunk_count": len(retrieved_chunks),
        "citation_count": len(result.get("citations") or []),
        "llm_used": bool(result.get("llm_used", False)),
        "llm_error": str(result.get("llm_error", "") or ""),
        "effective_question": str(result.get("effective_question", "") or ""),
        "fail_reasons": fail_reasons,
        "answer_checks": answer_evaluation,
        "result_preview": {
            "answer": str(result.get("answer", ""))[:900],
            "citations": result.get("citations") or [],
            "raw_candidates": [
                {
                    "chunk_id": chunk.get("chunk_id"),
                    "source_name": chunk.get("source_name"),
                    "article_anchor": str((chunk.get("metadata", {}) or {}).get("article_anchor", "")),
                    "score": chunk.get("score"),
                }
                for chunk in raw_candidate_chunks[:12]
            ],
            "retrieved_chunks": [
                {
                    "chunk_id": chunk.get("chunk_id"),
                    "source_name": chunk.get("source_name"),
                    "article_anchor": str((chunk.get("metadata", {}) or {}).get("article_anchor", "")),
                    "score": chunk.get("score"),
                }
                for chunk in retrieved_chunks[:12]
            ],
        },
    }


def build_exception_case_result(case: dict[str, Any], elapsed_ms: int, exc: Exception) -> dict[str, Any]:
    error_text = " ".join(f"{exc.__class__.__name__}: {exc}".split()).strip()
    guidance = ""
    if "idf vector is not fitted" in error_text.lower():
        guidance = "TF-IDF 索引与当前 scikit-learn 版本不兼容，请先重建知识库。"
    fail_reasons = [error_text]
    if guidance:
        fail_reasons.append(guidance)
    return {
        "id": str(case.get("id", "")).strip(),
        "question": str(case.get("question", "")).strip(),
        "elapsed_ms": elapsed_ms,
        "scope_pass": False,
        "actual_scope": "error",
        "expected_scope": str(case.get("expected_scope", "legal")).strip() or "legal",
        "candidate_retrieval_hit": False,
        "evidence_selection_hit": False,
        "citation_hit": False,
        "retrieval_pass": False,
        "citation_pass": False,
        "answer_pass": False,
        "case_pass": False,
        "candidate_match_count": 0,
        "retrieval_match_count": 0,
        "citation_match_count": 0,
        "expected_reference_count": len(get_expected_references(case)),
        "candidate_match_rate": 0.0,
        "retrieval_match_rate": 0.0,
        "citation_match_rate": 0.0,
        "candidate_matched": [],
        "candidate_missing": get_expected_references(case),
        "retrieval_matched": [],
        "retrieval_missing": get_expected_references(case),
        "citation_matched": [],
        "citation_missing": get_expected_references(case),
        "raw_candidate_count": 0,
        "retrieved_chunk_count": 0,
        "citation_count": 0,
        "llm_used": False,
        "llm_error": error_text,
        "effective_question": "",
        "fail_reasons": fail_reasons,
        "answer_checks": {
            "passed": False,
            "missing_include_all": [],
            "missing_include_any_groups": [],
            "violation_excludes": [],
        },
        "result_preview": {
            "answer": "",
            "citations": [],
            "raw_candidates": [],
            "retrieved_chunks": [],
        },
    }


def aggregate_mode_metrics(case_results: list[dict[str, Any]]) -> dict[str, Any]:
    total = max(len(case_results), 1)
    reference_cases = [item for item in case_results if int(item.get("expected_reference_count", 0)) > 0]
    total_references = max(sum(item["expected_reference_count"] for item in reference_cases), 1)
    return {
        "cases": len(case_results),
        "overall_pass_rate": sum(1 for item in case_results if item["case_pass"]) / total,
        "scope_accuracy": sum(1 for item in case_results if item["scope_pass"]) / total,
        "answer_pass_rate": sum(1 for item in case_results if item["answer_pass"]) / total,
        "retrieval_hit_rate": sum(1 for item in case_results if item["candidate_retrieval_hit"]) / total,
        "evidence_selection_hit_rate": sum(1 for item in case_results if item["evidence_selection_hit"]) / total,
        "citation_hit_rate": sum(1 for item in case_results if item["citation_hit"]) / total,
        "retrieval_reference_coverage": sum(item["candidate_match_count"] for item in reference_cases) / total_references,
        "evidence_selection_reference_coverage": sum(item["retrieval_match_count"] for item in reference_cases) / total_references,
        "citation_reference_coverage": sum(item["citation_match_count"] for item in reference_cases) / total_references,
        "llm_error_rate": sum(1 for item in case_results if item["llm_error"]) / total,
        "llm_used_rate": sum(1 for item in case_results if item["llm_used"]) / total,
        "avg_latency_ms": round(sum(item["elapsed_ms"] for item in case_results) / total, 2),
    }


def print_mode_summary(mode_name: str, metrics: dict[str, Any]) -> None:
    print(f"[mode] {mode_name}")
    print(
        "  "
        + "  ".join(
            [
                f"cases={metrics['cases']}",
                f"overall_pass_rate={metrics['overall_pass_rate']:.3f}",
                f"retrieval_hit_rate={metrics['retrieval_hit_rate']:.3f}",
                f"evidence_selection_hit_rate={metrics['evidence_selection_hit_rate']:.3f}",
                f"citation_hit_rate={metrics['citation_hit_rate']:.3f}",
                f"answer_pass_rate={metrics['answer_pass_rate']:.3f}",
                f"scope_accuracy={metrics['scope_accuracy']:.3f}",
                f"avg_latency_ms={metrics['avg_latency_ms']:.2f}",
            ]
        )
    )


def run_mode(
    agent: LegalRAGAgent,
    benchmark: list[dict[str, Any]],
    settings: LLMSettings,
    mode: HarnessMode,
    top_k: int | None,
    details: bool,
) -> dict[str, Any]:
    case_results: list[dict[str, Any]] = []
    for case in benchmark:
        start = time.perf_counter()
        try:
            raw_candidate_chunks = collect_raw_candidate_chunks(
                agent=agent,
                question=str(case["question"]).strip(),
                settings=settings,
                top_k=top_k,
            )
            result = agent.ask(
                question=str(case["question"]).strip(),
                session_id="",
                llm_settings=settings,
                top_k=top_k,
            )
            elapsed_ms = int((time.perf_counter() - start) * 1000)
            case_result = evaluate_case(case, result, elapsed_ms, raw_candidate_chunks)
        except Exception as exc:
            elapsed_ms = int((time.perf_counter() - start) * 1000)
            case_result = build_exception_case_result(case, elapsed_ms, exc)
        case_results.append(case_result)
        if details:
            status = "PASS" if case_result["case_pass"] else "FAIL"
            print(
                f"  - {case_result['id']} [{status}] retrieval_hit={case_result['candidate_retrieval_hit']} "
                f"evidence_selection_hit={case_result['evidence_selection_hit']} "
                f"citation_hit={case_result['citation_hit']} answer={case_result['answer_pass']} "
                f"latency_ms={case_result['elapsed_ms']}"
            )
            for reason in case_result["fail_reasons"]:
                print(f"    {reason}")
    metrics = aggregate_mode_metrics(case_results)
    print_mode_summary(mode.name, metrics)
    return {
        "mode": mode.name,
        "settings": asdict(settings),
        "metrics": metrics,
        "cases": case_results,
    }


def check_thresholds(report: dict[str, Any], args: argparse.Namespace) -> list[str]:
    failures: list[str] = []
    for mode_report in report.get("modes", []):
        metrics = mode_report.get("metrics", {})
        mode_name = str(mode_report.get("mode", ""))
        if args.fail_below_overall_pass_rate is not None:
            value = float(metrics.get("overall_pass_rate", 0.0))
            if value < args.fail_below_overall_pass_rate:
                failures.append(f"{mode_name} overall_pass_rate={value:.3f} < {args.fail_below_overall_pass_rate:.3f}")
        if args.fail_below_retrieval_hit_rate is not None:
            value = float(metrics.get("retrieval_hit_rate", 0.0))
            if value < args.fail_below_retrieval_hit_rate:
                failures.append(f"{mode_name} retrieval_hit_rate={value:.3f} < {args.fail_below_retrieval_hit_rate:.3f}")
        if args.fail_below_citation_hit_rate is not None:
            value = float(metrics.get("citation_hit_rate", 0.0))
            if value < args.fail_below_citation_hit_rate:
                failures.append(f"{mode_name} citation_hit_rate={value:.3f} < {args.fail_below_citation_hit_rate:.3f}")
    return failures


def main() -> int:
    args = parse_args()
    benchmark_path = Path(args.benchmark).resolve()
    config_ini_path = Path(args.config_ini).resolve()
    output_path = Path(args.output).resolve()

    benchmark = load_benchmark(benchmark_path)
    modes = parse_mode_specs(args.modes)
    base_llm_settings = load_llm_settings_from_ini(config_ini_path)

    config = get_default_config()
    store = LegalRAGStore(config)
    agent = LegalRAGAgent(store=store, config=config)

    report: dict[str, Any] = {
        "benchmark_path": str(benchmark_path),
        "config_ini_path": str(config_ini_path),
        "output_path": str(output_path),
        "knowledge_base_stats": asdict(store.get_stats()),
        "base_llm_enabled": base_llm_settings.enabled,
        "requested_modes": [mode.name for mode in modes],
        "skipped_modes": [],
        "modes": [],
    }

    print(f"benchmark={benchmark_path}")
    print(f"cases={len(benchmark)}")
    print(f"documents={report['knowledge_base_stats']['documents']}")
    print(f"chunks={report['knowledge_base_stats']['chunks']}")
    print(f"llm_enabled={base_llm_settings.enabled}")
    print("")

    for mode in modes:
        if mode.requires_llm and not base_llm_settings.enabled and args.skip_unavailable_modes:
            reason = f"{mode.name} requires LLM but config.ini/env is not enabled."
            print(f"[skip] {reason}")
            report["skipped_modes"].append({"mode": mode.name, "reason": reason})
            continue
        settings = build_mode_settings(base_llm_settings, mode)
        report["modes"].append(
            run_mode(
                agent=agent,
                benchmark=benchmark,
                settings=settings,
                mode=mode,
                top_k=args.top_k,
                details=args.details,
            )
        )
        print("")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"report_written={output_path}")

    failures = check_thresholds(report, args)
    if failures:
        print("")
        print("[threshold_failures]")
        for item in failures:
            print(f"- {item}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
