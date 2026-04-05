from __future__ import annotations

import re
from datetime import datetime
from typing import Any

from .retrieval import extract_priority_legal_terms, extract_query_terms


TERM_PATTERN = re.compile(r"[A-Za-z0-9]+|[\u4e00-\u9fff]{2,}")


def _dedupe_terms(terms: list[str]) -> list[str]:
    deduped: list[str] = []
    seen: set[str] = set()
    for term in terms:
        normalized = " ".join(str(term).split()).strip().lower()
        if len(normalized) < 2 or normalized in seen:
            continue
        seen.add(normalized)
        deduped.append(str(term).strip())
    return deduped


def _collect_terms(text: str) -> list[str]:
    content = str(text or "").strip()
    if not content:
        return []
    terms: list[str] = []
    terms.extend(extract_priority_legal_terms(content))
    terms.extend(extract_query_terms(content))
    terms.extend(TERM_PATTERN.findall(content))
    return _dedupe_terms(terms)


def _chunk_id(payload: dict[str, Any]) -> str:
    return str(payload.get("chunk_id", "")).strip()


def _normalize_list(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, dict)]


def evaluate_live_turn(payload: dict[str, Any]) -> dict[str, Any]:
    question = str(payload.get("question", "") or "").strip()
    answer = str(payload.get("answer", "") or "").strip()
    retrieved_chunks = _normalize_list(payload.get("retrieved_chunks", []))
    citations = _normalize_list(payload.get("citations", []))
    conversation_scope = str(payload.get("conversation_scope", "legal") or "legal").strip() or "legal"
    retrieval_mode = str(payload.get("retrieval_mode", "llm_retrieval") or "llm_retrieval").strip() or "llm_retrieval"
    scope_reason = str(payload.get("scope_reason", "") or "").strip()
    llm_error = str(payload.get("llm_error", "") or "").strip()

    retrieved_ids = {_chunk_id(chunk) for chunk in retrieved_chunks if _chunk_id(chunk)}
    citation_ids = {_chunk_id(citation) for citation in citations if _chunk_id(citation)}
    linked_ids = retrieved_ids & citation_ids
    linked_citation_count = sum(1 for citation in citations if _chunk_id(citation) in retrieved_ids)
    retrieved_count = len(retrieved_chunks)
    citation_count = len(citations)

    question_terms = _collect_terms(question)
    answer_terms = _collect_terms(answer)
    question_term_set = {term.lower() for term in question_terms}
    answer_term_set = {term.lower() for term in answer_terms}
    term_overlap_terms = sorted(question_term_set & answer_term_set)
    question_answer_overlap_score = len(term_overlap_terms) / max(len(question_term_set), 1)

    citation_link_score = linked_citation_count / max(citation_count, 1) if citation_count else 0.0
    retrieval_support_score = len(linked_ids) / max(retrieved_count, 1) if retrieved_count else 0.0
    answer_length_score = min(len(answer) / 260.0, 1.0)
    markdown_signal_score = 1.0 if re.search(r"(^|\n)\s*(?:[-*]|\d+\.)\s+", answer) or "**" in answer or "```" in answer else 0.0

    issues: list[str] = []
    if not answer:
        issues.append("回答为空")
    if conversation_scope == "legal":
        if retrieved_count == 0:
            issues.append("法律问答未记录候选证据")
        if citation_count == 0:
            issues.append("法律问答未输出引用证据")
        if citation_count > 0 and linked_citation_count == 0:
            issues.append("引用与候选证据没有建立关联")
        if question_term_set and question_answer_overlap_score < 0.12:
            issues.append("回答与问题关键词重叠偏低")
    else:
        if answer_length_score < 0.15:
            issues.append("通用回复过短")

    if llm_error:
        issues.append(f"存在运行错误：{llm_error}")

    if conversation_scope == "legal":
        overall_score = (
            question_answer_overlap_score * 0.34
            + citation_link_score * 0.26
            + retrieval_support_score * 0.22
            + answer_length_score * 0.18
        )
    else:
        overall_score = (
            question_answer_overlap_score * 0.50
            + answer_length_score * 0.35
            + markdown_signal_score * 0.15
        )

    overall_score = max(0.0, min(1.0, overall_score))
    is_pass = overall_score >= 0.6 and not (not answer or (conversation_scope == "legal" and citation_count == 0))
    summary = (
        f"关键词重叠 {question_answer_overlap_score:.2f}，"
        f"引用关联 {citation_link_score:.2f}，"
        f"候选关联 {retrieval_support_score:.2f}，"
        f"综合得分 {overall_score:.2f}"
    )

    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return {
        "status": "evaluated",
        "pass": is_pass,
        "overall_score": round(overall_score, 4),
        "question_answer_overlap_score": round(question_answer_overlap_score, 4),
        "retrieval_support_score": round(retrieval_support_score, 4),
        "citation_link_score": round(citation_link_score, 4),
        "answer_length_score": round(answer_length_score, 4),
        "markdown_signal_score": round(markdown_signal_score, 4),
        "question_term_count": len(question_term_set),
        "answer_term_count": len(answer_term_set),
        "retrieved_chunk_count": retrieved_count,
        "citation_count": citation_count,
        "linked_citation_count": linked_citation_count,
        "linked_chunk_count": len(linked_ids),
        "issue_count": len(issues),
        "issues": issues,
        "summary": summary,
        "conversation_scope": conversation_scope,
        "retrieval_mode": retrieval_mode,
        "scope_reason": scope_reason,
        "llm_error": llm_error,
        "created_at": now,
        "updated_at": now,
        "question": question,
        "answer": answer,
        "thinking": str(payload.get("thinking", "") or "").strip(),
    }
