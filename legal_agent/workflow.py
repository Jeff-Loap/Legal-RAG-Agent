from __future__ import annotations

import json
import re
from typing import TypedDict

from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import END, StateGraph
from openai import OpenAI

from .config import AppConfig, LLMSettings, get_default_config, load_llm_settings_from_env
from .memory import SessionMemoryRetriever
from .retrieval import (
    LocalHybridRetriever,
    extract_article_refs,
    extract_law_names,
    extract_priority_legal_terms,
    extract_query_terms,
)
from .storage import HistoryEntry, LegalRAGStore

DOMAIN_SOURCE_HINTS: dict[str, tuple[str, ...]] = {
    "criminal": ("刑法", "治安管理处罚法"),
    "civil": ("民法典",),
    "labor": ("劳动法", "劳动合同法"),
    "privacy": ("个人信息保护法",),
    "safety": ("安全生产法", "危险化学品"),
    "tax": ("增值税", "税"),
    "general": (),
}

MAX_COMPLETION_SEGMENTS = 4
CONTINUE_OUTPUT_PROMPT = (
    "继续输出剩余内容。"
    "不要重复已经输出的部分，直接从上文结尾处继续。"
    "保持原有结构、编号、语气和 Markdown 格式。"
)


def _new_usage_tracker() -> dict[str, int]:
    return {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "llm_calls": 0,
    }


def _record_response_usage(usage_tracker: dict[str, int] | None, response: object) -> None:
    if usage_tracker is None:
        return
    usage = getattr(response, "usage", None)
    if usage is None:
        raise RuntimeError("LLM response missing usage metadata.")
    usage_tracker["prompt_tokens"] += int(getattr(usage, "prompt_tokens", 0) or 0)
    usage_tracker["completion_tokens"] += int(getattr(usage, "completion_tokens", 0) or 0)
    usage_tracker["total_tokens"] += int(getattr(usage, "total_tokens", 0) or 0)
    usage_tracker["llm_calls"] += 1


class AgentState(TypedDict, total=False):
    question: str
    session_id: str
    top_k: int
    retrieval_mode: str
    conversation_scope: str
    scope_reason: str
    recent_conversation: str
    effective_question: str
    memory_hits: list[dict]
    retrieved_chunks: list[dict]
    answer: str
    citations: list[dict]
    llm_used: bool
    llm_error: str


class LegalRAGAgent:
    def __init__(
        self,
        store: LegalRAGStore | None = None,
        config: AppConfig | None = None,
    ):
        self.config = config or get_default_config()
        self.store = store or LegalRAGStore(self.config)
        self.retriever = LocalHybridRetriever(self.store.fetch_chunks(), self.config)

    def refresh(self) -> None:
        self.retriever = LocalHybridRetriever(self.store.fetch_chunks(), self.config)

    def ask(
        self,
        question: str,
        session_id: str | None = None,
        llm_settings: LLMSettings | None = None,
        top_k: int | None = None,
        usage_tracker: dict[str, int] | None = None,
    ) -> dict:
        llm_settings = llm_settings or load_llm_settings_from_env()
        session_id = session_id or ""
        recent_conversation, memory_hits, effective_question = self._prepare_context_layers(
            question,
            session_id,
        )
        scope_info = self._classify_question_scope(
            question=question,
            recent_conversation=recent_conversation,
            memory_hits=memory_hits,
            llm_settings=llm_settings,
            usage_tracker=usage_tracker,
        )

        if scope_info["scope"] == "general":
            result = self._ask_general_chat(
                question=question,
                session_id=session_id,
                recent_conversation=recent_conversation,
                memory_hits=memory_hits,
                effective_question=effective_question,
                llm_settings=llm_settings,
                scope_reason=scope_info["reason"],
                usage_tracker=usage_tracker,
            )
            enriched = self._attach_thinking_summary(result, llm_settings)
            if usage_tracker is not None:
                enriched["token_usage"] = dict(usage_tracker)
            return enriched

        if llm_settings.retrieval_mode == "llm_retrieval":
            result = self._ask_llm_retrieval(
                question=question,
                session_id=session_id,
                llm_settings=llm_settings,
                recent_conversation=recent_conversation,
                memory_hits=memory_hits,
                effective_question=effective_question,
                scope_reason=scope_info["reason"],
                usage_tracker=usage_tracker,
            )
        else:
            graph = self._build_graph(llm_settings, usage_tracker=usage_tracker)
            result = graph.invoke(
                {
                    "question": question,
                    "session_id": session_id,
                    "top_k": top_k or self.config.final_top_k,
                    "retrieval_mode": llm_settings.retrieval_mode,
                    "conversation_scope": scope_info["scope"],
                    "scope_reason": scope_info["reason"],
                    "recent_conversation": recent_conversation,
                    "memory_hits": memory_hits,
                    "effective_question": effective_question,
                }
            )
        enriched = self._attach_thinking_summary(result, llm_settings)
        if usage_tracker is not None:
            enriched["token_usage"] = dict(usage_tracker)
        return enriched

    def stream_ask(
        self,
        question: str,
        session_id: str | None = None,
        llm_settings: LLMSettings | None = None,
        top_k: int | None = None,
    ):
        llm_settings = llm_settings or load_llm_settings_from_env()
        session_id = session_id or ""
        recent_conversation, memory_hits, effective_question = self._prepare_context_layers(
            question,
            session_id,
        )
        scope_info = self._classify_question_scope(
            question=question,
            recent_conversation=recent_conversation,
            memory_hits=memory_hits,
            llm_settings=llm_settings,
        )
        base_result = {
            "recent_conversation": recent_conversation,
            "memory_hits": memory_hits,
            "effective_question": effective_question,
            "retrieved_chunks": [],
            "citations": [],
            "llm_used": False,
            "llm_error": "",
            "retrieval_mode": llm_settings.retrieval_mode,
            "conversation_scope": scope_info["scope"],
            "scope_reason": scope_info["reason"],
        }
        for token in self._stream_text(self._build_thinking_summary(base_result, llm_settings)):
            yield {"type": "thinking_token", "content": token}

        if scope_info["scope"] == "general":
            yield from self._stream_general_chat_result(
                question=question,
                recent_conversation=recent_conversation,
                memory_hits=memory_hits,
                effective_question=effective_question,
                llm_settings=llm_settings,
                scope_reason=scope_info["reason"],
                base_result=base_result,
            )
            return

        if llm_settings.retrieval_mode == "llm_retrieval":
            yield from self._stream_llm_retrieval_result(
                question=question,
                recent_conversation=recent_conversation,
                memory_hits=memory_hits,
                effective_question=effective_question,
                llm_settings=llm_settings,
                scope_reason=scope_info["reason"],
                base_result=base_result,
            )
            return

        yield from self._stream_hybrid_result(
            question=question,
            recent_conversation=recent_conversation,
            memory_hits=memory_hits,
            effective_question=effective_question,
            llm_settings=llm_settings,
            scope_reason=scope_info["reason"],
            base_result=base_result,
            top_k=top_k,
        )

    def _build_graph(self, llm_settings: LLMSettings, usage_tracker: dict[str, int] | None = None):
        graph = StateGraph(AgentState)
        graph.add_node("retrieve", lambda state: self._retrieve_node(state, llm_settings, usage_tracker=usage_tracker))
        graph.add_node("answer", lambda state: self._answer_node(state, llm_settings, usage_tracker=usage_tracker))
        graph.set_entry_point("retrieve")
        graph.add_edge("retrieve", "answer")
        graph.add_edge("answer", END)
        return graph.compile()

    def _retrieve_node(
        self,
        state: AgentState,
        llm_settings: LLMSettings,
        usage_tracker: dict[str, int] | None = None,
    ) -> AgentState:
        recent_conversation = state.get("recent_conversation", "")
        memory_hits = state.get("memory_hits", [])
        effective_question = state.get("effective_question", state["question"])
        if not recent_conversation and not memory_hits:
            recent_conversation, memory_hits, effective_question = self._prepare_context_layers(
                state["question"],
                state.get("session_id", ""),
            )
        retrieval_plan = self._plan_retrieval_strategy(
            question=state["question"],
            effective_question=effective_question,
            recent_conversation=recent_conversation,
            memory_hits=memory_hits,
            llm_settings=llm_settings,
            usage_tracker=usage_tracker,
        )
        chunks = self._retrieve_relevant_chunks(
            question=state["question"],
            retrieval_plan=retrieval_plan,
            llm_settings=llm_settings,
            retrieval_mode=state.get("retrieval_mode", "hybrid"),
            top_k=state.get("top_k", self.config.final_top_k),
            recent_conversation=recent_conversation,
            memory_hits=memory_hits,
        )
        return {
            "recent_conversation": recent_conversation,
            "memory_hits": self._serialize_memory_hits(memory_hits),
            "effective_question": retrieval_plan.get("retrieval_query", effective_question),
            "retrieved_chunks": chunks,
            "conversation_scope": state.get("conversation_scope", "legal"),
            "scope_reason": state.get("scope_reason", ""),
        }

    def _retrieve_chunks(self, question: str, top_k: int | None = None) -> list[dict]:
        chunks = self.retriever.retrieve(
            question,
            top_k=top_k or self.config.final_top_k,
        )
        return [
            {
                "chunk_id": chunk.chunk_id,
                "chunk_index": getattr(chunk, "chunk_index", 0),
                "score": chunk.score,
                "source_name": chunk.source_name,
                "source_path": chunk.source_path,
                "title": chunk.title,
                "text": chunk.text,
                "metadata": chunk.metadata,
                "file_type": chunk.file_type,
            }
            for chunk in chunks
        ]

    def _answer_node(
        self,
        state: AgentState,
        llm_settings: LLMSettings,
        usage_tracker: dict[str, int] | None = None,
    ) -> AgentState:
        chunks = state.get("retrieved_chunks", [])
        memory_hits = state.get("memory_hits", [])
        recent_conversation = state.get("recent_conversation", "无")
        if llm_settings.enabled and chunks:
            try:
                answer = self._call_llm(
                    question=state["question"],
                    recent_conversation=recent_conversation,
                    memory_hits=memory_hits,
                    chunks=chunks,
                    llm_settings=llm_settings,
                    usage_tracker=usage_tracker,
                )
                return {
                    "answer": answer,
                    "citations": self._select_citations_for_answer(
                        question=state["question"],
                        answer=answer,
                        chunks=chunks,
                        llm_settings=llm_settings,
                        retrieval_mode="hybrid",
                        usage_tracker=usage_tracker,
                    ),
                    "llm_used": True,
                    "llm_error": "",
                    "recent_conversation": recent_conversation,
                    "memory_hits": memory_hits,
                    "effective_question": state.get("effective_question", state["question"]),
                    "retrieval_mode": "hybrid",
                    "conversation_scope": state.get("conversation_scope", "legal"),
                    "scope_reason": state.get("scope_reason", ""),
                }
            except Exception as exc:
                error_text = self._format_llm_error(exc)
                return {
                    "answer": self._fallback_answer(
                        state["question"],
                        chunks,
                        recent_conversation=recent_conversation,
                        memory_hits=memory_hits,
                        llm_error=error_text,
                    ),
                    "citations": self._build_citations(chunks[:3], state["question"]),
                    "llm_used": False,
                    "llm_error": error_text,
                    "recent_conversation": recent_conversation,
                    "memory_hits": memory_hits,
                    "effective_question": state.get("effective_question", state["question"]),
                    "retrieval_mode": "hybrid",
                    "conversation_scope": state.get("conversation_scope", "legal"),
                    "scope_reason": state.get("scope_reason", ""),
                }

        return {
            "answer": self._fallback_answer(
                state["question"],
                chunks,
                recent_conversation=recent_conversation,
                memory_hits=memory_hits,
            ),
            "citations": self._build_citations(chunks[:3], state["question"]),
            "llm_used": False,
            "llm_error": "",
            "recent_conversation": recent_conversation,
            "memory_hits": memory_hits,
            "effective_question": state.get("effective_question", state["question"]),
            "retrieval_mode": "hybrid",
            "conversation_scope": state.get("conversation_scope", "legal"),
            "scope_reason": state.get("scope_reason", ""),
        }

    def _ask_llm_retrieval(
        self,
        question: str,
        session_id: str,
        llm_settings: LLMSettings,
        recent_conversation: str | None = None,
        memory_hits: list[dict] | None = None,
        effective_question: str | None = None,
        scope_reason: str = "",
        usage_tracker: dict[str, int] | None = None,
    ) -> dict:
        if not llm_settings.enabled:
            reason = llm_settings.disabled_reason or "请先填写 Base URL、API Key 和 Model。"
            message = f"当前选择的是“大模型检索”模式，但没有可用的 LLM 配置：{reason}"
            return {
                "answer": message,
                "citations": [],
                "llm_used": False,
                "llm_error": message,
                "memory_hits": [],
                "effective_question": question,
                "retrieved_chunks": [],
                "retrieval_mode": "llm_retrieval",
                "conversation_scope": "legal",
                "scope_reason": scope_reason,
            }

        recent_conversation = recent_conversation or "无"
        memory_hits = memory_hits or []
        effective_question = effective_question or question
        retrieval_plan = self._plan_retrieval_strategy(
            question=question,
            effective_question=effective_question,
            recent_conversation=recent_conversation,
            memory_hits=memory_hits,
            llm_settings=llm_settings,
            usage_tracker=usage_tracker,
        )
        candidate_chunks = self._retrieve_relevant_chunks(
            question=question,
            retrieval_plan=retrieval_plan,
            llm_settings=llm_settings,
            retrieval_mode="llm_retrieval",
            top_k=12,
            recent_conversation=recent_conversation,
            memory_hits=memory_hits,
        )
        if not candidate_chunks:
            message = "大模型检索模式下未能从本地文件中召回候选证据，请先确认文件是否已经入库。"
            return {
                "answer": message,
                "citations": [],
                "llm_used": False,
                "llm_error": "",
                "recent_conversation": recent_conversation,
                "memory_hits": memory_hits,
                "effective_question": retrieval_plan.get("retrieval_query", effective_question),
                "retrieved_chunks": [],
                "retrieval_mode": "llm_retrieval",
                "conversation_scope": "legal",
                "scope_reason": scope_reason,
            }

        try:
            answer = self._call_llm_retrieval(
                question=question,
                recent_conversation=recent_conversation,
                memory_hits=memory_hits,
                chunks=candidate_chunks,
                llm_settings=llm_settings,
                usage_tracker=usage_tracker,
            )
            citations = self._select_citations_for_answer(
                question=question,
                answer=answer,
                chunks=candidate_chunks,
                llm_settings=llm_settings,
                preferred_indexes=self._extract_cited_candidate_numbers(answer, len(candidate_chunks)),
                retrieval_mode="llm_retrieval",
                usage_tracker=usage_tracker,
            )
            return {
                "answer": answer,
                "citations": citations,
                "llm_used": True,
                "llm_error": "",
                "recent_conversation": recent_conversation,
                "memory_hits": memory_hits,
                "effective_question": retrieval_plan.get("retrieval_query", effective_question),
                "retrieved_chunks": candidate_chunks,
                "retrieval_mode": "llm_retrieval",
                "conversation_scope": "legal",
                "scope_reason": scope_reason,
            }
        except Exception as exc:
            error_text = self._format_llm_error(exc)
            message = f"大模型检索模式调用失败：{error_text}"
            return {
                "answer": message,
                "citations": [],
                "llm_used": False,
                "llm_error": error_text,
                "recent_conversation": recent_conversation,
                "memory_hits": memory_hits,
                "effective_question": retrieval_plan.get("retrieval_query", effective_question),
                "retrieved_chunks": candidate_chunks,
                "retrieval_mode": "llm_retrieval",
                "conversation_scope": "legal",
                "scope_reason": scope_reason,
            }

    def _ask_general_chat(
        self,
        question: str,
        session_id: str,
        recent_conversation: str,
        memory_hits: list[dict],
        effective_question: str,
        llm_settings: LLMSettings,
        scope_reason: str,
        usage_tracker: dict[str, int] | None = None,
    ) -> dict:
        if llm_settings.enabled:
            try:
                answer = self._call_general_chat_llm(
                    question=question,
                    recent_conversation=recent_conversation,
                    memory_hits=memory_hits,
                    llm_settings=llm_settings,
                    usage_tracker=usage_tracker,
                )
                return {
                    "answer": answer,
                    "citations": [],
                    "llm_used": True,
                    "llm_error": "",
                    "recent_conversation": recent_conversation,
                    "memory_hits": memory_hits,
                    "effective_question": effective_question,
                    "retrieved_chunks": [],
                    "retrieval_mode": llm_settings.retrieval_mode,
                    "conversation_scope": "general",
                    "scope_reason": scope_reason,
                }
            except Exception as exc:
                error_text = self._format_llm_error(exc)
                return {
                    "answer": (
                        "当前问题已识别为通用对话，已跳过法律检索。"
                        f"\n\n通用对话调用失败：{error_text}"
                    ),
                    "citations": [],
                    "llm_used": False,
                    "llm_error": error_text,
                    "recent_conversation": recent_conversation,
                    "memory_hits": memory_hits,
                    "effective_question": effective_question,
                    "retrieved_chunks": [],
                    "retrieval_mode": llm_settings.retrieval_mode,
                    "conversation_scope": "general",
                    "scope_reason": scope_reason,
                }

        return {
            "answer": (
                "当前问题已识别为通用对话，已跳过法律检索。"
                f"\n\n当前未配置可用的大模型，因此无法继续进行普通对话回答：{llm_settings.disabled_reason or '缺少必要配置。'}"
            ),
            "citations": [],
            "llm_used": False,
            "llm_error": llm_settings.disabled_reason or "",
            "recent_conversation": recent_conversation,
            "memory_hits": memory_hits,
            "effective_question": effective_question,
            "retrieved_chunks": [],
            "retrieval_mode": llm_settings.retrieval_mode,
            "conversation_scope": "general",
            "scope_reason": scope_reason,
        }

    def _stream_ask_llm_retrieval(
        self,
        question: str,
        session_id: str,
        llm_settings: LLMSettings,
    ):
        result = self._ask_llm_retrieval(
            question=question,
            session_id=session_id,
            llm_settings=llm_settings,
        )
        for token in self._stream_text(result.get("answer", "")):
            yield {"type": "token", "content": token}
        yield {"type": "done", "result": result}

    def _stream_general_chat_result(
        self,
        question: str,
        recent_conversation: str,
        memory_hits: list[dict],
        effective_question: str,
        llm_settings: LLMSettings,
        scope_reason: str,
        base_result: dict,
    ):
        if not llm_settings.enabled:
            answer = (
                "当前问题已识别为通用对话，已跳过法律检索。"
                "\n\n当前未配置可用的大模型，因此无法继续进行普通对话回答。"
            )
            for token in self._stream_text(answer):
                yield {"type": "token", "content": token}
            result = dict(base_result)
            result.update(
                {
                    "answer": answer,
                    "citations": [],
                    "llm_used": False,
                    "llm_error": "",
                    "recent_conversation": recent_conversation,
                    "memory_hits": memory_hits,
                    "effective_question": effective_question,
                    "retrieved_chunks": [],
                    "retrieval_mode": llm_settings.retrieval_mode,
                    "conversation_scope": "general",
                    "scope_reason": scope_reason,
                }
            )
            yield {"type": "done", "result": self._attach_thinking_summary(result, llm_settings)}
            return

        answer_parts: list[str] = []
        llm_error = ""
        try:
            for token in self._stream_general_chat_llm(
                question=question,
                recent_conversation=recent_conversation,
                memory_hits=memory_hits,
                llm_settings=llm_settings,
            ):
                answer_parts.append(token)
                yield {"type": "token", "content": token}
        except Exception as exc:
            llm_error = self._format_llm_error(exc)
            if not answer_parts:
                answer = (
                    "当前问题已识别为通用对话，已跳过法律检索。"
                    f"\n\n通用对话调用失败：{llm_error}"
                )
                for token in self._stream_text(answer):
                    yield {"type": "token", "content": token}
            else:
                answer = "".join(answer_parts).strip()
        else:
            answer = "".join(answer_parts).strip()

        result = dict(base_result)
        result.update(
            {
                "answer": answer,
                "citations": [],
                "llm_used": bool(answer_parts),
                "llm_error": llm_error,
                "recent_conversation": recent_conversation,
                "memory_hits": memory_hits,
                "effective_question": effective_question,
                "retrieved_chunks": [],
                "retrieval_mode": llm_settings.retrieval_mode,
                "conversation_scope": "general",
                "scope_reason": scope_reason,
            }
        )
        yield {"type": "done", "result": self._attach_thinking_summary(result, llm_settings)}

    def _stream_llm_retrieval_result(
        self,
        question: str,
        recent_conversation: str,
        memory_hits: list[dict],
        effective_question: str,
        llm_settings: LLMSettings,
        scope_reason: str,
        base_result: dict,
    ):
        retrieval_plan = self._plan_retrieval_strategy(
            question=question,
            effective_question=effective_question,
            recent_conversation=recent_conversation,
            memory_hits=memory_hits,
            llm_settings=llm_settings,
        )
        candidate_chunks = self._retrieve_relevant_chunks(
            question=question,
            retrieval_plan=retrieval_plan,
            llm_settings=llm_settings,
            retrieval_mode="llm_retrieval",
            top_k=12,
            recent_conversation=recent_conversation,
            memory_hits=memory_hits,
        )
        result = dict(base_result)
        result.update(
            {
                "effective_question": retrieval_plan.get("retrieval_query", effective_question),
                "retrieved_chunks": candidate_chunks,
                "retrieval_mode": "llm_retrieval",
                "conversation_scope": "legal",
                "scope_reason": scope_reason,
            }
        )
        if not candidate_chunks:
            answer = "大模型检索模式下未能从本地文件中召回候选证据，请先确认文件是否已经入库。"
            for token in self._stream_text(answer):
                yield {"type": "token", "content": token}
            result.update(
                {
                    "answer": answer,
                    "citations": [],
                    "llm_used": False,
                    "llm_error": "",
                }
            )
            yield {"type": "done", "result": self._attach_thinking_summary(result, llm_settings)}
            return

        answer_parts: list[str] = []
        llm_error = ""
        try:
            for token in self._stream_llm_retrieval(
                question=question,
                recent_conversation=recent_conversation,
                memory_hits=memory_hits,
                chunks=candidate_chunks,
                llm_settings=llm_settings,
            ):
                answer_parts.append(token)
                yield {"type": "token", "content": token}
        except Exception as exc:
            llm_error = self._format_llm_error(exc)
            if not answer_parts:
                answer = f"大模型检索模式调用失败：{llm_error}"
                for token in self._stream_text(answer):
                    yield {"type": "token", "content": token}
            else:
                answer = "".join(answer_parts).strip()
        else:
            answer = "".join(answer_parts).strip()

        if llm_settings.enabled and answer:
            try:
                answer = self._self_check_answer(
                    question=question,
                    answer=answer,
                    recent_conversation=recent_conversation,
                    memory_hits=memory_hits,
                    chunks=candidate_chunks,
                    llm_settings=llm_settings,
                    retrieval_mode="llm_retrieval",
                )
                answer = self._ensure_answer_contains_law_content(
                    answer=answer,
                    chunks=candidate_chunks,
                )
            except Exception as exc:
                llm_error = llm_error or self._format_llm_error(exc)

        answer = self._ensure_answer_contains_law_content(
            answer=answer,
            chunks=candidate_chunks,
        )
        citations = self._select_citations_for_answer(
            question=question,
            answer=answer,
            chunks=candidate_chunks,
            llm_settings=llm_settings,
            preferred_indexes=self._extract_cited_candidate_numbers(answer, len(candidate_chunks)),
            retrieval_mode="llm_retrieval",
        )
        result.update(
            {
                "answer": answer,
                "citations": citations,
                "llm_used": bool(answer_parts),
                "llm_error": llm_error,
            }
        )
        yield {"type": "done", "result": self._attach_thinking_summary(result, llm_settings)}

    def _stream_hybrid_result(
        self,
        question: str,
        recent_conversation: str,
        memory_hits: list[dict],
        effective_question: str,
        llm_settings: LLMSettings,
        scope_reason: str,
        base_result: dict,
        top_k: int | None,
    ):
        retrieval_plan = self._plan_retrieval_strategy(
            question=question,
            effective_question=effective_question,
            recent_conversation=recent_conversation,
            memory_hits=memory_hits,
            llm_settings=llm_settings,
        )
        chunks = self._retrieve_relevant_chunks(
            question=question,
            retrieval_plan=retrieval_plan,
            llm_settings=llm_settings,
            retrieval_mode="hybrid",
            top_k=top_k or self.config.final_top_k,
            recent_conversation=recent_conversation,
            memory_hits=memory_hits,
        )
        result = dict(base_result)
        result.update(
            {
                "effective_question": retrieval_plan.get("retrieval_query", effective_question),
                "retrieved_chunks": chunks,
                "retrieval_mode": "hybrid",
                "conversation_scope": "legal",
                "scope_reason": scope_reason,
            }
        )
        if not chunks:
            answer = self._fallback_answer(
                question,
                chunks,
                recent_conversation=recent_conversation,
                memory_hits=memory_hits,
            )
            for token in self._stream_text(answer):
                yield {"type": "token", "content": token}
            result.update(
                {
                    "answer": answer,
                    "citations": [],
                    "llm_used": False,
                    "llm_error": "",
                }
            )
            yield {"type": "done", "result": self._attach_thinking_summary(result, llm_settings)}
            return

        answer_parts: list[str] = []
        llm_error = ""
        try:
            for token in self._stream_llm(
                question=question,
                recent_conversation=recent_conversation,
                memory_hits=memory_hits,
                chunks=chunks,
                llm_settings=llm_settings,
            ):
                answer_parts.append(token)
                yield {"type": "token", "content": token}
        except Exception as exc:
            llm_error = self._format_llm_error(exc)
            if not answer_parts:
                answer = self._fallback_answer(
                    question,
                    chunks,
                    recent_conversation=recent_conversation,
                    memory_hits=memory_hits,
                    llm_error=llm_error,
                )
                for token in self._stream_text(answer):
                    yield {"type": "token", "content": token}
            else:
                answer = "".join(answer_parts).strip()
        else:
            answer = "".join(answer_parts).strip()

        if llm_settings.enabled and answer:
            try:
                answer = self._self_check_answer(
                    question=question,
                    answer=answer,
                    recent_conversation=recent_conversation,
                    memory_hits=memory_hits,
                    chunks=chunks,
                    llm_settings=llm_settings,
                    retrieval_mode="hybrid",
                )
                answer = self._ensure_answer_contains_law_content(
                    answer=answer,
                    chunks=chunks,
                )
            except Exception as exc:
                llm_error = llm_error or self._format_llm_error(exc)

        answer = self._ensure_answer_contains_law_content(
            answer=answer,
            chunks=chunks,
        )
        citations = self._select_citations_for_answer(
            question=question,
            answer=answer,
            chunks=chunks,
            llm_settings=llm_settings,
            preferred_indexes=self._extract_cited_candidate_numbers(answer, len(chunks)),
            retrieval_mode="hybrid",
        )
        result.update(
            {
                "answer": answer,
                "citations": citations,
                "llm_used": bool(answer_parts),
                "llm_error": llm_error,
            }
        )
        yield {"type": "done", "result": self._attach_thinking_summary(result, llm_settings)}

    def _classify_question_scope(
        self,
        question: str,
        recent_conversation: str,
        memory_hits: list[dict],
        llm_settings: LLMSettings,
        usage_tracker: dict[str, int] | None = None,
    ) -> dict[str, str]:
        if llm_settings.enabled:
            memory_context = self._format_memory_context(memory_hits)
            prompt = ChatPromptTemplate.from_messages(
                [
                    (
                        "system",
                        "你是问题路由器。"
                        "请判断当前输入是法律问题还是通用对话。"
                        "legal：需要结合法律法规、法条、合规、责任、处罚、合同、劳动、隐私、税务、安全生产等法律知识来回答。"
                        "general：普通聊天、写作、闲聊、生活常识、技术问题、情绪表达、寒暄等，不需要法律检索。"
                        "只输出 JSON，格式为 {{\"scope\":\"legal\"}} 或 {{\"scope\":\"general\"}}。",
                    ),
                    (
                        "human",
                        "最近 3 轮对话上下文：\n{recent_conversation}\n\n"
                        "更早历史记忆 RAG：\n{memory_context}\n\n"
                        "当前输入：{question}\n\n"
                        "请判断 scope。",
                    ),
                ]
            )
            try:
                content = self._invoke_json_prompt(
                    prompt=prompt,
                    payload={
                        "question": question,
                        "recent_conversation": recent_conversation,
                        "memory_context": memory_context,
                    },
                    llm_settings=llm_settings,
                    max_tokens=80,
                    usage_tracker=usage_tracker,
                )
                payload = self._parse_json_object(content)
                scope = str((payload or {}).get("scope", "")).strip().lower()
                if scope in {"legal", "general"}:
                    return {"scope": scope, "reason": "llm_router"}
            except Exception:
                pass

        return {
            "scope": self._heuristic_question_scope(
                question=question,
                recent_conversation=recent_conversation,
                memory_hits=memory_hits,
            ),
            "reason": "heuristic_router",
        }

    def _call_general_chat_llm(
        self,
        question: str,
        recent_conversation: str,
        memory_hits: list[dict],
        llm_settings: LLMSettings,
        usage_tracker: dict[str, int] | None = None,
    ) -> str:
        memory_context = self._format_memory_context(memory_hits)
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "你是通用中文助手。"
                    "当前输入已被判定为非法律检索问题。"
                    "请直接进行普通对话回答，不要引用法律条文，不要假装检索过本地法律知识库。"
                    "如果用户实际在追问之前的非法律内容，优先承接最近上下文。"
                    "回答保持简洁、自然、直接。",
                ),
                (
                    "human",
                    "最近 3 轮对话上下文：\n{recent_conversation}\n\n"
                    "更早历史记忆：\n{memory_context}\n\n"
                    "当前输入：{question}",
                ),
            ]
        )
        content = self._invoke_text_prompt(
            prompt=prompt,
            payload={
                "question": question,
                "recent_conversation": recent_conversation,
                "memory_context": memory_context,
            },
            llm_settings=llm_settings,
            max_tokens=max(240, min(llm_settings.max_tokens, 900)),
            temperature=llm_settings.temperature,
            usage_tracker=usage_tracker,
        )
        return content.strip()

    def _stream_general_chat_llm(
        self,
        question: str,
        recent_conversation: str,
        memory_hits: list[dict],
        llm_settings: LLMSettings,
    ):
        memory_context = self._format_memory_context(memory_hits)
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "你是通用中文助手。"
                    "当前输入已被判定为非法律检索问题。"
                    "请直接进行普通对话回答，不要引用法律条文，不要假装检索过本地法律知识库。"
                    "如果用户实际在追问之前的非法律内容，优先承接最近上下文。"
                    "回答保持简洁、自然、直接。",
                ),
                (
                    "human",
                    "最近 3 轮对话上下文：\n{recent_conversation}\n\n"
                    "更早历史记忆：\n{memory_context}\n\n"
                    "当前输入：{question}",
                ),
            ]
        )
        messages = prompt.invoke(
            {
                "question": question,
                "recent_conversation": recent_conversation,
                "memory_context": memory_context,
            }
        ).messages
        yield from self._stream_serialized_messages(
            messages=self._serialize_prompt_messages(messages),
            llm_settings=llm_settings,
            temperature=llm_settings.temperature,
            max_tokens=max(240, min(llm_settings.max_tokens, 900)),
        )

    @staticmethod
    def _heuristic_question_scope(
        question: str,
        recent_conversation: str,
        memory_hits: list[dict],
    ) -> str:
        text = " ".join(question.split())
        if not text:
            return "general"

        legal_pattern = re.compile(
            r"(法律|法条|法规|条例|规定|刑法|民法|合同|劳动|仲裁|诉讼|起诉|判决|赔偿|侵权|违法|犯罪|"
            r"正当防卫|防卫过当|责任|处罚|罚款|合规|个人信息|隐私|税|增值税|安全生产|工伤|离职|社保|"
            r"试用期|解除劳动合同|用人单位|通知用人单位|道路交通|交通安全|交通管理|公安机关|"
            r"公安机关交通管理部门|驾驶证|机动车)"
        )
        if legal_pattern.search(text):
            return "legal"

        if len(text) <= 32 and re.search(r"(这个|那个|继续|展开|刚才|上面|前面|那如果|那这个)", text):
            recent_context = recent_conversation
            if not recent_context and memory_hits:
                recent_context = " ".join(
                    [str(memory_hits[0].get("question", "")), str(memory_hits[0].get("answer", ""))]
                )
            if legal_pattern.search(recent_context):
                return "legal"

        return "general"

    @staticmethod
    def _create_llm_client(llm_settings: LLMSettings) -> OpenAI:
        if not llm_settings.enabled:
            reason = llm_settings.disabled_reason or "LLM 配置无效。"
            raise ValueError(reason)
        return OpenAI(
            base_url=llm_settings.base_url,
            api_key=llm_settings.api_key,
            timeout=90.0,
        )

    def _call_llm_retrieval(
        self,
        question: str,
        recent_conversation: str,
        memory_hits: list[dict],
        chunks: list[dict],
        llm_settings: LLMSettings,
        usage_tracker: dict[str, int] | None = None,
    ) -> str:
        answer = self._invoke_serialized_messages(
            messages=self._build_llm_retrieval_messages(
                question=question,
                recent_conversation=recent_conversation,
                memory_hits=memory_hits,
                chunks=chunks,
            ),
            llm_settings=llm_settings,
            temperature=llm_settings.temperature,
            max_tokens=llm_settings.max_tokens,
            usage_tracker=usage_tracker,
        ).strip()
        checked_answer = self._self_check_answer(
            question=question,
            answer=answer,
            recent_conversation=recent_conversation,
            memory_hits=memory_hits,
            chunks=chunks,
            llm_settings=llm_settings,
            retrieval_mode="llm_retrieval",
            usage_tracker=usage_tracker,
        )
        return self._ensure_answer_contains_law_content(
            answer=checked_answer,
            chunks=chunks,
        )

    def _stream_llm_retrieval(
        self,
        question: str,
        recent_conversation: str,
        memory_hits: list[dict],
        chunks: list[dict],
        llm_settings: LLMSettings,
    ):
        yield from self._stream_serialized_messages(
            messages=self._build_llm_retrieval_messages(
                question=question,
                recent_conversation=recent_conversation,
                memory_hits=memory_hits,
                chunks=chunks,
            ),
            llm_settings=llm_settings,
            temperature=llm_settings.temperature,
            max_tokens=llm_settings.max_tokens,
        )

    def _build_llm_retrieval_messages(
        self,
        question: str,
        recent_conversation: str,
        memory_hits: list[dict],
        chunks: list[dict],
    ) -> list[dict[str, str]]:
        candidate_context = "\n\n".join(
            [
                f"[{idx}] 来源：{chunk['source_name']}\n内容：{chunk['text'][:900]}"
                for idx, chunk in enumerate(chunks, start=1)
            ]
        )
        memory_context = self._format_memory_context(memory_hits)
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "你是法律知识库中的证据筛选与回答助手。当前模式为“大模型检索”。"
                    "你必须只依据给定的本地候选证据作答，并从候选中自行挑选最相关的证据。"
                    "如果候选中存在与当前事件或问题无关的条文，必须直接忽略，不得引用、不得讨论。"
                    "如果某条候选最终不相关，不得在答案中写出它的编号、来源或内容。"
                    "不得伪造本地法条引用、不得编造未给出的来源编号。"
                    "关键依据必须明确到具体法律条款，并在条款内容处使用 Markdown 加粗。",
                ),
                (
                    "human",
                    "最近 3 轮对话上下文（优先用于承接指代）：\n{conversation_context}\n\n"
                    "更早历史记忆 RAG（仅作补充背景，不能替代法律依据）：\n{memory_context}\n\n"
                    "本地候选证据：\n{candidate_context}\n\n"
                    "当前问题：{question}\n\n"
                    "请按以下结构回答：\n"
                    "1. 直接回答\n"
                    "2. 依据与推理\n"
                    "3. 风险与不确定性\n"
                    "4. 建议如何进一步核验\n\n"
                    "引用要求：只在正文中使用候选证据编号引用，例如 [1]、[3][5]。\n"
                    "法条展示要求：在“依据与推理”中逐条写成“《法律名称》第X条：**条文关键句** [n]”。\n"
                    "如无可直接支持的具体条款，必须明确写“未检索到可直接支持的具体法条”，不得编造。",
                ),
            ]
        )
        messages = prompt.invoke(
            {
                "question": question,
                "conversation_context": recent_conversation,
                "memory_context": memory_context,
                "candidate_context": candidate_context,
            }
        ).messages
        return self._serialize_prompt_messages(messages)

    def _retrieve_llm_candidates(self, question: str, top_k: int = 12) -> list[dict]:
        chunks = self.retriever.retrieve_lexical(question, top_k=top_k)
        return [
            {
                "chunk_id": chunk.chunk_id,
                "chunk_index": getattr(chunk, "chunk_index", 0),
                "score": chunk.score,
                "source_name": chunk.source_name,
                "source_path": chunk.source_path,
                "title": chunk.title,
                "text": chunk.text,
                "metadata": chunk.metadata,
                "file_type": chunk.file_type,
            }
            for chunk in chunks
        ]

    def _plan_retrieval_strategy(
        self,
        question: str,
        effective_question: str,
        recent_conversation: str,
        memory_hits: list[dict],
        llm_settings: LLMSettings,
        usage_tracker: dict[str, int] | None = None,
    ) -> dict:
        rewritten = effective_question
        domains: list[str] = []
        issues: list[str] = []
        queries: list[str] = []

        if llm_settings.enabled:
            rewritten = (
                self._rewrite_question_for_legal_retrieval(
                    question=question,
                    effective_question=effective_question,
                    recent_conversation=recent_conversation,
                    memory_hits=memory_hits,
                    llm_settings=llm_settings,
                    usage_tracker=usage_tracker,
                )
                or effective_question
            )
            domains = self._route_legal_domains(
                question=question,
                effective_question=rewritten,
                recent_conversation=recent_conversation,
                memory_hits=memory_hits,
                llm_settings=llm_settings,
                usage_tracker=usage_tracker,
            )
            issues, queries = self._decompose_legal_issues(
                question=question,
                effective_question=rewritten,
                recent_conversation=recent_conversation,
                memory_hits=memory_hits,
                domains=domains,
                llm_settings=llm_settings,
                usage_tracker=usage_tracker,
            )

        query_variants: list[str] = []
        for candidate in [rewritten, *queries, question, effective_question]:
            normalized = " ".join(str(candidate).split()).strip()
            if normalized and normalized not in query_variants:
                query_variants.append(normalized)

        return {
            "retrieval_query": rewritten,
            "query_variants": query_variants,
            "domains": domains,
            "issues": issues,
        }

    def _rewrite_question_for_legal_retrieval(
        self,
        question: str,
        effective_question: str,
        recent_conversation: str,
        memory_hits: list[dict],
        llm_settings: LLMSettings,
        usage_tracker: dict[str, int] | None = None,
    ) -> str | None:
        memory_context = self._format_memory_context(memory_hits)
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "你是法律检索改写器。"
                    "请把用户问题改写成更适合法律条文检索的表达。"
                    "要求：保留原始事实，不得擅自补充结论；把口语或日常表达改成法律中更常见、同等意义的词汇；"
                    "优先提炼行为、主体、责任类型、法律争点、可能的法条术语。"
                    "只输出一行改写后的检索语句，不要解释。",
                ),
                (
                    "human",
                    "最近 3 轮对话上下文：\n{recent_conversation}\n\n"
                    "更早历史记忆 RAG：\n{memory_context}\n\n"
                    "当前问题：{question}\n\n"
                    "当前上下文扩展版问题：{effective_question}\n\n"
                    "请输出适合法律检索的改写语句。",
                ),
            ]
        )
        messages = prompt.invoke(
            {
                "question": question,
                "effective_question": effective_question,
                "recent_conversation": recent_conversation,
                "memory_context": memory_context,
            }
        ).messages
        client = self._create_llm_client(llm_settings)
        try:
            response = client.chat.completions.create(
                model=llm_settings.model,
                temperature=0,
                max_tokens=160,
                stream=False,
                messages=[
                    {
                        "role": {
                            "human": "user",
                            "ai": "assistant",
                            "system": "system",
                        }.get(message.type, message.type),
                        "content": message.content,
                    }
                    for message in messages
                ],
            )
        except Exception:
            return None
        _record_response_usage(usage_tracker, response)

        rewritten = " ".join((response.choices[0].message.content or "").split())
        if not rewritten:
            return None
        if rewritten.startswith("```"):
            rewritten = re.sub(r"^```[\w-]*\s*|\s*```$", "", rewritten).strip()
        if len(rewritten) < 6:
            return None
        return rewritten[:320]

    def _route_legal_domains(
        self,
        question: str,
        effective_question: str,
        recent_conversation: str,
        memory_hits: list[dict],
        llm_settings: LLMSettings,
        usage_tracker: dict[str, int] | None = None,
    ) -> list[str]:
        memory_context = self._format_memory_context(memory_hits)
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "你是法律法域路由器。"
                    "请判断当前问题最相关的法域。"
                    "可选值只有：criminal, civil, labor, privacy, safety, tax, general。"
                    "只输出 JSON，格式为 {{\"domains\":[\"criminal\",\"civil\"]}}。",
                ),
                (
                    "human",
                    "最近 3 轮对话上下文：\n{recent_conversation}\n\n"
                    "更早历史记忆 RAG：\n{memory_context}\n\n"
                    "当前问题：{question}\n\n"
                    "检索改写问题：{effective_question}\n\n"
                    "请返回最相关的 1 到 3 个法域。",
                ),
            ]
        )
        try:
            content = self._invoke_json_prompt(
                prompt=prompt,
                payload={
                    "question": question,
                    "effective_question": effective_question,
                    "recent_conversation": recent_conversation,
                    "memory_context": memory_context,
                },
                llm_settings=llm_settings,
                max_tokens=120,
                usage_tracker=usage_tracker,
            )
        except Exception:
            return []
        payload = self._parse_json_object(content)
        raw_domains = payload.get("domains", []) if payload else []
        domains: list[str] = []
        seen: set[str] = set()
        for domain in raw_domains:
            key = str(domain).strip().lower()
            if key in DOMAIN_SOURCE_HINTS and key not in seen:
                seen.add(key)
                domains.append(key)
        return domains[:3]

    def _decompose_legal_issues(
        self,
        question: str,
        effective_question: str,
        recent_conversation: str,
        memory_hits: list[dict],
        domains: list[str],
        llm_settings: LLMSettings,
        usage_tracker: dict[str, int] | None = None,
    ) -> tuple[list[str], list[str]]:
        memory_context = self._format_memory_context(memory_hits)
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "你是法律争点拆解器。"
                    "请把问题拆成 1 到 3 个检索争点，并给出每个争点对应的一句法律检索语句。"
                    "不得编造事实，只能重述现有问题中的事实和争点。"
                    "只输出 JSON，格式为 {{\"issues\":[\"...\"],\"queries\":[\"...\"]}}。",
                ),
                (
                    "human",
                    "最近 3 轮对话上下文：\n{recent_conversation}\n\n"
                    "更早历史记忆 RAG：\n{memory_context}\n\n"
                    "当前问题：{question}\n\n"
                    "检索改写问题：{effective_question}\n\n"
                    "法域路由结果：{domains}\n\n"
                    "请拆解争点并生成检索语句。",
                ),
            ]
        )
        try:
            content = self._invoke_json_prompt(
                prompt=prompt,
                payload={
                    "question": question,
                    "effective_question": effective_question,
                    "recent_conversation": recent_conversation,
                    "memory_context": memory_context,
                    "domains": ", ".join(domains) if domains else "general",
                },
                llm_settings=llm_settings,
                max_tokens=220,
                usage_tracker=usage_tracker,
            )
        except Exception:
            return [], []
        payload = self._parse_json_object(content)
        if not payload:
            return [], []
        issues = [
            " ".join(str(item).split())[:120]
            for item in payload.get("issues", [])
            if str(item).strip()
        ][:3]
        queries = [
            " ".join(str(item).split())[:220]
            for item in payload.get("queries", [])
            if str(item).strip()
        ][:3]
        return issues, queries

    def _retrieve_relevant_chunks(
        self,
        question: str,
        retrieval_plan: dict,
        llm_settings: LLMSettings,
        retrieval_mode: str,
        top_k: int,
        recent_conversation: str,
        memory_hits: list[dict],
    ) -> list[dict]:
        target_top_k = max(int(top_k or self.config.final_top_k), 1)
        candidate_sizes: list[int] = []
        for size in (
            target_top_k,
            max(target_top_k * 2, 8),
            max(target_top_k * 3, 12),
        ):
            if size not in candidate_sizes:
                candidate_sizes.append(size)

        query_variants = retrieval_plan.get("query_variants", []) or [question.strip()]
        domains = retrieval_plan.get("domains", []) or []

        for query_variant in query_variants:
            for candidate_k in candidate_sizes:
                candidate_chunks = self._retrieve_candidate_chunks_for_mode(
                    query_variant,
                    retrieval_mode=retrieval_mode,
                    top_k=candidate_k,
                )
                candidate_chunks = self._prioritize_domain_chunks(candidate_chunks, domains)
                filtered_chunks = self._filter_relevant_chunks(
                    question=question,
                    chunks=candidate_chunks,
                    llm_settings=llm_settings,
                    recent_conversation=recent_conversation,
                    memory_hits=memory_hits,
                )
                if filtered_chunks:
                    return filtered_chunks[:target_top_k]
        return []

    @staticmethod
    def _prioritize_domain_chunks(chunks: list[dict], domains: list[str]) -> list[dict]:
        if not domains:
            return chunks
        scored: list[tuple[int, dict]] = []
        for index, chunk in enumerate(chunks):
            source_text = f"{chunk.get('source_name', '')} {chunk.get('title', '')}"
            score = 0
            for domain in domains:
                hints = DOMAIN_SOURCE_HINTS.get(domain, ())
                if hints and any(hint in source_text for hint in hints):
                    score += 1
            scored.append((score, chunk))
        scored.sort(key=lambda item: item[0], reverse=True)
        return [chunk for _, chunk in scored]

    def _retrieve_candidate_chunks_for_mode(
        self,
        query: str,
        retrieval_mode: str,
        top_k: int,
    ) -> list[dict]:
        if retrieval_mode == "llm_retrieval":
            return self._retrieve_llm_candidates(query, top_k=top_k)
        return self._retrieve_chunks(query, top_k=top_k)

    def _filter_relevant_chunks(
        self,
        question: str,
        chunks: list[dict],
        llm_settings: LLMSettings,
        recent_conversation: str,
        memory_hits: list[dict],
    ) -> list[dict]:
        if not chunks:
            return []

        if llm_settings.enabled:
            candidate_pool = chunks[:10]
            selected_indexes = self._llm_select_relevant_chunk_indexes(
                question=question,
                recent_conversation=recent_conversation,
                memory_hits=memory_hits,
                chunks=candidate_pool,
                llm_settings=llm_settings,
            )
            if selected_indexes is not None:
                selected_chunks = [
                    candidate_pool[index - 1]
                    for index in selected_indexes
                    if 1 <= index <= len(candidate_pool)
                ]
                return selected_chunks

        heuristic_chunks = self._heuristic_filter_relevant_chunks(question, chunks)
        return heuristic_chunks[: self.config.final_top_k]

    def _heuristic_filter_relevant_chunks(
        self,
        question: str,
        chunks: list[dict],
    ) -> list[dict]:
        query_terms = self._extract_focus_terms(question)
        preferred_terms = extract_priority_legal_terms(question)
        article_refs = extract_article_refs(question)
        law_names = extract_law_names(question)

        scored_chunks: list[tuple[float, dict]] = []
        for chunk in chunks:
            score = self._score_chunk_relevance(
                question=question,
                chunk=chunk,
                query_terms=query_terms,
                preferred_terms=preferred_terms,
                article_refs=article_refs,
                law_names=law_names,
            )
            if score > 0:
                scored_chunks.append((score, chunk))

        if not scored_chunks:
            return []

        scored_chunks.sort(key=lambda item: item[0], reverse=True)
        threshold = 0.2 if preferred_terms else 0.14
        selected = [chunk for score, chunk in scored_chunks if score >= threshold]
        if not selected and scored_chunks and scored_chunks[0][0] >= threshold * 0.85:
            selected = [scored_chunks[0][1]]
        return selected[:8]

    @staticmethod
    def _extract_focus_terms(question: str) -> set[str]:
        focus_terms: set[str] = set()
        focus_terms.update(extract_priority_legal_terms(question))
        focus_terms.update(extract_law_names(question))
        focus_terms.update(extract_article_refs(question))
        generic_terms = {
            "责任",
            "义务",
            "行为",
            "情节",
            "规定",
            "承担",
            "造成",
            "管理",
            "违法",
            "他人",
            "可以",
            "应当",
            "平台",
            "网络",
        }
        raw_terms = extract_query_terms(question)
        for term in raw_terms:
            text = str(term).strip()
            if len(text) < 2 or len(text) > 10:
                continue
            if text in generic_terms:
                continue
            if len(text) < 3 and text not in {"侮辱", "诽谤", "强奸", "防卫", "换脸"}:
                continue
            if re.search(r"(罪|权|法|责|义务|侵权|侮辱|诽谤|强奸|防卫|平台|审核|网络|通知|删除|处罚|劳动|合同|试用|离职|用人|工资|工伤|社保|仲裁|道路|交通|公安|机关|驾驶|车辆|安全)", text):
                focus_terms.add(text)
        if not focus_terms:
            focus_terms = raw_terms
        return focus_terms

    @staticmethod
    def _score_chunk_relevance(
        question: str,
        chunk: dict,
        query_terms: set[str],
        preferred_terms: list[str],
        article_refs: set[str],
        law_names: list[str],
    ) -> float:
        text = chunk.get("text", "")
        metadata = chunk.get("metadata", {}) or {}
        coverage = sum(1 for term in query_terms if term in text)
        coverage_ratio = coverage / max(len(query_terms), 8)
        preferred_hits = sum(1 for term in preferred_terms if term in text)
        early_window = text[:180]
        early_hits = sum(1 for term in preferred_terms[:6] if term in early_window)
        article_anchor = str(metadata.get("article_anchor", "") or "")
        article_match = bool(article_refs and article_anchor in article_refs)
        law_match = bool(
            law_names
            and any(
                name in str(chunk.get("source_name", "")) or name in str(chunk.get("title", ""))
                for name in law_names
            )
        )

        score = min(coverage_ratio, 0.65)
        score += min(preferred_hits, 4) * 0.18
        score += min(early_hits, 3) * 0.05
        if article_match:
            score += 0.55
        if law_match:
            score += 0.18
        if metadata.get("law_chunk_type") == "article":
            score += 0.04
        if preferred_terms and preferred_hits == 0:
            score -= 0.28
            if not law_match and not article_match:
                score -= 0.18
        if any(term in {"侮辱", "诽谤", "侮辱罪", "诽谤罪"} for term in preferred_terms):
            if not re.search(r"(侮辱|诽谤)", text):
                score -= 0.35
        if any(term in {"审核义务", "通知删除", "网络服务提供者", "知道或者应当知道"} for term in preferred_terms):
            if not re.search(r"(网络服务|通知|删除|应当知道|平台|连带责任)", text):
                score -= 0.2
        if article_refs and not article_match:
            score -= 0.12
        if re.search(r"(是什么|定义|含义|适用范围|规定了什么|主要规范)", question) and coverage == 0:
            score -= 0.12
        return score

    def _llm_select_relevant_chunk_indexes(
        self,
        question: str,
        recent_conversation: str,
        memory_hits: list[dict],
        chunks: list[dict],
        llm_settings: LLMSettings,
        usage_tracker: dict[str, int] | None = None,
    ) -> list[int] | None:
        memory_context = self._format_memory_context(memory_hits)
        candidate_context = "\n\n".join(
            [
                f"[{idx}] 来源：{chunk['source_name']}\n内容：{chunk['text'][:700]}"
                for idx, chunk in enumerate(chunks, start=1)
            ]
        )
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "你是法律证据相关性筛选器。"
                    "你的任务是只保留与当前事件、行为、法律争点直接相关的法条。"
                    "相邻条文、泛化责任条款、目录、定义页、与当前行为无直接关系的法条都必须排除。"
                    "但不要误删直接回答行为性质、构成要件、责任后果、抗辩事由、处罚后果的条文。"
                    "只要某条法条能直接支撑当前问题的结论或判断边界，就应保留。"
                    "只输出 JSON，格式为 {{\"relevant\":[1,2]}}，如果都不相关则输出 {{\"relevant\":[]}}。",
                ),
                (
                    "human",
                    "最近 3 轮对话上下文：\n{recent_conversation}\n\n"
                    "更早历史记忆 RAG：\n{memory_context}\n\n"
                    "当前问题：{question}\n\n"
                    "候选法条：\n{candidate_context}\n\n"
                    "请只返回 JSON。",
                ),
            ]
        )
        messages = prompt.invoke(
            {
                "question": question,
                "recent_conversation": recent_conversation,
                "memory_context": memory_context,
                "candidate_context": candidate_context,
            }
        ).messages
        client = self._create_llm_client(llm_settings)
        try:
            response = client.chat.completions.create(
                model=llm_settings.model,
                temperature=0,
                max_tokens=180,
                stream=False,
                messages=[
                    {
                        "role": {
                            "human": "user",
                            "ai": "assistant",
                            "system": "system",
                        }.get(message.type, message.type),
                        "content": message.content,
                    }
                    for message in messages
                ],
            )
        except Exception:
            return None
        _record_response_usage(usage_tracker, response)

        content = response.choices[0].message.content or ""
        return self._parse_relevant_index_response(content, len(chunks))

    @staticmethod
    def _parse_relevant_index_response(content: str, limit: int) -> list[int] | None:
        text = content.strip()
        if not text:
            return None

        payload = None
        match = re.search(r"\{.*\}", text, re.S)
        if match:
            try:
                payload = json.loads(match.group(0))
            except json.JSONDecodeError:
                payload = None
        if payload is None:
            if re.search(r"relevant\s*[:=]\s*\[\s*\]", text, re.I) or text in {"[]", "无", "none", "None"}:
                return []
            numbers = re.findall(r"\d+", text)
        else:
            numbers = payload.get("relevant", [])

        ordered: list[int] = []
        seen: set[int] = set()
        for raw in numbers:
            try:
                index = int(raw)
            except (TypeError, ValueError):
                continue
            if index < 1 or index > limit or index in seen:
                continue
            seen.add(index)
            ordered.append(index)
        return ordered

    def _self_check_answer(
        self,
        question: str,
        answer: str,
        recent_conversation: str,
        memory_hits: list[dict],
        chunks: list[dict],
        llm_settings: LLMSettings,
        retrieval_mode: str,
        usage_tracker: dict[str, int] | None = None,
    ) -> str:
        if not llm_settings.enabled or not answer.strip():
            return answer
        memory_context = self._format_memory_context(memory_hits)
        evidence_context = "\n\n".join(
            [
                f"[{idx}] 来源：{chunk['source_name']}\n内容：{chunk['text'][:800]}"
                for idx, chunk in enumerate(chunks, start=1)
            ]
        )
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "你是法律答案自检器。"
                    "请检查答案是否超出证据、是否遗漏关键限制条件、是否把不确定事实说成确定结论。"
                    "如果原答案没有问题，就原样返回；如果有问题，就在不脱离证据的前提下修正后返回。"
                    "只输出修正后的最终答案，不要解释你做了什么。"
                    "如果答案中有法律依据，必须保留并规范为“《法律名称》第X条：**条文关键句** [n]”格式。",
                ),
                (
                    "human",
                    "检索模式：{retrieval_mode}\n\n"
                    "最近 3 轮对话上下文：\n{recent_conversation}\n\n"
                    "更早历史记忆 RAG：\n{memory_context}\n\n"
                    "当前问题：{question}\n\n"
                    "候选证据：\n{evidence_context}\n\n"
                    "原答案：\n{answer}\n\n"
                    "请输出最终可交付答案。"
                    "若原答案未写明具体法条，请在“关键依据”补上法律名称与条次，并将对应条文关键句加粗。",
                ),
            ]
        )
        try:
            content = self._invoke_text_prompt(
                prompt=prompt,
                payload={
                    "retrieval_mode": retrieval_mode,
                    "recent_conversation": recent_conversation,
                    "memory_context": memory_context,
                    "question": question,
                    "evidence_context": evidence_context,
                    "answer": answer,
                },
                llm_settings=llm_settings,
                max_tokens=max(300, min(llm_settings.max_tokens, 900)),
                usage_tracker=usage_tracker,
            )
        except Exception:
            return answer
        checked = content.strip()
        return checked or answer

    def _select_citations_for_answer(
        self,
        question: str,
        answer: str,
        chunks: list[dict],
        llm_settings: LLMSettings,
        preferred_indexes: list[int] | None = None,
        retrieval_mode: str = "hybrid",
        usage_tracker: dict[str, int] | None = None,
    ) -> list[dict]:
        if not chunks:
            return []

        selected_indexes: list[int] | None = None
        if llm_settings.enabled:
            selected_indexes = self._validate_citation_indexes(
                question=question,
                answer=answer,
                chunks=chunks,
                llm_settings=llm_settings,
                usage_tracker=usage_tracker,
            )

        if selected_indexes is None and preferred_indexes:
            selected_indexes = preferred_indexes

        if selected_indexes is None:
            selected_indexes = list(range(1, min(len(chunks), 3) + 1))

        selected_chunks = [
            chunks[index - 1]
            for index in selected_indexes
            if 1 <= index <= len(chunks)
        ]
        return self._build_citations(selected_chunks, question)

    def _validate_citation_indexes(
        self,
        question: str,
        answer: str,
        chunks: list[dict],
        llm_settings: LLMSettings,
        usage_tracker: dict[str, int] | None = None,
    ) -> list[int] | None:
        candidate_context = "\n\n".join(
            [
                f"[{idx}] 来源：{chunk['source_name']}\n内容：{chunk['text'][:700]}"
                for idx, chunk in enumerate(chunks, start=1)
            ]
        )
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "你是法律引用一致性校验器。"
                    "请只保留能够直接支撑最终答案核心判断的证据编号。"
                    "不能因为候选证据看起来相关就保留，必须与最终答案中的判断有直接支撑关系。"
                    "只输出 JSON，格式为 {{\"citations\":[1,2]}}；如果没有可直接支撑的证据，则输出 {{\"citations\":[]}}。",
                ),
                (
                    "human",
                    "当前问题：{question}\n\n"
                    "最终答案：\n{answer}\n\n"
                    "候选证据：\n{candidate_context}\n\n"
                    "请返回需要保留的证据编号。",
                ),
            ]
        )
        try:
            content = self._invoke_json_prompt(
                prompt=prompt,
                payload={
                    "question": question,
                    "answer": answer,
                    "candidate_context": candidate_context,
                },
                llm_settings=llm_settings,
                max_tokens=160,
                usage_tracker=usage_tracker,
            )
        except Exception:
            return None
        payload = self._parse_json_object(content)
        if not payload:
            return None
        return self._normalize_index_list(payload.get("citations", []), len(chunks))

    def _invoke_json_prompt(
        self,
        prompt: ChatPromptTemplate,
        payload: dict,
        llm_settings: LLMSettings,
        max_tokens: int,
        usage_tracker: dict[str, int] | None = None,
    ) -> str:
        return self._invoke_text_prompt(
            prompt=prompt,
            payload=payload,
            llm_settings=llm_settings,
            max_tokens=max_tokens,
            usage_tracker=usage_tracker,
        )

    def _invoke_text_prompt(
        self,
        prompt: ChatPromptTemplate,
        payload: dict,
        llm_settings: LLMSettings,
        max_tokens: int,
        temperature: float = 0,
        usage_tracker: dict[str, int] | None = None,
    ) -> str:
        messages = prompt.invoke(payload).messages
        return self._invoke_serialized_messages(
            messages=self._serialize_prompt_messages(messages),
            llm_settings=llm_settings,
            temperature=temperature,
            max_tokens=max_tokens,
            usage_tracker=usage_tracker,
        )

    @staticmethod
    def _serialize_prompt_messages(messages: list) -> list[dict[str, str]]:
        return [
            {
                "role": {
                    "human": "user",
                    "ai": "assistant",
                    "system": "system",
                }.get(message.type, message.type),
                "content": message.content,
            }
            for message in messages
        ]

    @staticmethod
    def _build_continuation_messages(
        base_messages: list[dict[str, str]],
        accumulated_text: str,
    ) -> list[dict[str, str]]:
        return [
            *base_messages,
            {"role": "assistant", "content": accumulated_text},
            {"role": "user", "content": CONTINUE_OUTPUT_PROMPT},
        ]

    def _invoke_serialized_messages(
        self,
        messages: list[dict[str, str]],
        llm_settings: LLMSettings,
        temperature: float,
        max_tokens: int,
        usage_tracker: dict[str, int] | None = None,
    ) -> str:
        client = self._create_llm_client(llm_settings)
        base_messages = [dict(message) for message in messages]
        current_messages = [dict(message) for message in base_messages]
        parts: list[str] = []

        for _ in range(MAX_COMPLETION_SEGMENTS):
            response = client.chat.completions.create(
                model=llm_settings.model,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=False,
                messages=current_messages,
            )
            _record_response_usage(usage_tracker, response)
            if not response.choices:
                raise RuntimeError("LLM returned no choices.")
            choice = response.choices[0]
            content = choice.message.content or ""
            if content:
                parts.append(content)
            finish_reason = getattr(choice, "finish_reason", None)
            if finish_reason != "length":
                return "".join(parts)
            accumulated = "".join(parts).strip()
            if not accumulated:
                raise RuntimeError("LLM output was truncated before any content was returned.")
            current_messages = self._build_continuation_messages(base_messages, accumulated)

        raise RuntimeError("LLM output exceeded continuation limit.")

    def _stream_serialized_messages(
        self,
        messages: list[dict[str, str]],
        llm_settings: LLMSettings,
        temperature: float,
        max_tokens: int,
    ):
        client = self._create_llm_client(llm_settings)
        base_messages = [dict(message) for message in messages]
        current_messages = [dict(message) for message in base_messages]
        accumulated_parts: list[str] = []

        for _ in range(MAX_COMPLETION_SEGMENTS):
            finish_reason = None
            stream = client.chat.completions.create(
                model=llm_settings.model,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=True,
                messages=current_messages,
            )
            for event in stream:
                if not event.choices:
                    continue
                choice = event.choices[0]
                if getattr(choice, "finish_reason", None):
                    finish_reason = choice.finish_reason
                delta = getattr(choice, "delta", None)
                content = getattr(delta, "content", None)
                if content:
                    accumulated_parts.append(content)
                    yield content
            if finish_reason != "length":
                return
            accumulated = "".join(accumulated_parts).strip()
            if not accumulated:
                raise RuntimeError("LLM stream was truncated before any content was returned.")
            current_messages = self._build_continuation_messages(base_messages, accumulated)

        raise RuntimeError("LLM stream exceeded continuation limit.")

    @staticmethod
    def _parse_json_object(content: str) -> dict | None:
        text = (content or "").strip()
        if not text:
            return None
        match = re.search(r"\{.*\}", text, re.S)
        if not match:
            return None
        try:
            payload = json.loads(match.group(0))
        except json.JSONDecodeError:
            return None
        return payload if isinstance(payload, dict) else None

    @staticmethod
    def _normalize_index_list(values: list | tuple, limit: int) -> list[int]:
        ordered: list[int] = []
        seen: set[int] = set()
        for raw in values:
            try:
                index = int(raw)
            except (TypeError, ValueError):
                continue
            if index < 1 or index > limit or index in seen:
                continue
            seen.add(index)
            ordered.append(index)
        return ordered

    def _build_llm_retrieval_citations(
        self,
        chunks: list[dict],
        answer: str,
        question: str,
    ) -> list[dict]:
        cited_indexes = self._extract_cited_candidate_numbers(answer, len(chunks))
        selected_chunks = [chunks[index - 1] for index in cited_indexes] if cited_indexes else chunks[:3]
        return self._build_citations(selected_chunks, question)

    @staticmethod
    def _extract_cited_candidate_numbers(answer: str, limit: int) -> list[int]:
        seen: set[int] = set()
        ordered: list[int] = []
        for match in re.findall(r"\[(\d{1,2})\]", answer):
            idx = int(match)
            if idx < 1 or idx > limit or idx in seen:
                continue
            seen.add(idx)
            ordered.append(idx)
        return ordered

    def _call_llm(
        self,
        question: str,
        recent_conversation: str,
        memory_hits: list[dict],
        chunks: list[dict],
        llm_settings: LLMSettings,
        usage_tracker: dict[str, int] | None = None,
    ) -> str:
        answer = self._invoke_serialized_messages(
            messages=self._build_chat_messages(
                question=question,
                recent_conversation=recent_conversation,
                memory_hits=memory_hits,
                chunks=chunks,
            ),
            llm_settings=llm_settings,
            temperature=llm_settings.temperature,
            max_tokens=llm_settings.max_tokens,
            usage_tracker=usage_tracker,
        ).strip()
        checked_answer = self._self_check_answer(
            question=question,
            answer=answer,
            recent_conversation=recent_conversation,
            memory_hits=memory_hits,
            chunks=chunks,
            llm_settings=llm_settings,
            retrieval_mode="hybrid",
            usage_tracker=usage_tracker,
        )
        return self._ensure_answer_contains_law_content(
            answer=checked_answer,
            chunks=chunks,
        )

    def _stream_llm(
        self,
        question: str,
        recent_conversation: str,
        memory_hits: list[dict],
        chunks: list[dict],
        llm_settings: LLMSettings,
    ):
        yield from self._stream_serialized_messages(
            messages=self._build_chat_messages(
                question=question,
                recent_conversation=recent_conversation,
                memory_hits=memory_hits,
                chunks=chunks,
            ),
            llm_settings=llm_settings,
            temperature=llm_settings.temperature,
            max_tokens=llm_settings.max_tokens,
        )

    def _build_chat_messages(
        self,
        question: str,
        recent_conversation: str,
        memory_hits: list[dict],
        chunks: list[dict],
    ) -> list[dict[str, str]]:
        prompt_bundle = self._select_prompt_bundle(question)
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    prompt_bundle["system"] + self._law_citation_system_requirement(),
                ),
                (
                    "human",
                    prompt_bundle["human"] + self._law_citation_output_requirement(),
                ),
            ]
        )
        context = "\n\n".join(
            [
                f"[{idx + 1}] 来源：{chunk['source_name']}\n内容：{chunk['text'][:900]}"
                for idx, chunk in enumerate(chunks)
            ]
        )
        memory_context = self._format_memory_context(memory_hits)
        messages = prompt.invoke(
            {
                "question": question,
                "recent_conversation": recent_conversation,
                "memory_context": memory_context,
                "context": context,
            }
        ).messages
        return self._serialize_prompt_messages(messages)

    @staticmethod
    def _select_prompt_bundle(question: str) -> dict[str, str]:
        text = question.strip()

        if re.search(r"(合法么|合法性|违法|违规|合规|风险|处罚|责任|罚款|会不会被判|是否构成)", text):
            return {
                "system": (
                    "你是法律知识库中的合规分析助手。"
                    "只能依据检索证据作答，禁止虚构法条、案例或结论。"
                    "如果证据不足，只能说明不足，不能替用户下最终法律结论。"
                    "如果某条检索证据与当前事件无关，必须直接忽略，不得引用或写入答案。"
                ),
                "human": (
                    "问题：{question}\n\n"
                    "最近 3 轮对话上下文（优先用于承接指代）：\n{recent_conversation}\n\n"
                    "更早历史记忆 RAG（仅用于理解背景，不能替代法律依据）：\n{memory_context}\n\n"
                    "检索证据：\n{context}\n\n"
                    "请按以下结构回答：\n"
                    "1. 初步判断：明确说明基于现有证据能判断到什么程度\n"
                    "2. 法律依据：列出最关键的法条或规则点\n"
                    "3. 风险点：指出触发风险或处罚的条件\n"
                    "4. 不确定性：哪些事实缺失会影响判断"
                ),
            }

        if re.search(r"(怎么做|怎么办|流程|步骤|如何|申请|办理|提交|材料|程序)", text):
            return {
                "system": (
                    "你是法律知识库中的程序指引助手。"
                    "只能依据检索证据回答，不得补造流程。"
                    "如果检索证据没有给出完整程序，必须明确指出缺口。"
                    "与当前问题无关的证据必须忽略，不得引用或展开。"
                ),
                "human": (
                    "问题：{question}\n\n"
                    "最近 3 轮对话上下文（优先用于承接指代）：\n{recent_conversation}\n\n"
                    "更早历史记忆 RAG（仅用于理解背景，不能替代法律依据）：\n{memory_context}\n\n"
                    "检索证据：\n{context}\n\n"
                    "请按以下结构回答：\n"
                    "1. 直接结论\n"
                    "2. 操作步骤：按先后顺序整理\n"
                    "3. 关键条件或材料\n"
                    "4. 证据不足或地区差异说明"
                ),
            }

        if re.search(r"(是什么|含义|定义|解释|适用范围|主要规范|规定了什么|包括什么)", text):
            return {
                "system": (
                    "你是法律知识库中的法条解释助手。"
                    "只能根据检索证据解释概念、适用范围和规范内容，"
                    "禁止输出证据中不存在的扩展解释。"
                    "与当前问题无关的条文不得引用或写入答案。"
                ),
                "human": (
                    "问题：{question}\n\n"
                    "最近 3 轮对话上下文（优先用于承接指代）：\n{recent_conversation}\n\n"
                    "更早历史记忆 RAG（仅用于理解背景，不能替代法律依据）：\n{memory_context}\n\n"
                    "检索证据：\n{context}\n\n"
                    "请按以下结构回答：\n"
                    "1. 核心结论：先用简洁自然语言回答\n"
                    "2. 关键依据：提炼法条核心内容\n"
                    "3. 适用边界：说明适用对象、范围或例外\n"
                    "4. 不确定性：如果证据不足，明确指出"
                ),
            }

        return {
            "system": (
                "你是法律知识库助手。"
                "你不能虚构内容，只能基于给定检索证据回答；若证据不足，要明确说明。"
                "如果某条检索证据与当前问题无关，直接忽略，不得引用。"
            ),
            "human": (
                "问题：{question}\n\n"
                "最近 3 轮对话上下文（优先用于承接指代）：\n{recent_conversation}\n\n"
                "更早历史记忆 RAG（仅用于理解背景，不能替代法律依据）：\n{memory_context}\n\n"
                "检索证据：\n{context}\n\n"
                "请输出：1. 直接回答 2. 关键依据 3. 如有冲突说明不确定性。"
            ),
        }

    @staticmethod
    def _law_citation_system_requirement() -> str:
        return (
            " 回答中若引用法律依据，必须精确到“法律名称 + 第X条”。"
            " 对应条文内容必须使用 Markdown 加粗格式展示。"
            " 不得引用未在候选证据中出现的法条。"
        )

    @staticmethod
    def _law_citation_output_requirement() -> str:
        return (
            "\n\n法条格式强制要求："
            "\n- 在“关键依据/法律依据”部分按“《法律名称》第X条：**条文关键句** [n]”输出。"
            "\n- 至少列出 1 条与结论直接相关的具体法条；若确实没有，明确写“未检索到可直接支持的具体法条”。"
            "\n- 条文关键句必须来自给定检索证据，不得杜撰。"
        )

    @staticmethod
    def _extract_article_display_title(chunk: dict) -> str:
        metadata = chunk.get("metadata", {}) or {}
        source_name = str(chunk.get("source_name", "") or "").strip()
        title = str(chunk.get("title", "") or "").strip()
        article_anchor = str(metadata.get("article_anchor", "") or "").strip()
        law_name = ""
        for candidate in (source_name, title):
            candidate = candidate.strip()
            if not candidate:
                continue
            if "第" in candidate and "条" in candidate:
                continue
            law_name = candidate
            break
        if law_name and article_anchor:
            return f"《{law_name}》{article_anchor}"
        if law_name:
            return f"《{law_name}》"
        if title:
            return title
        return source_name or "相关法律依据"

    @staticmethod
    def _extract_law_quote_from_chunk(chunk: dict, max_length: int = 220) -> str:
        text = " ".join(str(chunk.get("text", "") or "").split())
        if not text:
            return ""
        metadata = chunk.get("metadata", {}) or {}
        article_anchor = str(metadata.get("article_anchor", "") or "").strip()
        if article_anchor and text.startswith(article_anchor):
            text = text[len(article_anchor):].lstrip("：: 　")
        text = re.sub(r"^第[一二三四五六七八九十百千0-9]+条[：:]?", "", text).strip()
        text = re.sub(r"^【[^】]{1,30}】", "", text).strip()
        text = text.replace("　", " ")
        text = re.sub(r"\s+", " ", text).strip(" ：:")
        if not text:
            return ""
        if len(text) > max_length:
            cut_points = [text.rfind(sep, 0, max_length) for sep in ("。", "；", ";", "，", ",")]
            cut = max(cut_points)
            if cut >= max_length // 2:
                text = text[: cut + 1]
            else:
                text = text[:max_length].rstrip()
            if not text.endswith(("。", "；")):
                text += "……"
        return text.strip()

    def _build_explicit_law_blocks(
        self,
        chunks: list[dict],
        cited_indexes: list[int] | None = None,
        limit: int = 3,
    ) -> list[str]:
        blocks: list[str] = []
        indexes = cited_indexes or list(range(1, min(len(chunks), limit) + 1))
        seen: set[tuple[str, str]] = set()
        for index in indexes:
            if index < 1 or index > len(chunks):
                continue
            chunk = chunks[index - 1]
            quote = self._extract_law_quote_from_chunk(chunk)
            if not quote:
                continue
            display_title = self._extract_article_display_title(chunk)
            key = (display_title, quote)
            if key in seen:
                continue
            seen.add(key)
            blocks.append(f"{display_title}：**{quote}** [{index}]")
            if len(blocks) >= limit:
                break
        return blocks

    def _ensure_answer_contains_law_content(self, answer: str, chunks: list[dict]) -> str:
        text = (answer or "").strip()
        if not text or not chunks:
            return text
        cited_indexes = self._extract_cited_candidate_numbers(text, len(chunks))
        explicit_blocks = self._build_explicit_law_blocks(chunks=chunks, cited_indexes=cited_indexes)
        if not explicit_blocks:
            return text

        has_explicit_law_line = bool(
            re.search(r"《[^》]+》[^\n]{0,20}第[^\n]{0,20}条：\*\*.+?\*\*\s*\[\d+\]", text)
            or re.search(r"《[^》]+》[^\n]{0,20}第[^\n]{0,20}条：[^\n]{8,}", text)
        )
        if has_explicit_law_line:
            return text

        has_law_section = "法律依据" in text or "关键依据" in text or "依据与推理" in text
        section_title = "法律条文摘录" if has_law_section else "补充法条依据"
        law_block = section_title + "：\n" + "\n".join(f"- {line}" for line in explicit_blocks)
        return text.rstrip() + "\n\n" + law_block

    def _fallback_answer(
        self,
        question: str,
        chunks: list[dict],
        recent_conversation: str = "无",
        memory_hits: list[dict] | None = None,
        llm_error: str = "",
    ) -> str:
        memory_hits = memory_hits or []
        if not chunks:
            message = "当前知识库里没有检索到足够相关的法律依据，请尝试补充更具体的事实、行为、主体或法条关键词。"
            if recent_conversation and recent_conversation != "无":
                message += "\n\n最近 3 轮对话已纳入上下文。"
            if memory_hits:
                message += f"\n\n已从更早历史记忆中召回 {len(memory_hits)} 条相关上下文，但仍未匹配到足够相关的法律证据。"
            if llm_error:
                message += f"\n\nLLM 整理阶段已跳过：{llm_error}"
            return message

        query_terms = extract_query_terms(question)
        lead = max(
            chunks,
            key=lambda chunk: sum(1 for term in query_terms if term in chunk["text"]),
        )
        lines = [
            f"问题：{question}",
            f"本地检索最相关的来源是《{lead['title']}》。",
            "我先给出基于检索证据的摘要：",
        ]
        if memory_hits:
            lines.insert(
                1,
                f"已从更早历史记忆库中召回 {len(memory_hits)} 条相关上下文。",
            )
        if recent_conversation and recent_conversation != "无":
            lines.insert(1, "最近 3 轮对话已直接作为上下文参与回答。")
        for index, chunk in enumerate(chunks[:3], start=1):
            snippet = self._build_preview_snippet(chunk, question)
            lines.append(f"{index}. {chunk['source_name']}：{snippet[:160]}...")
        if llm_error:
            lines.append(f"LLM 整理阶段失败，已自动降级为本地摘要：{llm_error}")
        else:
            lines.append("当前处于无大模型模式，以上是基于本地检索结果的证据摘要。")
        return "\n".join(lines)

    def _prepare_context_layers(self, question: str, session_id: str) -> tuple[str, list[dict], str]:
        recent_entries, older_entries = self._split_session_entries(session_id, recent_turns=3)
        recent_conversation = self._format_history_entries(recent_entries)
        memory_hits = self._retrieve_memory_hits_from_entries(
            question=question,
            session_id=session_id,
            entries=older_entries,
        )
        effective_question = self._build_effective_question(
            question=question,
            recent_entries=recent_entries,
            memory_hits=memory_hits,
        )
        return recent_conversation, memory_hits, effective_question

    def _split_session_entries(
        self,
        session_id: str,
        recent_turns: int = 3,
    ) -> tuple[list[HistoryEntry], list[HistoryEntry]]:
        if not session_id:
            return [], []
        entries = self.store.list_session_entries(session_id)
        if not entries:
            return [], []
        if recent_turns <= 0:
            return [], entries
        recent_entries = entries[-recent_turns:]
        older_entries = entries[:-recent_turns]
        return recent_entries, older_entries

    def _retrieve_memory_hits_from_entries(
        self,
        question: str,
        session_id: str,
        entries: list[HistoryEntry],
    ) -> list[dict]:
        if not session_id or not entries:
            return []
        grouped_entries = self._build_memory_entry_groups(entries)
        target_group_ids = self._select_memory_target_groups(question, grouped_entries)
        if not target_group_ids:
            return []
        candidate_entries: list[HistoryEntry] = []
        for group_id in target_group_ids:
            candidate_entries.extend(grouped_entries[group_id]["entries"])
        candidate_entries.sort(key=lambda item: item.id)
        memory_rows = self._history_entries_to_memory_rows(candidate_entries)
        retriever = SessionMemoryRetriever(memory_rows, self.config)
        hits = retriever.retrieve(
            question,
            min_relevance=self.config.memory_relevance_threshold,
        )
        return self._serialize_memory_hits(hits)

    @staticmethod
    def _build_memory_entry_groups(entries: list[HistoryEntry]) -> dict[int, dict[str, object]]:
        groups: dict[int, dict[str, object]] = {}
        for entry in entries:
            if entry.memory_group_id <= 0:
                raise ValueError(f"Invalid memory_group_id on history entry: {entry.id}")
            keywords = [str(token).strip() for token in entry.memory_keywords if str(token).strip()]
            if not keywords:
                raise ValueError(f"Missing memory keywords on history entry: {entry.id}")
            group = groups.setdefault(
                entry.memory_group_id,
                {
                    "entries": [],
                    "keywords": [],
                    "keyword_set": set(),
                    "latest_id": 0,
                },
            )
            group["entries"].append(entry)
            group["latest_id"] = max(int(group["latest_id"]), entry.id)
            keyword_set = group["keyword_set"]
            for keyword in keywords:
                normalized = keyword.lower()
                if normalized in keyword_set:
                    continue
                keyword_set.add(normalized)
                group["keywords"].append(keyword)
        return groups

    @staticmethod
    def _extract_memory_query_keywords(question: str, limit: int = 18) -> list[str]:
        if not question.strip():
            raise ValueError("Question is empty; cannot extract memory query keywords.")
        tokens: list[str] = []
        tokens.extend(extract_priority_legal_terms(question))
        sorted_terms = sorted(extract_query_terms(question), key=len, reverse=True)
        tokens.extend(term for term in sorted_terms if len(term) >= 2)
        deduped: list[str] = []
        seen: set[str] = set()
        for token in tokens:
            normalized = token.strip().lower()
            if len(normalized) < 2:
                continue
            if normalized in seen:
                continue
            seen.add(normalized)
            deduped.append(token.strip())
            if len(deduped) >= limit:
                break
        if not deduped:
            raise ValueError("Failed to extract memory query keywords.")
        return deduped

    def _select_memory_target_groups(
        self,
        question: str,
        grouped_entries: dict[int, dict[str, object]],
    ) -> list[int]:
        query_keywords = self._extract_memory_query_keywords(question)
        query_set = {token.lower() for token in query_keywords}
        scored_groups: list[tuple[int, float]] = []
        for group_id, payload in grouped_entries.items():
            group_set = payload["keyword_set"]
            overlap = len(query_set & group_set)
            if overlap == 0:
                continue
            union = len(query_set | group_set)
            jaccard = overlap / max(union, 1)
            score = overlap * 0.55 + jaccard * 0.45
            scored_groups.append((group_id, score))
        if not scored_groups:
            return []
        scored_groups.sort(key=lambda item: item[1], reverse=True)
        return [group_id for group_id, _ in scored_groups[:3]]

    @staticmethod
    def _history_entries_to_memory_rows(entries: list[HistoryEntry]) -> list[dict[str, object]]:
        return [
            {
                "id": entry.id,
                "session_id": entry.session_id,
                "memory_group_id": entry.memory_group_id,
                "memory_keywords": entry.memory_keywords,
                "question": entry.question,
                "answer": entry.answer,
                "created_at": entry.created_at,
                "text": f"问题：{entry.question}\n回答：{entry.answer}",
            }
            for entry in entries
        ]

    @staticmethod
    def _serialize_memory_hits(memory_hits: list[object]) -> list[dict]:
        serialized: list[dict] = []
        for item in memory_hits:
            if isinstance(item, dict):
                serialized.append(dict(item))
                continue
            serialized.append(
                {
                    "entry_id": getattr(item, "entry_id", 0),
                    "session_id": getattr(item, "session_id", ""),
                    "score": getattr(item, "score", 0.0),
                    "relevance": getattr(item, "relevance", 0.0),
                    "question": getattr(item, "question", ""),
                    "answer": getattr(item, "answer", ""),
                    "created_at": getattr(item, "created_at", ""),
                    "text": getattr(item, "text", ""),
                    "metadata": getattr(item, "metadata", {}),
                }
            )
        return serialized

    @staticmethod
    def _build_effective_question(
        question: str,
        recent_entries: list[HistoryEntry],
        memory_hits: list[dict],
    ) -> str:
        if not recent_entries and not memory_hits:
            return question
        if not LegalRAGAgent._should_expand_with_context(question, recent_entries, memory_hits):
            return question
        parts = [question.strip()]
        if recent_entries:
            lead_recent = recent_entries[-1]
            parts.append(f"最近对话问题：{lead_recent.question}".strip())
            if lead_recent.answer:
                answer = " ".join(lead_recent.answer.split())
                parts.append(f"最近对话结论：{answer[:180]}")
        if memory_hits:
            lead_memory = memory_hits[0]
            parts.append(f"更早历史问题：{lead_memory.get('question', '')}".strip())
            if lead_memory.get("answer"):
                answer = " ".join(str(lead_memory["answer"]).split())
                parts.append(f"更早历史结论：{answer[:180]}")
        return "\n".join(part for part in parts if part)

    @staticmethod
    def _should_expand_with_context(
        question: str,
        recent_entries: list[HistoryEntry],
        memory_hits: list[dict],
    ) -> bool:
        query = question.strip()
        if len(query) <= 24:
            return bool(recent_entries or memory_hits)
        if re.search(r"(这个|那这个|那种|这种|上面|前面|继续|展开|详细|为什么|依据|刚才|上一轮)", query):
            return bool(recent_entries or memory_hits)
        if recent_entries and len(query) <= 40:
            return True
        if not memory_hits:
            return False
        top_relevance = float(
            memory_hits[0].get("relevance", memory_hits[0].get("score", 0.0)) or 0.0
        )
        return top_relevance >= 0.7

    @staticmethod
    def _format_memory_context(memory_hits: list[dict]) -> str:
        if not memory_hits:
            return "无"
        lines = []
        for index, hit in enumerate(memory_hits, start=1):
            answer = " ".join(str(hit.get("answer", "")).split())
            lines.append(
                f"[M{index}] 历史问题：{hit.get('question', '')}\n"
                f"历史回答摘要：{answer[:220]}"
            )
        return "\n\n".join(lines)

    @staticmethod
    def _format_history_entries(entries: list[HistoryEntry], limit: int | None = None) -> str:
        if not entries:
            return "无"
        selected_entries = entries[-limit:] if limit else entries
        lines = []
        for entry in selected_entries:
            lines.append(f"用户：{entry.question}")
            lines.append(f"助手：{' '.join(entry.answer.split())[:260]}")
        return "\n".join(lines) if lines else "无"

    def _format_recent_conversation(self, session_id: str, limit: int = 3) -> str:
        recent_entries, _ = self._split_session_entries(session_id, recent_turns=limit)
        return self._format_history_entries(recent_entries)

    @staticmethod
    def _build_citations(chunks: list[dict], question: str) -> list[dict]:
        citations = []
        for index, chunk in enumerate(chunks, start=1):
            metadata = chunk.get("metadata", {}) or {}
            page_start = metadata.get("page_start")
            page_end = metadata.get("page_end")
            page_label = ""
            if page_start and page_end:
                page_label = (
                    f"第 {page_start} 页"
                    if page_start == page_end
                    else f"第 {page_start}-{page_end} 页"
                )
            label = f"{chunk['source_name']} / chunk {index}"
            if page_label:
                label = f"{label} / {page_label}"
            citations.append(
                {
                    "label": label,
                    "source_name": chunk["source_name"],
                    "source_path": chunk["source_path"],
                    "title": chunk["title"],
                    "chunk_id": chunk["chunk_id"],
                    "chunk_index": chunk.get("chunk_index", index - 1),
                    "page_start": page_start,
                    "page_end": page_end,
                    "page_numbers": metadata.get("page_numbers", []),
                    "file_type": chunk.get("file_type", ""),
                    "snippet": LegalRAGAgent._build_preview_snippet(chunk, question)[:220],
                }
            )
        return citations

    @staticmethod
    def _build_preview_snippet(chunk: dict, question: str = "") -> str:
        text = chunk.get("text", "")
        metadata = chunk.get("metadata", {}) or {}
        preview_offset = LegalRAGAgent._locate_preview_start(text, metadata, question)
        if 0 < preview_offset < len(text):
            text = text[preview_offset:]
        text = LegalRAGAgent._trim_to_legal_anchor(text)
        return " ".join(text.split())

    @staticmethod
    def _locate_preview_start(text: str, metadata: dict, question: str) -> int:
        base_offset = int(metadata.get("preview_offset_in_chunk", 0) or 0)
        if not question:
            return base_offset

        query_terms = LegalRAGAgent._extract_preview_terms(question)
        matches = [
            (term, text.find(term))
            for term in query_terms
            if text.find(term) >= 0
        ]
        if not matches:
            return base_offset

        best_term, hit_pos = max(
            matches,
            key=lambda item: (len(item[0]), -item[1]),
        )
        anchor_matches = list(
            re.finditer(r"第[一二三四五六七八九十百零〇\d]+条", text)
        )
        anchor_pos = None
        for match in anchor_matches:
            if match.start() <= hit_pos:
                anchor_pos = match.start()
            else:
                break

        if anchor_pos is not None and hit_pos - anchor_pos <= 220:
            return max(base_offset, anchor_pos)
        backtrack = 20 if len(best_term) >= 4 else 40
        return max(base_offset, max(0, hit_pos - backtrack))

    @staticmethod
    def _extract_preview_terms(question: str) -> list[str]:
        preferred_terms: list[str] = []
        legal_phrases = re.findall(
            r"(正当防卫|防卫过当|特殊防卫|紧急避险|强奸|抢劫|杀人|绑架|行凶|刑事责任|民事责任|治安处罚)",
            question,
        )
        preferred_terms.extend(legal_phrases)

        for block in re.findall(r"[\u4e00-\u9fff]{4,}", question):
            preferred_terms.append(block)
            for size in range(min(8, len(block)), 3, -1):
                for idx in range(0, len(block) - size + 1):
                    preferred_terms.append(block[idx : idx + size])

        deduped: list[str] = []
        seen: set[str] = set()
        for term in sorted(preferred_terms, key=len, reverse=True):
            if term in seen:
                continue
            seen.add(term)
            deduped.append(term)
        return deduped

    @staticmethod
    def _trim_to_legal_anchor(text: str) -> str:
        anchors = [
            r"第[一二三四五六七八九十百零〇\d]+条",
            r"第[一二三四五六七八九十百零〇\d]+章",
            r"第[一二三四五六七八九十百零〇\d]+节",
        ]
        matches = []
        for pattern in anchors:
            matches.extend(re.finditer(pattern, text))
        matches = sorted(matches, key=lambda item: item.start())
        if not matches:
            return text

        first = next((match for match in matches if match.start() <= 180), None)
        if first is None:
            return text

        trailing = text[first.end(): first.end() + 8]
        if re.match(r"(和|、|及|至|或者|并|，|,)", trailing):
            next_anchor = next(
                (
                    match
                    for match in matches
                    if match.start() > first.start() and match.start() <= 180
                    and text[match.start() - 1] in {" ", "\n", "。", "；", ";", "：", ":"}
                ),
                None,
            )
            if next_anchor is not None:
                return text[next_anchor.start():]
        if first.start() > 0:
            return text[first.start():]
        return text

    def _attach_thinking_summary(self, result: dict, llm_settings: LLMSettings) -> dict:
        if not isinstance(result, dict):
            return result
        enriched = dict(result)
        enriched["thinking"] = self._build_thinking_summary(enriched, llm_settings)
        return enriched

    @staticmethod
    def _build_thinking_summary(result: dict, llm_settings: LLMSettings) -> str:
        scope = str(result.get("conversation_scope", "legal") or "legal")
        mode = str(result.get("retrieval_mode", llm_settings.retrieval_mode) or llm_settings.retrieval_mode)
        effective_question = str(result.get("effective_question", "") or "").strip()
        memory_hits = result.get("memory_hits") or []
        retrieved_chunks = result.get("retrieved_chunks") or []
        citations = result.get("citations") or []
        llm_error = str(result.get("llm_error", "") or "").strip()

        lines: list[str] = []
        if scope == "general":
            lines.append("1. 已识别为通用对话，跳过法律检索。")
            if llm_settings.enabled:
                lines.append("2. 直接走通用对话回复。")
            else:
                lines.append("2. 当前未配置可用大模型，只能返回通用对话提示。")
            return "\n".join(lines)

        if mode == "llm_retrieval":
            lines.append("1. 已识别为法律问题，采用大模型检索模式。")
        else:
            lines.append("1. 已识别为法律问题，采用混合检索模式。")

        lines.append("2. 当前使用精确模式，保留法律改写、法域路由、争点拆解、自检和引用校验。")

        if effective_question:
            lines.append(f"3. 检索语句：{effective_question}")
        if memory_hits:
            lines.append(f"4. 已召回会话记忆：{len(memory_hits)} 条。")
        if retrieved_chunks:
            lines.append(f"5. 已召回候选证据：{len(retrieved_chunks)} 条。")
        if citations:
            lines.append(f"6. 最终保留引用：{len(citations)} 条。")
        if llm_error:
            lines.append(f"7. 当前轮存在降级或失败提示：{llm_error}")
        return "\n".join(lines)

    @staticmethod
    def _format_llm_error(exc: Exception) -> str:
        text = " ".join(str(exc).split()).strip()
        if not text:
            return exc.__class__.__name__
        if len(text) > 220:
            text = text[:217] + "..."
        return text

    @staticmethod
    def _stream_text(text: str):
        segments = [part for part in re.split(r"(?<=[。！？\n])", text) if part]
        if not segments:
            segments = [text]
        for segment in segments:
            yield segment
