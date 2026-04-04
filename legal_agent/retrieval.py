from __future__ import annotations

import os
import pickle
import re
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import faiss
import numpy as np

from .runtime_env import configure_local_ml_runtime

configure_local_ml_runtime()

from sentence_transformers import CrossEncoder, SentenceTransformer

from .config import AppConfig, get_default_config


@dataclass
class RetrievedChunk:
    chunk_id: str
    chunk_index: int
    score: float
    source_name: str
    source_path: str
    title: str
    text: str
    metadata: dict[str, Any]
    file_type: str


DOMAIN_KEYWORD_HINTS: dict[str, tuple[str, ...]] = {
    "criminal": (
        "刑法",
        "刑事",
        "犯罪",
        "治安",
        "强奸",
        "抢劫",
        "杀人",
        "绑架",
        "侮辱",
        "诽谤",
        "造谣",
    ),
    "civil": (
        "民法典",
        "民事",
        "侵权",
        "合同",
        "人格权",
        "赔偿",
        "名誉权",
        "肖像权",
        "网络侵权",
        "连带责任",
        "通知删除",
    ),
    "labor": ("劳动法", "劳动合同法", "劳动", "工伤", "社保", "仲裁"),
    "privacy": (
        "个人信息保护法",
        "个人信息",
        "隐私",
        "数据",
        "网信",
        "网络安全",
    ),
    "safety": ("安全生产法", "危险化学品", "危化", "应急管理", "安全生产"),
    "tax": ("税", "增值税", "企业所得税", "个税", "税务"),
    "general": (),
}

DOC_TYPE_HINTS: dict[str, tuple[str, ...]] = {
    "article": ("第", "条", "章", "节"),
    "case": ("案例", "判决", "裁判", "案号", "法院"),
    "interpretation": ("解释", "意见", "批复", "指导"),
    "qa": ("问答", "FAQ", "常见问题"),
    "manual": ("指南", "指引", "手册"),
}

GROUP_STOP_TERMS = {
    "中华人民共和国",
    "法律",
    "法规",
    "规定",
    "办法",
    "条例",
    "条款",
    "本法",
    "本条",
    "上述",
    "相关",
    "责任",
    "行为",
    "情节",
    "问题",
    "是否",
    "如何",
    "什么",
    "当前",
    "规定",
    "条文",
}


LEGAL_SLOT_NEGATIVE_TERMS: dict[str, tuple[str, ...]] = {
    "personality_rights": ("英雄烈士", "英烈", "烈士", "消费者", "经营者", "商品", "服务合同"),
    "platform_liability": ("消费者", "经营者", "购物", "商品", "售后", "英雄烈士"),
    "criminal_defamation": ("英雄烈士", "荣誉权", "消费者", "经营者"),
}


def extract_legal_slots(query: str) -> dict[str, set[str]]:
    text = str(query or '')
    slots: dict[str, set[str]] = {
        'actors': set(),
        'victims': set(),
        'conducts': set(),
        'rights': set(),
        'issues': set(),
        'negative_terms': set(),
    }

    actor_patterns: dict[str, tuple[str, ...]] = {
        'direct_actor': ('行为人', '侵权人', '制作者', '传播者', '发布者', '上传者'),
        'platform': ('平台', '网络服务提供者', '网站', '应用平台', '审核义务', '必要措施'),
    }
    victim_patterns: dict[str, tuple[str, ...]] = {
        'natural_person': ('他人', '自然人', '本人', '受害人', '被害人', '特定人'),
        'consumer': ('消费者',),
        'martyr': ('英雄烈士', '英烈', '烈士'),
    }
    conduct_patterns: dict[str, tuple[str, ...]] = {
        'deepfake': ('AI换脸', '换脸', '深度伪造', 'AI 合成', 'AI生成', '合成视频'),
        'explicit_fake_content': ('虚假不雅视频', '不雅视频', '淫秽视频', '裸照', '裸聊', '色情视频'),
        'fabricate': ('制作', '合成', '伪造', '捏造'),
        'disseminate': ('传播', '发布', '散布', '扩散', '上传', '转发'),
    }
    right_patterns: dict[str, tuple[str, ...]] = {
        'reputation': ('名誉权', '名誉'),
        'portrait': ('肖像权', '肖像'),
        'privacy': ('隐私权', '隐私', '个人信息'),
        'personality_rights': ('人格权', '人格利益'),
    }
    issue_patterns: dict[str, tuple[str, ...]] = {
        'insult': ('侮辱罪', '侮辱'),
        'defamation': ('诽谤罪', '诽谤'),
        'civil_liability': ('民事责任', '民事侵权', '赔偿', '停止侵害', '赔礼道歉'),
        'criminal_liability': ('刑事责任', '刑责', '定罪', '入罪', '追究刑事责任'),
        'platform_duty': ('审核义务', '必要措施', '删除', '屏蔽', '断开链接', '通知', '明知', '应知'),
    }

    for label, patterns in actor_patterns.items():
        if any(pattern in text for pattern in patterns):
            slots['actors'].add(label)
    for label, patterns in victim_patterns.items():
        if any(pattern in text for pattern in patterns):
            slots['victims'].add(label)
    for label, patterns in conduct_patterns.items():
        if any(pattern in text for pattern in patterns):
            slots['conducts'].add(label)
    for label, patterns in right_patterns.items():
        if any(pattern in text for pattern in patterns):
            slots['rights'].add(label)
    for label, patterns in issue_patterns.items():
        if any(pattern in text for pattern in patterns):
            slots['issues'].add(label)

    if not slots['actors']:
        slots['actors'].add('direct_actor')
    if not slots['victims']:
        slots['victims'].add('natural_person')

    negative_terms: set[str] = set()
    rights = slots['rights']
    issues = slots['issues']
    actors = slots['actors']

    if rights & {'reputation', 'portrait', 'privacy', 'personality_rights'}:
        negative_terms.update(LEGAL_SLOT_NEGATIVE_TERMS['personality_rights'])
    if 'platform' in actors or 'platform_duty' in issues:
        negative_terms.update(LEGAL_SLOT_NEGATIVE_TERMS['platform_liability'])
    if issues & {'insult', 'defamation', 'criminal_liability'}:
        negative_terms.update(LEGAL_SLOT_NEGATIVE_TERMS['criminal_defamation'])

    if 'martyr' not in slots['victims']:
        negative_terms.update({'英雄烈士', '英烈', '烈士'})
    if 'consumer' not in slots['victims']:
        negative_terms.update({'消费者', '经营者'})

    slots['negative_terms'] = {term for term in negative_terms if term and term not in text}
    return slots


def embed_texts(texts: list[str], config: AppConfig | None = None) -> np.ndarray:
    model = load_embedding_model(config or get_default_config())
    return model.encode(texts, batch_size=16, show_progress_bar=False)


@lru_cache(maxsize=1)
def load_embedding_model(config: AppConfig) -> SentenceTransformer:
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    if not config.embedding_model_dir.exists():
        raise FileNotFoundError(
            f"未找到本地 embedding 模型缓存：{config.embedding_model_dir}"
        )
    return SentenceTransformer(
        str(config.embedding_model_dir),
        local_files_only=True,
    )


@lru_cache(maxsize=1)
def load_reranker_model(config: AppConfig) -> CrossEncoder | None:
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    if not config.reranker_model_dir or not config.reranker_model_dir.exists():
        return None
    try:
        return CrossEncoder(
            str(config.reranker_model_dir),
            local_files_only=True,
        )
    except Exception:
        return None


class LocalHybridRetriever:
    def __init__(self, chunk_rows: list[dict], config: AppConfig | None = None):
        self.config = config or get_default_config()
        self.chunk_rows = chunk_rows
        self.chunk_map = {row["chunk_id"]: row for row in chunk_rows}
        self.chunk_ids = [row["chunk_id"] for row in chunk_rows]
        self.reranker = load_reranker_model(self.config)
        self.index = faiss.read_index(str(self.config.faiss_path))
        with self.config.tfidf_path.open("rb") as handle:
            payload = pickle.load(handle)
        self.vectorizer = payload["vectorizer"]
        self.tfidf_matrix = payload["matrix"]
        self.tfidf_chunk_ids = payload["chunk_ids"]
        self.chunk_group_map: dict[str, str] = {}
        self.group_to_chunk_ids: dict[str, list[str]] = {}
        self.group_keywords: dict[str, set[str]] = {}
        self._build_chunk_groups()

    def retrieve(self, query: str, top_k: int | None = None) -> list[RetrievedChunk]:
        top_k = top_k or self.config.final_top_k
        target_groups = self._select_target_groups(query, max_groups=12)
        boosted_chunk_ids = self._collect_group_chunk_ids(target_groups)

        dense_search_k = min(
            len(self.chunk_ids),
            max(self.config.dense_candidate_k * 3, top_k * 12, 120),
        )
        sparse_search_k = min(
            len(self.chunk_ids),
            max(self.config.sparse_candidate_k * 3, top_k * 12, 120),
        )
        term_search_k = min(
            len(self.chunk_ids),
            max(self.config.sparse_candidate_k * 3, top_k * 10, 80),
        )

        dense_rank = self._dense_candidates(
            query,
            dense_search_k,
        )
        sparse_rank = self._sparse_candidates(
            query,
            sparse_search_k,
        )
        preferred_terms = extract_priority_legal_terms(query)
        dense_weight, sparse_weight, term_weight = self._get_retrieval_weights(query)
        term_rank = self._term_candidates(
            preferred_terms=preferred_terms,
            top_n=term_search_k,
        )

        all_ids = set(dense_rank) | set(sparse_rank) | set(term_rank)
        if not all_ids and boosted_chunk_ids:
            all_ids = set(list(boosted_chunk_ids)[: max(top_k * 10, 60)])
        if not all_ids:
            return []

        scored: list[RetrievedChunk] = []
        query_terms = extract_query_terms(query)
        article_refs = extract_article_refs(query)
        law_names = extract_law_names(query)
        legal_slots = extract_legal_slots(query)

        for chunk_id in all_ids:
            row = self.chunk_map[chunk_id]
            metadata = row.get("metadata", {}) or {}
            coverage = sum(1 for term in query_terms if term in row["text"])
            preferred_hits = sum(1 for term in preferred_terms if term in row["text"])
            score = dense_rank.get(chunk_id, 0.0) * dense_weight
            score += sparse_rank.get(chunk_id, 0.0) * sparse_weight
            score += term_rank.get(chunk_id, 0.0) * term_weight
            score += min(coverage, 12) * 0.018
            score += min(preferred_hits, 6) * 0.075
            if preferred_terms and preferred_hits == 0:
                score -= 0.2
            score += self._group_match_bonus(
                chunk_id=chunk_id,
                metadata=metadata,
                target_groups=target_groups,
                boosted_chunk_ids=boosted_chunk_ids,
            )
            score += self._legal_rule_bonus(
                query=query,
                row=row,
                metadata=metadata,
                article_refs=article_refs,
                law_names=law_names,
                preferred_terms=preferred_terms,
            )
            score += self._compute_legal_slot_relevance(
                query=query,
                row=row,
                metadata=metadata,
                legal_slots=legal_slots,
            )

            scored.append(
                RetrievedChunk(
                    chunk_id=chunk_id,
                    chunk_index=row["chunk_index"],
                    score=score,
                    source_name=row["source_name"],
                    source_path=row["source_path"],
                    title=row["title"],
                    text=row["text"],
                    metadata=metadata,
                    file_type=row.get("file_type", ""),
                )
            )

        scored.sort(key=lambda item: item.score, reverse=True)
        scored = self._rerank(query, scored)
        scored.sort(key=lambda item: item.score, reverse=True)
        return scored[:top_k]

    def retrieve_lexical(self, query: str, top_k: int = 12) -> list[RetrievedChunk]:
        target_groups = self._select_target_groups(query, max_groups=12)
        boosted_chunk_ids = self._collect_group_chunk_ids(target_groups)

        sparse_rank = self._sparse_candidates(
            query,
            min(max(top_k * 10, 60), len(self.chunk_ids)),
        )
        preferred_terms = extract_priority_legal_terms(query)
        term_rank = self._term_candidates(
            preferred_terms=preferred_terms,
            top_n=min(max(top_k * 10, 60), len(self.chunk_ids)),
        )
        if not sparse_rank and not term_rank and not boosted_chunk_ids:
            return []

        scored: list[RetrievedChunk] = []
        query_terms = extract_query_terms(query)
        article_refs = extract_article_refs(query)
        law_names = extract_law_names(query)
        legal_slots = extract_legal_slots(query)

        for chunk_id in set(sparse_rank) | set(term_rank) | set(list(boosted_chunk_ids)[: max(top_k * 6, 36)]):
            sparse_score = sparse_rank.get(chunk_id, 0.0)
            row = self.chunk_map[chunk_id]
            metadata = row.get("metadata", {}) or {}
            coverage = sum(1 for term in query_terms if term in row["text"])
            preferred_hits = sum(1 for term in preferred_terms if term in row["text"])
            score = sparse_score * 0.62
            score += term_rank.get(chunk_id, 0.0) * 0.28
            score += min(coverage, 12) * 0.03
            score += min(preferred_hits, 6) * 0.09
            if preferred_terms and preferred_hits == 0:
                score -= 0.18
            score += self._group_match_bonus(
                chunk_id=chunk_id,
                metadata=metadata,
                target_groups=target_groups,
                boosted_chunk_ids=boosted_chunk_ids,
            )
            score += self._legal_rule_bonus(
                query=query,
                row=row,
                metadata=metadata,
                article_refs=article_refs,
                law_names=law_names,
                preferred_terms=preferred_terms,
            )
            score += self._compute_legal_slot_relevance(
                query=query,
                row=row,
                metadata=metadata,
                legal_slots=legal_slots,
            )
            scored.append(
                RetrievedChunk(
                    chunk_id=chunk_id,
                    chunk_index=row["chunk_index"],
                    score=score,
                    source_name=row["source_name"],
                    source_path=row["source_path"],
                    title=row["title"],
                    text=row["text"],
                    metadata=metadata,
                    file_type=row.get("file_type", ""),
                )
            )

        scored.sort(key=lambda item: item.score, reverse=True)
        return scored[:top_k]

    def _dense_candidates(
        self,
        query: str,
        top_n: int,
        allowed_chunk_ids: set[str] | None = None,
    ) -> dict[str, float]:
        if allowed_chunk_ids is not None and not allowed_chunk_ids:
            return {}
        embedding = np.asarray(embed_texts([query], self.config), dtype="float32")
        faiss.normalize_L2(embedding)
        if allowed_chunk_ids is None:
            search_n = min(max(top_n, 1), len(self.chunk_ids))
        else:
            search_n = min(len(self.chunk_ids), max(top_n * 8, 96))
        scores, indices = self.index.search(embedding, search_n)
        raw_ranked: list[tuple[str, float]] = []
        for raw_score, idx in zip(scores[0], indices[0], strict=False):
            if idx < 0:
                continue
            chunk_id = self.chunk_ids[idx]
            if allowed_chunk_ids is not None and chunk_id not in allowed_chunk_ids:
                continue
            score = max(0.0, float(raw_score))
            if score <= 0:
                continue
            raw_ranked.append((chunk_id, score))
            if len(raw_ranked) >= top_n:
                break
        if not raw_ranked:
            return {}
        max_score = max(score for _, score in raw_ranked)
        ranked: dict[str, float] = {}
        for chunk_id, score in raw_ranked:
            ranked[chunk_id] = score / max_score if max_score > 0 else 0.0
        return ranked

    def _sparse_candidates(
        self,
        query: str,
        top_n: int,
        allowed_chunk_ids: set[str] | None = None,
    ) -> dict[str, float]:
        if allowed_chunk_ids is not None and not allowed_chunk_ids:
            return {}
        query_vector = self.vectorizer.transform([query])
        scores = (self.tfidf_matrix @ query_vector.T).toarray().ravel()
        if not len(scores):
            return {}
        search_n = top_n if allowed_chunk_ids is None else min(len(scores), max(top_n * 8, 96))
        indices = np.argsort(scores)[::-1][:search_n]
        raw_ranked: list[tuple[str, float]] = []
        for idx in indices:
            if scores[idx] <= 0:
                continue
            chunk_id = self.tfidf_chunk_ids[idx]
            if allowed_chunk_ids is not None and chunk_id not in allowed_chunk_ids:
                continue
            raw_ranked.append((chunk_id, float(scores[idx])))
            if len(raw_ranked) >= top_n:
                break
        if not raw_ranked:
            return {}
        max_score = max(score for _, score in raw_ranked)
        ranked: dict[str, float] = {}
        for chunk_id, score in raw_ranked:
            ranked[chunk_id] = score / max_score if max_score > 0 else 0.0
        return ranked

    def _rerank(self, query: str, candidates: list[RetrievedChunk]) -> list[RetrievedChunk]:
        if not candidates:
            return candidates

        if self.reranker is not None:
            head = candidates[: self.config.rerank_candidate_k]
            pairs = [(query, item.text[:1000]) for item in head]
            try:
                rerank_scores = self.reranker.predict(pairs)
                rerank_max = max((float(score) for score in rerank_scores), default=0.0)
                rerank_min = min((float(score) for score in rerank_scores), default=0.0)
                spread = max(rerank_max - rerank_min, 1e-6)
                for item, rerank_score in zip(head, rerank_scores, strict=False):
                    normalized = (float(rerank_score) - rerank_min) / spread
                    item.score += normalized * 0.35
            except Exception:
                pass
        return candidates

    def _term_candidates(
        self,
        preferred_terms: list[str],
        top_n: int,
        allowed_chunk_ids: set[str] | None = None,
    ) -> dict[str, float]:
        if not preferred_terms:
            return {}
        if allowed_chunk_ids is not None and not allowed_chunk_ids:
            return {}
        normalized_terms: list[str] = []
        seen_terms: set[str] = set()
        for term in preferred_terms:
            normalized = str(term).strip()
            if not normalized:
                continue
            if len(normalized) < 2 or len(normalized) > 10:
                continue
            if normalized in GROUP_STOP_TERMS:
                continue
            if normalized in seen_terms:
                continue
            seen_terms.add(normalized)
            normalized_terms.append(normalized)
        if not normalized_terms:
            return {}
        ranked: list[tuple[str, float]] = []
        for chunk_id, row in self.chunk_map.items():
            if allowed_chunk_ids is not None and chunk_id not in allowed_chunk_ids:
                continue
            text = str(row.get("text", ""))
            if not text:
                continue
            hits = sum(1 for term in normalized_terms if term in text)
            if hits <= 0:
                continue
            early_hits = sum(1 for term in normalized_terms[:8] if term in text[:220])
            score = hits * 1.0 + early_hits * 0.25
            ranked.append((chunk_id, score))
        if not ranked:
            return {}
        ranked.sort(key=lambda item: item[1], reverse=True)
        ranked = ranked[:top_n]
        max_score = max(score for _, score in ranked)
        return {
            chunk_id: (score / max_score if max_score > 0 else 0.0)
            for chunk_id, score in ranked
        }

    def _get_retrieval_weights(self, query: str) -> tuple[float, float, float]:
        model_name = str(self.config.embedding_model_name or "").lower()
        is_chinese_query = bool(re.search(r"[\u4e00-\u9fff]", query))
        if is_chinese_query and "all-minilm" in model_name:
            return (0.18, 0.52, 0.30)
        return (0.48, 0.28, 0.24)

    def _group_match_bonus(
        self,
        chunk_id: str,
        metadata: dict[str, Any],
        target_groups: list[str],
        boosted_chunk_ids: set[str],
    ) -> float:
        if not target_groups:
            return 0.0
        score = 0.0
        group_key = self.chunk_group_map.get(chunk_id, "")
        if group_key and group_key in target_groups:
            try:
                rank = target_groups.index(group_key)
            except ValueError:
                rank = None
            score += 0.12
            if rank is not None:
                score += max(0.0, 0.08 - rank * 0.006)
        if chunk_id in boosted_chunk_ids:
            score += 0.04
        group_domain = str(metadata.get("kb_group_domain", "") or "")
        if group_domain and any(group.startswith(f"{group_domain}|") for group in target_groups):
            score += 0.03
        return score

    @staticmethod
    def _legal_rule_bonus(
        query: str,
        row: dict[str, Any],
        metadata: dict[str, Any],
        article_refs: set[str],
        law_names: list[str],
        preferred_terms: list[str],
    ) -> float:
        score = 0.0
        text = row["text"]
        source_name = row.get("source_name", "")
        title = row.get("title", "")
        article_anchor = str(metadata.get("article_anchor", "") or "")

        if metadata.get("law_chunk_type") == "article":
            score += 0.05
        if article_refs and article_anchor in article_refs:
            score += 0.22
        if article_anchor and text.startswith(article_anchor):
            score += 0.03
        if law_names and any(name in source_name or name in title for name in law_names):
            score += 0.12
        if preferred_terms:
            early_window = text[:180]
            early_hits = sum(1 for term in preferred_terms[:8] if term in early_window)
            score += early_hits * 0.03
            if any(term in {"侮辱", "诽谤", "侮辱罪", "诽谤罪"} for term in preferred_terms):
                if "刑法" in source_name or "刑法" in title:
                    score += 0.08
            if any(
                term in {"名誉权", "肖像权", "人格权", "网络侵权", "通知删除", "连带责任"}
                for term in preferred_terms
            ):
                if "民法典" in source_name or "民法典" in title:
                    score += 0.08
        if "第" in query and "条" in query and article_anchor:
            score += 0.02
        return score

    def _compute_legal_slot_relevance(
        self,
        query: str,
        row: dict[str, Any],
        metadata: dict[str, Any],
        legal_slots: dict[str, set[str]],
    ) -> float:
        if not legal_slots:
            return 0.0
        text = f"{row.get('source_name', '')} {row.get('title', '')} {row.get('text', '')[:1600]}"
        score = 0.0

        slot_patterns: dict[str, dict[str, tuple[str, ...]]] = {
            "actors": {
                "direct_actor": ("行为人", "侵权人", "制作者", "传播者", "发布者"),
                "platform": ("平台", "网络服务提供者", "信息网络", "网站", "应用平台"),
            },
            "victims": {
                "natural_person": ("他人", "自然人", "受害人", "被害人", "特定人"),
                "consumer": ("消费者",),
                "martyr": ("英雄烈士", "英烈", "烈士"),
            },
            "conducts": {
                "deepfake": ("AI换脸", "换脸", "深度伪造"),
                "explicit_fake_content": ("虚假不雅视频", "不雅视频", "淫秽视频", "裸照", "裸聊"),
                "fabricate": ("制作", "合成", "伪造"),
                "disseminate": ("传播", "发布", "散布", "扩散", "上传"),
            },
            "rights": {
                "reputation": ("名誉权", "名誉"),
                "portrait": ("肖像权", "肖像"),
                "privacy": ("隐私权", "隐私", "个人信息"),
                "personality_rights": ("人格权", "人格利益"),
            },
            "issues": {
                "insult": ("侮辱罪", "侮辱"),
                "defamation": ("诽谤罪", "诽谤"),
                "civil_liability": ("民事责任", "民事侵权", "赔偿", "停止侵害"),
                "criminal_liability": ("刑事责任", "刑责", "定罪", "追究刑事责任"),
                "platform_duty": ("审核义务", "必要措施", "删除", "屏蔽", "断开链接", "知道或者应当知道", "明知", "应知"),
            },
        }

        for slot_name, labels in slot_patterns.items():
            requested_labels = legal_slots.get(slot_name, set())
            if not requested_labels:
                continue
            for label in requested_labels:
                patterns = labels.get(label, ())
                if any(pattern in text for pattern in patterns):
                    score += 0.06
                elif slot_name in {"actors", "issues", "rights"}:
                    score -= 0.015

        negative_hits = 0
        for term in legal_slots.get("negative_terms", set()):
            if term and term in text:
                negative_hits += 1
        score -= min(negative_hits, 4) * 0.16

        group_law = str(metadata.get("kb_group_law", "") or "")
        group_doc_type = str(metadata.get("kb_group_doc_type", "") or "")
        if "platform" in legal_slots.get("actors", set()):
            if any(token in text for token in ("网络服务提供者", "信息网络", "必要措施", "通知", "删除")):
                score += 0.08
            if any(token in text for token in ("消费者", "经营者")):
                score -= 0.12
        if legal_slots.get("issues", set()) & {"insult", "defamation", "criminal_liability"}:
            if "刑法" in group_law or "刑法" in row.get("source_name", "") or "刑法" in row.get("title", ""):
                score += 0.10
            if any(token in text for token in ("英雄烈士", "英烈")):
                score -= 0.25
        if legal_slots.get("rights", set()) & {"reputation", "portrait", "privacy", "personality_rights"}:
            if "民法典" in group_law or "人格权" in text:
                score += 0.08
        if legal_slots.get("conducts", set()) & {"deepfake", "explicit_fake_content"}:
            if any(token in text for token in ("AI", "换脸", "深度伪造", "网络", "视频", "传播")):
                score += 0.06
        if group_doc_type == "case" and any(token in query for token in ("是否构成", "承担", "责任", "构成")):
            score += 0.03
        return score

    def _build_chunk_groups(self) -> None:
        for row in self.chunk_rows:
            metadata = row.get("metadata", {}) or {}
            payload = classify_chunk_group(
                source_name=str(row.get("source_name", "")),
                title=str(row.get("title", "")),
                text=str(row.get("text", "")),
                metadata=metadata,
            )
            metadata.update(payload)
            row["metadata"] = metadata
            group_key = payload["kb_group_key"]
            group_keywords = payload["kb_group_keywords"]
            chunk_id = str(row["chunk_id"])
            self.chunk_group_map[chunk_id] = group_key
            self.group_to_chunk_ids.setdefault(group_key, []).append(chunk_id)
            keyword_set = self.group_keywords.setdefault(group_key, set())
            keyword_set.update(
                str(token).strip().lower()
                for token in group_keywords
                if str(token).strip()
            )

    def _collect_group_chunk_ids(self, target_groups: list[str]) -> set[str]:
        return {
            chunk_id
            for group_key in target_groups
            for chunk_id in self.group_to_chunk_ids.get(group_key, [])
        }

    def _select_target_groups(self, query: str, max_groups: int = 8) -> list[str]:
        query_terms = build_group_query_terms(query)
        query_set = {term.strip().lower() for term in query_terms if term.strip()}
        query_laws = {name.strip().lower() for name in extract_law_names(query)}
        query_domain = detect_domain_from_text(query)
        candidate_group_keys = list(self.group_keywords.keys())

        scored: list[tuple[str, float]] = []
        for group_key in candidate_group_keys:
            keywords = self.group_keywords[group_key]
            overlap = len(query_set & keywords) if query_set else 0
            union = len(query_set | keywords) if query_set else len(keywords)
            score = 0.0
            if overlap > 0:
                score += overlap * 0.55 + (overlap / max(union, 1)) * 0.45
            group_lower = group_key.lower()
            if query_laws and any(law in group_lower for law in query_laws):
                score += 0.35
            if query_domain and query_domain != "general" and group_key.startswith(f"{query_domain}|"):
                score += 0.25
            if not query_set and query_domain and query_domain != "general":
                if group_key.startswith(f"{query_domain}|"):
                    score += 0.1
            if score <= 0:
                continue
            scored.append((group_key, score))

        scored.sort(key=lambda item: item[1], reverse=True)
        selected = [group for group, _ in scored[:max_groups]]
        if selected:
            return selected

        if query_domain and query_domain != "general":
            domain_groups = [
                group_key
                for group_key in self.group_keywords.keys()
                if group_key.startswith(f"{query_domain}|")
            ]
            domain_groups.sort(
                key=lambda key: len(self.group_to_chunk_ids.get(key, [])),
                reverse=True,
            )
            return domain_groups[:max_groups]

        fallback_groups = sorted(
            self.group_keywords.keys(),
            key=lambda key: len(self.group_to_chunk_ids.get(key, [])),
            reverse=True,
        )
        return fallback_groups[:max_groups]


def extract_query_terms(query: str) -> set[str]:
    chinese_terms: set[str] = set()
    for block in re.findall(r"[\u4e00-\u9fff]{2,}", query):
        chinese_terms.add(block)
        for size in range(2, min(4, len(block)) + 1):
            for idx in range(0, len(block) - size + 1):
                chinese_terms.add(block[idx : idx + size])
    ascii_terms = {term.lower() for term in re.findall(r"[A-Za-z0-9_]{2,}", query)}
    return chinese_terms | ascii_terms


def extract_priority_legal_terms(query: str) -> list[str]:
    terms = re.findall(
        r"(正当防卫|防卫过当|特殊防卫|紧急避险|强奸|抢劫|杀人|绑架|行凶|侮辱罪|诽谤罪|侮辱|诽谤|名誉权|肖像权|人格权|深度伪造|换脸|AI换脸|审核义务|通知删除|知道或者应当知道|网络侵权|网络服务提供者|连带责任|拒不履行信息网络安全管理义务罪|非法利用信息网络罪|帮助信息网络犯罪活动罪|民事责任|刑事责任|行政处罚|治安处罚|个人信息|劳动合同|劳动合同法|劳动法|试用期|离职|解除劳动合同|提前通知|通知用人单位|道路交通安全法|道路交通安全|交通安全|交通管理|公安机关交通管理部门|国务院公安部门|危险化学品)",
        query,
    )
    for block in re.findall(r"[\u4e00-\u9fff]{4,}", query):
        terms.append(block)
    deduped: list[str] = []
    seen: set[str] = set()
    enriched_terms = list(terms)
    for term in terms:
        if term.endswith("罪") and len(term) >= 3:
            enriched_terms.append(term[:-1])
        if term == "AI换脸":
            enriched_terms.extend(["换脸", "深度伪造"])
        if term == "侮辱罪":
            enriched_terms.append("侮辱")
        if term == "诽谤罪":
            enriched_terms.append("诽谤")
    enriched_terms.extend(_extract_legal_subterms(query, max_terms=36))
    for term in sorted(enriched_terms, key=len, reverse=True):
        if term in seen:
            continue
        seen.add(term)
        deduped.append(term)
    return deduped


def extract_article_refs(query: str) -> set[str]:
    return set(
        re.findall(
            r"第[一二三四五六七八九十百千万零〇\d]+条(?:之[一二三四五六七八九十百千万零〇\d]+)?",
            query,
        )
    )


def extract_law_names(query: str) -> list[str]:
    names = re.findall(
        r"((?:中华人民共和国)?[\u4e00-\u9fff]{2,24}(?:法典|法律|法|条例|办法|规定|解释))",
        query,
    )
    deduped: list[str] = []
    seen: set[str] = set()
    for name in names:
        if name in seen:
            continue
        seen.add(name)
        deduped.append(name)
    return deduped


def detect_domain_from_text(text: str) -> str:
    normalized = str(text or "")
    best_domain = "general"
    best_hits = 0
    for domain, hints in DOMAIN_KEYWORD_HINTS.items():
        if domain == "general":
            continue
        hits = sum(1 for hint in hints if hint in normalized)
        if hits > best_hits:
            best_hits = hits
            best_domain = domain
    return best_domain


def classify_chunk_group(
    source_name: str,
    title: str,
    text: str,
    metadata: dict[str, Any] | None,
) -> dict[str, Any]:
    metadata = metadata or {}
    combined = f"{source_name} {title} {text[:640]}"
    law_names = extract_law_names(combined)
    law_name = law_names[0] if law_names else _guess_law_name_from_text(combined)
    domain = _domain_from_law_name(law_name) or "general"
    doc_type = _detect_doc_type(combined, metadata)
    keywords = _build_group_keywords(
        combined=combined,
        law_name=law_name,
        domain=domain,
        doc_type=doc_type,
        limit=18,
    )
    if not keywords:
        raise ValueError(
            f"Failed to derive knowledge-group keywords for chunk source={source_name} title={title}"
        )
    group_key = f"{domain}|{law_name}|{doc_type}"
    return {
        "kb_group_key": group_key,
        "kb_group_domain": domain,
        "kb_group_law": law_name,
        "kb_group_doc_type": doc_type,
        "kb_group_keywords": keywords,
    }


def build_group_query_terms(query: str) -> list[str]:
    terms: list[str] = []
    terms.extend(extract_law_names(query))
    terms.extend(extract_priority_legal_terms(query))
    terms.extend(_expand_query_alias_terms(query))
    terms.extend(_extract_phrase_terms(query, min_len=3, max_len=20, max_terms=18))
    terms.extend(_extract_legal_subterms(query, max_terms=28))
    terms.extend(re.findall(r"[A-Za-z][A-Za-z0-9_]{2,20}", query))
    domain = detect_domain_from_text(query)
    terms.extend(_domain_alias_terms(domain))
    deduped: list[str] = []
    seen: set[str] = set()
    for term in terms:
        normalized = term.strip().lower()
        if len(normalized) < 2:
            continue
        if normalized in GROUP_STOP_TERMS:
            continue
        if normalized in seen:
            continue
        seen.add(normalized)
        deduped.append(term.strip())
    return deduped[:24]


def _guess_law_name_from_text(text: str) -> str:
    aliases = [
        "中华人民共和国刑法",
        "中华人民共和国民法典",
        "中华人民共和国劳动合同法",
        "中华人民共和国劳动法",
        "中华人民共和国个人信息保护法",
        "中华人民共和国安全生产法",
        "中华人民共和国治安管理处罚法",
    ]
    for alias in aliases:
        if alias in text:
            return alias
    for alias in aliases:
        short = alias.replace("中华人民共和国", "")
        if short in text:
            return alias
    return "通用法律资料"


def _domain_from_law_name(law_name: str) -> str:
    lower = law_name.lower()
    if any(key in lower for key in ("刑法", "治安管理处罚法")):
        return "criminal"
    if any(key in lower for key in ("民法典", "民法")):
        return "civil"
    if any(key in lower for key in ("劳动合同法", "劳动法")):
        return "labor"
    if any(key in lower for key in ("个人信息保护法", "网络安全法")):
        return "privacy"
    if any(key in lower for key in ("安全生产法", "危险化学品")):
        return "safety"
    if "税" in lower:
        return "tax"
    return ""


def _detect_doc_type(text: str, metadata: dict[str, Any]) -> str:
    if metadata.get("law_chunk_type") == "article":
        return "article"
    for doc_type, hints in DOC_TYPE_HINTS.items():
        if hints and any(hint in text for hint in hints):
            return doc_type
    return "general"


def _domain_alias_terms(domain: str) -> list[str]:
    if not domain or domain == "general":
        return []
    return [domain, *DOMAIN_KEYWORD_HINTS.get(domain, ())]


def _build_group_keywords(
    combined: str,
    law_name: str,
    domain: str,
    doc_type: str,
    limit: int = 18,
) -> list[str]:
    tokens: list[str] = []
    if law_name:
        tokens.append(law_name)
        tokens.append(law_name.replace("中华人民共和国", ""))
    tokens.extend(_domain_alias_terms(domain))
    tokens.append(doc_type)
    tokens.extend(extract_priority_legal_terms(combined))
    tokens.extend(_extract_phrase_terms(combined, min_len=3, max_len=24, max_terms=20))
    deduped: list[str] = []
    seen: set[str] = set()
    for token in tokens:
        normalized = token.strip().lower()
        if len(normalized) < 2:
            continue
        if normalized in GROUP_STOP_TERMS:
            continue
        if normalized in seen:
            continue
        seen.add(normalized)
        deduped.append(token.strip())
        if len(deduped) >= limit:
            break
    return deduped


def _extract_legal_subterms(
    text: str,
    max_terms: int = 28,
) -> list[str]:
    terms: list[str] = []
    seen: set[str] = set()
    blocks = re.findall(r"[\u4e00-\u9fff]{3,36}", text)
    preferred_lengths = (4, 3, 2)
    for block in blocks:
        for size in preferred_lengths:
            if len(block) < size:
                continue
            for idx in range(0, len(block) - size + 1):
                token = block[idx : idx + size]
                normalized = token.lower()
                if normalized in seen:
                    continue
                if normalized in GROUP_STOP_TERMS:
                    continue
                if token in {"可能", "大量", "平台", "传播", "承担", "义务", "责任"}:
                    continue
                if not re.search(r"(罪|法|权|责|刑|民|侵|审|赔|罚|防卫|诽谤|侮辱|名誉|肖像|隐私|网络|劳动|合同|试用|离职|通知|用人|工资|工伤|社保|仲裁|道路|交通|公安|机关|驾驶|车辆|安全)", token):
                    continue
                seen.add(normalized)
                terms.append(token)
                if len(terms) >= max_terms:
                    return terms
    return terms


def _extract_phrase_terms(
    text: str,
    min_len: int = 3,
    max_len: int = 24,
    max_terms: int = 20,
) -> list[str]:
    terms: list[str] = []
    seen: set[str] = set()
    pattern = rf"[\u4e00-\u9fff]{{{min_len},{max_len}}}"
    for block in re.findall(pattern, text):
        token = block.strip()
        if token in GROUP_STOP_TERMS:
            continue
        normalized = token.lower()
        if normalized in seen:
            continue
        seen.add(normalized)
        terms.append(token)
        if len(terms) >= max_terms:
            break
    return terms


def _expand_query_alias_terms(query: str) -> list[str]:
    text = str(query or "")
    terms: list[str] = []
    if re.search(r"(侮辱|诽谤|造谣)", text):
        terms.extend(
            [
                "中华人民共和国刑法",
                "刑法",
                "侮辱罪",
                "诽谤罪",
            ]
        )
    if re.search(r"(平台|网络服务|审核|删除|屏蔽|断开链接|应当知道|通知)", text):
        terms.extend(
            [
                "中华人民共和国民法典",
                "网络侵权责任",
                "网络服务提供者",
                "通知删除",
                "知道或者应当知道",
            ]
        )
    if re.search(r"(名誉|肖像|人格权|隐私)", text):
        terms.extend(
            [
                "中华人民共和国民法典",
                "名誉权",
                "肖像权",
                "人格权",
            ]
        )
    if re.search(r"(AI换脸|深度伪造|换脸)", text, re.I):
        terms.extend(
            [
                "深度伪造",
                "AI换脸",
                "个人信息保护法",
                "网络安全法",
            ]
        )
    if re.search(r"(试用期|离职|解除劳动合同|通知用人单位|提前几天通知)", text):
        terms.extend(
            [
                "中华人民共和国劳动合同法",
                "劳动合同法",
                "试用期",
                "离职",
                "解除劳动合同",
                "提前通知",
                "通知用人单位",
                "第三十七条",
            ]
        )
    if re.search(r"(道路交通|交通安全|交通管理|公安机关交通管理部门|国务院公安部门|驾驶证|机动车)", text):
        terms.extend(
            [
                "中华人民共和国道路交通安全法",
                "道路交通安全法",
                "道路交通安全管理",
                "交通管理",
                "公安机关交通管理部门",
                "国务院公安部门",
                "第五条",
            ]
        )
    return terms
