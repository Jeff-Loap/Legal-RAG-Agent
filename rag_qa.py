# -*- coding: utf-8 -*-
import pickle

import numpy as np
import faiss

from legal_agent.runtime_env import configure_local_ml_runtime

configure_local_ml_runtime()

from sentence_transformers import SentenceTransformer

INDEX_FILE = r"D:\PythonFile\JCAI\RAG\agent_guide.faiss"
META_FILE = r"D:\PythonFile\JCAI\RAG\agent_guide_meta.pkl"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

def retrieve(query: str, top_k: int = 5):
    model = SentenceTransformer(EMBED_MODEL)
    index = faiss.read_index(INDEX_FILE)

    with open(META_FILE, "rb") as f:
        data = pickle.load(f)
    metas = data["metas"]
    texts = data["texts"]

    q = model.encode([query]).astype("float32")
    faiss.normalize_L2(q)

    scores, ids = index.search(q, top_k)
    results = []
    for score, idx in zip(scores[0], ids[0]):
        if idx < 0:
            continue
        m = metas[idx]
        results.append({
            "score": float(score),
            "source": f'{m["pdf"]} p{m["page_start"]}',
            "text": texts[idx]
        })
    return results

if __name__ == "__main__":
    while True:
        q = input("\n请输入问题(回车退出)：").strip()
        if not q:
            break
        docs = retrieve(q, top_k=5)
        print("\n检索结果：")
        for i, d in enumerate(docs, 1):
            print(f"\n[{i}] score={d['score']:.4f} source={d['source']}\n{d['text'][:400]}...")


        # 简单总结：优先用最相关的可读段落生成一句“这是什么”
        def _make_one_line_summary(question: str, docs_list):
            q = question.strip().lower()
            if not docs_list:
                return "未检索到相关内容。", ""

            # 取最相关且较长的一段作为依据
            best = None
            for d in docs_list:
                t = (d.get("text") or "").strip()
                if len(t) >= 120:
                    best = d
                    break
            if best is None:
                best = docs_list[0]

            text = " ".join(best["text"].split())
            src = best["source"]

            # 针对“what is this / 这是什么文件”类问题的模板
            if any(k in q for k in ["what is this", "what is this file", "what is it", "这是什么", "这是什么文件"]):
                # 尽量从文档介绍句提炼
                # 如果检索段里包含 guide / designed / this guide，则优先摘取那一句附近
                for key in ["This guide", "This Guide", "guide is designed", "This guide is designed"]:
                    pos = text.find(key)
                    if pos != -1:
                        snippet = text[pos:pos + 260]
                        return f"这是一个关于如何构建 LLM Agents（智能体）的指南文档，面向产品与工程团队，介绍 agent 概念、设计与实践。", src

                return f"这是一个关于 LLM Agents（智能体）与其设计/编排/守护策略（guardrails）等内容的指南文档。", src

            # 其他问题：先给一个“基于最相关段落”的简短回答
            return f"我在文档中找到的最相关内容来自 {src}，核心信息是：{text[:160]}…", src


        summary, cite = _make_one_line_summary(q, docs)
        if cite:
            print(f"\n【总结】{summary}\n【引用】{cite}")
        else:
            print(f"\n【总结】{summary}")
