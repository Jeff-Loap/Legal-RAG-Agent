# -*- coding: utf-8 -*-
import json
import os
import pickle

import numpy as np
import faiss

from legal_agent.runtime_env import configure_local_ml_runtime

configure_local_ml_runtime()

from sentence_transformers import SentenceTransformer

CHUNKS_JSONL = r"D:\PythonFile\JCAI\RAG\agent_guide_chunks.jsonl"
INDEX_FILE = r"D:\PythonFile\JCAI\RAG\agent_guide.faiss"
META_FILE = r"D:\PythonFile\JCAI\RAG\agent_guide_meta.pkl"

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

def main():
    model = SentenceTransformer(EMBED_MODEL)

    texts = []
    metas = []
    with open(CHUNKS_JSONL, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            texts.append(obj["text"])
            metas.append({
                "id": obj["id"],
                **obj["metadata"]
            })

    embs = model.encode(texts, batch_size=32, show_progress_bar=True)
    embs = np.asarray(embs, dtype="float32")

    # cosine 相似度：对向量做 L2 normalize，然后用 inner product
    faiss.normalize_L2(embs)
    dim = embs.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embs)

    faiss.write_index(index, INDEX_FILE)
    with open(META_FILE, "wb") as f:
        pickle.dump({"metas": metas, "texts": texts}, f)

    print(f"FAISS 入库完成：{len(texts)} chunks")
    print("INDEX:", INDEX_FILE)
    print("META :", META_FILE)

if __name__ == "__main__":
    os.makedirs(os.path.dirname(INDEX_FILE), exist_ok=True)
    main()
