"""
build_index.py — ITA Technician Assistant

Builds two retrieval indexes over the chunked manual corpus:
  1. "Dense" semantic index: TF-IDF -> Truncated SVD (LSA) -> FAISS flat
     inner-product index. This gives a real continuous vector space that
     captures topic-level similarity (not just exact keyword overlap),
     entirely offline — no external embedding API or model download needed.
  2. Sparse index: BM25 over tokenized chunk text (rank_bm25), which is
     strong on exact terminology matches (model numbers, error strings,
     button names) that a semantic vector space can under-weight.

Why hybrid: technician queries mix both patterns — "how do I factory reset
a switch" is semantic/paraphrase-tolerant, while "AN-310-SW-R-8-POE" or
"192.168.20.254" needs exact lexical matching. Fusing both retrieval
signals covers both query styles. rag_engine.py does the fusion at query
time; this script only builds the two indexes.

Note on the embedding choice: this deployment runs fully offline (TF-IDF+LSA)
because this environment has no network path to a model hub. In a real
deployment with normal internet access, swap `DenseIndexer` for a proper
sentence-transformer (e.g. all-MiniLM-L6-v2) for stronger semantic recall —
the rest of the pipeline (FAISS index, hybrid fusion, reranking) is unchanged.
"""

import os
import json
import pickle

import numpy as np
import faiss
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize
from rank_bm25 import BM25Okapi

BASE_DIR = os.path.dirname(__file__)
CHUNKS_PATH = os.path.join(BASE_DIR, "index", "chunks.jsonl")
FAISS_PATH = os.path.join(BASE_DIR, "index", "dense.faiss")
BM25_PATH = os.path.join(BASE_DIR, "index", "bm25.pkl")
META_PATH = os.path.join(BASE_DIR, "index", "meta.pkl")
VECTORIZER_PATH = os.path.join(BASE_DIR, "index", "tfidf_svd.pkl")

SVD_DIMS = 100  # latent semantic dimensions; corpus is small so keep this modest


def load_chunks():
    chunks = []
    with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
        for line in f:
            chunks.append(json.loads(line))
    return chunks


def simple_tokenize(text: str):
    return text.lower().replace("/", " ").replace("-", " ").split()


def main():
    chunks = load_chunks()
    texts = [c["text"] for c in chunks]
    print(f"Loaded {len(texts)} chunks")

    n_svd_dims = min(SVD_DIMS, len(texts) - 1, max(2, len(texts) // 2))
    print(f"Fitting TF-IDF + TruncatedSVD (dims={n_svd_dims}) ...")

    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        min_df=1,
        max_df=0.9,
    )
    tfidf_matrix = vectorizer.fit_transform(texts)

    svd = TruncatedSVD(n_components=n_svd_dims, random_state=42)
    lsa_vectors = svd.fit_transform(tfidf_matrix)
    lsa_vectors = normalize(lsa_vectors, norm="l2", axis=1).astype("float32")

    dim = lsa_vectors.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(lsa_vectors)
    faiss.write_index(index, FAISS_PATH)
    print(f"Wrote FAISS index ({index.ntotal} vectors, dim={dim}) -> {FAISS_PATH}")

    with open(VECTORIZER_PATH, "wb") as f:
        pickle.dump({"vectorizer": vectorizer, "svd": svd}, f)
    print(f"Wrote TF-IDF/SVD transformer -> {VECTORIZER_PATH}")

    tokenized = [simple_tokenize(t) for t in texts]
    bm25 = BM25Okapi(tokenized)
    with open(BM25_PATH, "wb") as f:
        pickle.dump(bm25, f)
    print(f"Wrote BM25 index -> {BM25_PATH}")

    with open(META_PATH, "wb") as f:
        pickle.dump(chunks, f)
    print(f"Wrote metadata -> {META_PATH}")


if __name__ == "__main__":
    main()
