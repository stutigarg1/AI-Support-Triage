"""
Builds a semantic vector index over the entire support corpus using
sentence-transformers. No API required — runs fully offline.

The index is built once per process and cached as a module-level singleton.
"""

import os
import re
from pathlib import Path
from typing import Optional

import numpy as np
from sentence_transformers import SentenceTransformer

DATA_DIR = Path(__file__).parent.parent / "data"

COMPANY_TO_DIR = {
    "hackerrank": DATA_DIR / "hackerrank",
    "claude": DATA_DIR / "claude",
    "visa": DATA_DIR / "visa",
}

# Small but accurate model — 80 MB, no API needed
MODEL_NAME = "all-MiniLM-L6-v2"


def _parse_frontmatter(text: str) -> tuple[dict, str]:
    meta = {}
    body = text
    if text.startswith("---"):
        end = text.find("---", 3)
        if end != -1:
            fm = text[3:end].strip()
            for line in fm.splitlines():
                if ":" in line:
                    k, _, v = line.partition(":")
                    meta[k.strip()] = v.strip().strip('"')
            body = text[end + 3:].strip()
    return meta, body


def _infer_product_area(file_path: Path, company: str) -> str:
    rel = file_path.relative_to(COMPANY_TO_DIR[company])
    parts = rel.parts
    return parts[0] if len(parts) > 1 else company


def _load_docs(company: str) -> list[dict]:
    docs = []
    base = COMPANY_TO_DIR.get(company)
    if not base or not base.exists():
        return docs
    for md_file in base.rglob("*.md"):
        if md_file.name == "index.md":
            continue
        try:
            text = md_file.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        meta, body = _parse_frontmatter(text)
        title = meta.get("title", md_file.stem)
        product_area = _infer_product_area(md_file, company)
        # Embed title + first 400 chars of body for speed
        embed_text = f"{title}. {body[:400]}"
        docs.append({
            "title": title,
            "body": body,
            "embed_text": embed_text,
            "product_area": product_area,
            "source_url": meta.get("source_url", ""),
            "company": company,
            "path": str(md_file),
        })
    return docs


class SemanticRetriever:
    def __init__(self):
        print("Loading embedding model...")
        self._model = SentenceTransformer(MODEL_NAME)
        self._docs: dict[str, list[dict]] = {}
        self._embeddings: dict[str, np.ndarray] = {}
        self._load_all()

    def _load_all(self):
        all_docs = []
        all_texts = []

        for company in COMPANY_TO_DIR:
            docs = _load_docs(company)
            self._docs[company] = docs
            all_docs.extend(docs)
            all_texts.extend(d["embed_text"] for d in docs)

        print(f"Embedding {len(all_docs)} documents...")
        all_vecs = self._model.encode(
            all_texts,
            batch_size=64,
            show_progress_bar=True,
            normalize_embeddings=True,
        )

        # Split back into per-company arrays for fast filtered search
        offset = 0
        for company in COMPANY_TO_DIR:
            n = len(self._docs[company])
            self._embeddings[company] = all_vecs[offset: offset + n]
            offset += n

        print("Index ready.")

    def search(self, query: str, company: Optional[str], top_k: int = 5) -> list[dict]:
        """Return top_k semantically similar docs. company=None searches all."""
        q_vec = self._model.encode(query, normalize_embeddings=True)

        company_key = (company or "").lower().strip()
        if company_key in self._embeddings:
            return self._search_one(q_vec, company_key, top_k)
        return self._search_all(q_vec, top_k)

    def _search_one(self, q_vec: np.ndarray, company: str, top_k: int) -> list[dict]:
        scores = self._embeddings[company] @ q_vec  # cosine (normalised)
        ranked = np.argsort(scores)[::-1][:top_k]
        return [
            {**self._docs[company][i], "score": float(scores[i])}
            for i in ranked
        ]

    def _search_all(self, q_vec: np.ndarray, top_k: int) -> list[dict]:
        candidates = []
        for company in COMPANY_TO_DIR:
            scores = self._embeddings[company] @ q_vec
            for i, score in enumerate(scores):
                candidates.append((float(score), self._docs[company][i]))
        candidates.sort(key=lambda x: x[0], reverse=True)
        return [
            {**doc, "score": score}
            for score, doc in candidates[:top_k]
        ]


_retriever: Optional[SemanticRetriever] = None


def get_retriever() -> SemanticRetriever:
    global _retriever
    if _retriever is None:
        _retriever = SemanticRetriever()
    return _retriever
