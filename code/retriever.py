"""
Loads all .md docs from the data/ corpus and builds a BM25 index per company.
Exposes a search() function that returns the top-k most relevant chunks.
"""

import os
import re
from pathlib import Path
from typing import Optional

from rank_bm25 import BM25Okapi

DATA_DIR = Path(__file__).parent.parent / "data"

COMPANY_TO_DIR = {
    "hackerrank": DATA_DIR / "hackerrank",
    "claude": DATA_DIR / "claude",
    "visa": DATA_DIR / "visa",
}


def _parse_frontmatter(text: str) -> tuple[dict, str]:
    """Extract YAML frontmatter and body from a markdown string."""
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
    """Derive product_area from the folder path relative to company root."""
    rel = file_path.relative_to(COMPANY_TO_DIR[company])
    parts = rel.parts
    # Use first subfolder name; fall back to company name
    return parts[0] if len(parts) > 1 else company


def _load_docs(company: str) -> list[dict]:
    """Load all .md files for a company into a list of doc dicts."""
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
        docs.append({
            "title": title,
            "body": body,
            "product_area": product_area,
            "source_url": meta.get("source_url", ""),
            "company": company,
            "path": str(md_file),
        })
    return docs


def _tokenize(text: str) -> list[str]:
    """Simple whitespace + lowercase tokenizer."""
    return re.findall(r"\w+", text.lower())


class Retriever:
    def __init__(self):
        self._indices: dict[str, BM25Okapi] = {}
        self._docs: dict[str, list[dict]] = {}
        self._load_all()

    def _load_all(self):
        for company in COMPANY_TO_DIR:
            docs = _load_docs(company)
            if not docs:
                continue
            corpus = [_tokenize(d["title"] + " " + d["body"]) for d in docs]
            self._indices[company] = BM25Okapi(corpus)
            self._docs[company] = docs

    def search(self, query: str, company: Optional[str], top_k: int = 5) -> list[dict]:
        """
        Return top_k docs for query.
        If company is None or unrecognised, search all companies and merge.
        """
        company_key = (company or "").lower().strip()
        if company_key not in self._indices:
            # Search all, pick globally best top_k
            return self._search_all(query, top_k)
        return self._search_one(query, company_key, top_k)

    def _search_one(self, query: str, company: str, top_k: int) -> list[dict]:
        tokens = _tokenize(query)
        scores = self._indices[company].get_scores(tokens)
        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
        return [self._docs[company][i] for i, _ in ranked[:top_k]]

    def _search_all(self, query: str, top_k: int) -> list[dict]:
        tokens = _tokenize(query)
        candidates = []
        for company, index in self._indices.items():
            scores = index.get_scores(tokens)
            for i, score in enumerate(scores):
                candidates.append((score, self._docs[company][i]))
        candidates.sort(key=lambda x: x[0], reverse=True)
        return [doc for _, doc in candidates[:top_k]]


# Singleton — built once per process
_retriever: Optional[Retriever] = None


def get_retriever() -> Retriever:
    global _retriever
    if _retriever is None:
        _retriever = Retriever()
    return _retriever
