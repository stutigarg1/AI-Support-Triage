"""
Pure ML/RAG entry point. No LLM API required.
Reads support_tickets/support_tickets.csv and writes support_tickets/output_rag.csv.

Pipeline per ticket:
  1. Semantic retrieval (sentence-transformers, local)
  2. Rule-based classification (status, request_type)
  3. Extractive response from top retrieved doc
"""

import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).parent.parent
INPUT_CSV = REPO_ROOT / "support_tickets" / "support_tickets.csv"
OUTPUT_CSV = REPO_ROOT / "support_tickets" / "output.csv"
TOP_K = 5

from embedder import get_retriever
from classifier import classify_status, classify_request_type, infer_company, is_too_vague
from extractor import extract_response, out_of_scope_response

COMPANY_MAP = {
    "hackerrank": "hackerrank",
    "claude": "claude",
    "visa": "visa",
}


def normalise_company(raw: str) -> str | None:
    if not isinstance(raw, str):
        return None
    return COMPANY_MAP.get(raw.strip().lower())


def run():
    if not INPUT_CSV.exists():
        print(f"ERROR: input file not found: {INPUT_CSV}", file=sys.stderr)
        sys.exit(1)

    retriever = get_retriever()

    df = pd.read_csv(INPUT_CSV)
    total = len(df)
    results = []

    for idx, row in df.iterrows():
        issue = str(row.get("Issue", "") or "").strip()
        subject = str(row.get("Subject", "") or "").strip()
        company_raw = str(row.get("Company", "") or "").strip()
        company = normalise_company(company_raw)

        # If company is None/unknown, try to infer from text
        if not company:
            inferred = infer_company(issue, subject)
            company = inferred  # may still be None → searches all

        query = f"{subject} {issue}".strip()
        docs = retriever.search(query, company, top_k=TOP_K)

        top_score = docs[0]["score"] if docs else 0.0

        # For vague/no-company tickets, don't guess a product area
        if is_too_vague(issue, subject) and not company:
            product_area = "unknown"
        else:
            product_area = docs[0]["product_area"] if docs else (company or "unknown")

        # Classify
        status = classify_status(issue, subject, top_score)
        request_type = classify_request_type(issue, subject, top_score)

        # Generate response
        if request_type == "invalid":
            response, justification = out_of_scope_response(issue)
            # Invalid tickets that aren't malicious → replied
            status = "replied"
        else:
            response, justification = extract_response(issue, docs, status)

        print(f"[{idx + 1}/{total}] {company_raw or 'None'} | score={top_score:.2f} | {status} | {request_type}")
        print(f"  area={product_area} | {issue[:60]}...")

        results.append({
            "Issue": issue,
            "Subject": subject,
            "Company": company_raw,
            "status": status,
            "product_area": product_area,
            "response": response,
            "justification": justification,
            "request_type": request_type,
        })

    output_df = pd.DataFrame(results)
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nDone. Output written to {OUTPUT_CSV}")


if __name__ == "__main__":
    run()
