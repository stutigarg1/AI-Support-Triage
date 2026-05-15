"""
[ALTERNATIVE] Gemini Flash LLM-based pipeline entry point.
This file is kept for reference but is NOT the active pipeline.
The primary pipeline is main_rag.py (pure ML, no API required).

To use this approach: set GEMINI_API_KEY in .env, then run:
    python main.py
"""

import os
import sys
from pathlib import Path

import time

import pandas as pd
from dotenv import load_dotenv

# Load .env from repo root
load_dotenv(Path(__file__).parent.parent / ".env")

from retriever import get_retriever
from agent import triage

REPO_ROOT = Path(__file__).parent.parent
INPUT_CSV = REPO_ROOT / "support_tickets" / "support_tickets.csv"
OUTPUT_CSV = REPO_ROOT / "support_tickets" / "output.csv"
TOP_K = 5

COMPANY_MAP = {
    "hackerrank": "hackerrank",
    "claude": "claude",
    "visa": "visa",
}


def normalise_company(raw: str) -> str:
    """Map raw company field to a retriever key, or None for cross-domain."""
    if not isinstance(raw, str):
        return None
    key = raw.strip().lower()
    return COMPANY_MAP.get(key)


def run():
    if not INPUT_CSV.exists():
        print(f"ERROR: input file not found: {INPUT_CSV}", file=sys.stderr)
        sys.exit(1)

    if "GEMINI_API_KEY" not in os.environ:
        print("ERROR: GEMINI_API_KEY not set. Copy .env.example to .env and add your key.", file=sys.stderr)
        sys.exit(1)

    print("Building BM25 index over corpus...")
    retriever = get_retriever()
    print("Index ready.\n")

    df = pd.read_csv(INPUT_CSV)
    total = len(df)

    # Load existing output if present — skip rows already successfully processed
    existing = {}
    if OUTPUT_CSV.exists():
        try:
            prev = pd.read_csv(OUTPUT_CSV)
            for _, row in prev.iterrows():
                key = str(row.get("Issue", "")).strip()
                # Only keep rows that were actually processed (not error fallbacks)
                if "Unable to process" not in str(row.get("response", "")):
                    existing[key] = row.to_dict()
            if existing:
                print(f"Resuming: {len(existing)} tickets already processed, skipping them.\n")
        except Exception:
            pass

    results = []

    for idx, row in df.iterrows():
        issue = str(row.get("Issue", "") or "").strip()
        subject = str(row.get("Subject", "") or "").strip()
        company_raw = str(row.get("Company", "") or "").strip()
        company = normalise_company(company_raw)

        # Skip if already successfully processed
        if issue in existing:
            print(f"[{idx + 1}/{total}] SKIP (cached): {issue[:60]}...")
            results.append(existing[issue])
            continue

        print(f"[{idx + 1}/{total}] {company_raw or 'None'}: {issue[:60]}...")

        # Retrieve relevant docs
        query = f"{subject} {issue}".strip()
        docs = retriever.search(query, company, top_k=TOP_K)

        # Run triage
        out = triage(issue, subject, company_raw, docs)

        results.append({
            "Issue": issue,
            "Subject": subject,
            "Company": company_raw,
            "status": out["status"],
            "product_area": out["product_area"],
            "response": out["response"],
            "justification": out["justification"],
            "request_type": out["request_type"],
        })

        print(f"  -> status={out['status']}  request_type={out['request_type']}  area={out['product_area']}")

        # Save incrementally after each ticket
        pd.DataFrame(results).to_csv(OUTPUT_CSV, index=False)
        time.sleep(4)  # Avoid free-tier rate limits (~15 req/min max)

    output_df = pd.DataFrame(results)
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nDone. Output written to {OUTPUT_CSV}")


if __name__ == "__main__":
    run()
