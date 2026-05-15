"""
[ALTERNATIVE] Gemini Flash LLM-based triage agent.
This file is kept for reference but is NOT the active pipeline.
The primary pipeline is main_rag.py (pure ML, no API required).

To use this approach: set GEMINI_API_KEY in .env and run main.py instead.
"""

import json
import os
import re
import time

from google import genai
from google.genai import types

MODEL = "gemini-2.5-flash"
MAX_RETRIES = 3

SYSTEM_PROMPT = """You are a support triage agent for three products: HackerRank, Claude, and Visa.

You will receive a support ticket and relevant documentation excerpts. Read every doc carefully before deciding.

Output a JSON object with exactly these five fields:

{
  "status": "replied" or "escalated",
  "product_area": "<most relevant support category from the docs>",
  "response": "<user-facing answer grounded in the docs>",
  "justification": "<1-2 sentences explaining your decision>",
  "request_type": "product_issue" | "feature_request" | "bug" | "invalid"
}

Decision rules (apply in order):

1. ESCALATE (status="escalated") only for:
   - Live system-wide outage or service down reports
   - Active payment fraud, card theft, or identity theft in progress
   - Security vulnerabilities in the product itself
   - Issues completely outside all provided documentation with no safe answer possible

2. REPLY (status="replied") for everything else, including:
   - Account issues, billing, cancellation, feature questions
   - Privacy questions (e.g. deleting conversations, data export)
   - How-to questions answerable from the docs
   - Out-of-scope or nonsense requests → reply politely that it is out of scope, set request_type="invalid"

3. request_type values:
   - "product_issue" — user has trouble with a product feature
   - "feature_request" — user wants a new capability
   - "bug" — reproducible technical defect
   - "invalid" — off-topic, spam, nonsense, or malicious

4. NEVER hallucinate steps, URLs, phone numbers, or policies not in the provided docs.
5. If multiple questions, answer the primary one.
6. Output raw JSON only — no markdown, no extra text.
"""


def _build_context(docs: list[dict]) -> str:
    parts = []
    for i, doc in enumerate(docs, 1):
        parts.append(
            f"[Doc {i}] {doc['title']} (area: {doc['product_area']})\n{doc['body'][:1500]}"
        )
    return "\n\n---\n\n".join(parts)


def _build_user_message(issue: str, subject: str, company: str, docs: list[dict]) -> str:
    context = _build_context(docs)
    return f"""Support Ticket:
Company: {company or 'Unknown'}
Subject: {subject or '(none)'}
Issue: {issue}

Relevant Documentation:
{context}

Now produce the JSON triage output."""


def _get_client() -> genai.Client:
    return genai.Client(api_key=os.environ["GEMINI_API_KEY"])


def triage(issue: str, subject: str, company: str, docs: list[dict]) -> dict:
    """
    Call Gemini and return a dict with: status, product_area, response,
    justification, request_type.
    """
    client = _get_client()
    user_msg = _build_user_message(issue, subject, company, docs)

    result = None
    last_error = None

    for attempt in range(MAX_RETRIES):
        try:
            response = client.models.generate_content(
                model=MODEL,
                contents=user_msg,
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM_PROMPT,
                    temperature=1.0,  # Required when thinking is disabled
                    response_mime_type="application/json",
                    thinking_config=types.ThinkingConfig(thinking_budget=0),
                ),
            )
            raw = response.text.strip()
            raw = re.sub(r"^```(?:json)?\s*", "", raw)
            raw = re.sub(r"\s*```$", "", raw)
            result = json.loads(raw)
            break  # success
        except Exception as e:
            last_error = e
            if "429" in str(e) and attempt < MAX_RETRIES - 1:
                # Extract retry delay from error if present, else back off
                import re as _re
                m = _re.search(r'retryDelay.*?(\d+)s', str(e))
                wait = int(m.group(1)) + 5 if m else 30 * (attempt + 1)
                print(f"  [WARN] Rate limited, retrying in {wait}s...")
                time.sleep(wait)
            else:
                print(f"  [WARN] Gemini error: {e}")
                break

    if result is None:
        result = {
            "status": "escalated",
            "product_area": company.lower() if company and company != "nan" else "unknown",
            "response": "Unable to process this ticket automatically. Please escalate to a human agent.",
            "justification": f"Agent error: {last_error}",
            "request_type": "product_issue",
        }

    # Normalise to allowed enum values
    result["status"] = result.get("status", "escalated").lower()
    if result["status"] not in ("replied", "escalated"):
        result["status"] = "escalated"

    result["request_type"] = result.get("request_type", "product_issue").lower()
    if result["request_type"] not in ("product_issue", "feature_request", "bug", "invalid"):
        result["request_type"] = "product_issue"

    return result
