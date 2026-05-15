# HackerRank Orchestrate Hackathon (May 1-2, 2026)

## Support Triage Agent

Terminal-based AI agent that triages support tickets across HackerRank, Claude, and Visa.

**Primary approach: Pure ML/RAG, no LLM API, no rate limits, works fully offline.**

I built a four-stage pipeline. First, a semantic retriever loads all 770 markdown support docs across HackerRank, Claude, and Visa, encodes them using a local sentence-transformer model, and builds a per-company vector index. 

Second, a query comes in from a ticket, gets matched against that index using cosine similarity, and returns the top 5 most relevant documents. 

Third, a rule-based classifier looks at the ticket text and retrieval score to decide status and request type. 

Fourth, an extractive response generator pulls the most relevant paragraph directly from the top retrieved document and returns it as the response.


## Architecture

```
support_tickets.csv
       ↓
  [embedder.py]     Semantic search via sentence-transformers (local, ~80MB model)
       ↓
  [classifier.py]   Rule-based status + request_type classification
       ↓
  [extractor.py]    Extractive response from top retrieved doc (zero hallucination)
       ↓
  output.csv
```

**Why this design?**
- **Semantic retrieval** (all-MiniLM-L6-v2) outperforms keyword search for paraphrased queries
- **Extractive responses** are 100% grounded in the corpus impossible to hallucinate
- **Rule-based escalation** is explicit and auditable, not a black box
- **No API dependency**- reproducible, deterministic, no quota limits

## Setup

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Run the agent:
   ```bash
   cd code
   python main_rag.py
   ```

Output is written to `support_tickets/output.csv`. Runtime: ~15 seconds for 29 tickets.

## Files

| File | Purpose |
|---|---|
| `main_rag.py` | **Primary entry point** - RAG pipeline |
| `embedder.py` | Loads all `.md` docs, builds semantic vector index |
| `classifier.py` | Rule-based escalation logic + request_type detection |
| `extractor.py` | Extractive response generator from top retrieved doc |
| `requirements.txt` | Python dependencies |
| `main.py` | *Alternative* - Gemini LLM pipeline (requires API key) |
| `agent.py` | *Alternative* - Gemini call handler |
| `retriever.py` | *Alternative* - BM25 retriever used by Gemini pipeline |

## Output columns

| Column | Values |
|---|---|
| `status` | `replied` or `escalated` |
| `product_area` | Support category derived from corpus folder structure |
| `response` | Extractive answer from the relevant support doc |
| `justification` | Retrieval score + source doc cited |
| `request_type` | `product_issue`, `feature_request`, `bug`, `invalid` |

## Escalation logic

The agent escalates when any of these conditions are true:
- System-wide outage detected (e.g. "site is down", "all requests failing")
- Active fraud or identity theft reported
- Security vulnerability disclosed
- Ticket is too vague to answer safely (< 6 words, no context)
- No relevant doc found (similarity score below threshold)
