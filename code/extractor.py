"""
Extractive response generator.
Pulls the most relevant paragraphs from the top retrieved doc.
100% grounded — no hallucination possible.
"""

import re


def _split_paragraphs(text: str) -> list[str]:
    """Split doc body into non-empty paragraphs."""
    paras = re.split(r"\n{2,}", text)
    cleaned = []
    for p in paras:
        p = p.strip()
        # Skip image embeds, short noise, and markdown headers alone
        if not p or len(p) < 30:
            continue
        if re.match(r"^!\[.*\]\(.*\)$", p):
            continue
        if re.match(r"^#{1,3}\s+\S+$", p):
            continue
        cleaned.append(p)
    return cleaned


def _score_paragraph(para: str, query_tokens: set[str]) -> float:
    """Simple token overlap score between paragraph and query."""
    para_tokens = set(re.findall(r"\w+", para.lower()))
    if not para_tokens:
        return 0.0
    overlap = len(query_tokens & para_tokens)
    return overlap / (len(query_tokens) + 1)


def _clean(text: str) -> str:
    """Strip markdown syntax for a cleaner user-facing response."""
    # Remove metadata lines like _Last updated: ..._ or _Last modified: ..._
    text = re.sub(r"_Last (updated|modified):.*?_\n?", "", text)
    # Remove image embeds
    text = re.sub(r"!\[.*?\]\(.*?\)", "", text)
    # Links → just the label
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    # Bold/italic
    text = re.sub(r"\*{1,2}([^*]+)\*{1,2}", r"\1", text)
    # Inline code
    text = re.sub(r"`([^`]+)`", r"\1", text)
    # Markdown headers → plain text
    text = re.sub(r"^#{1,4}\s+", "", text, flags=re.MULTILINE)
    # Collapse excess blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Strip leading/trailing whitespace on each line
    lines = [l.strip() for l in text.splitlines()]
    # Drop lines that are only punctuation or very short noise
    lines = [l for l in lines if len(l) > 2 or l == ""]
    return "\n".join(lines).strip()


def extract_response(
    issue: str,
    docs: list[dict],
    status: str,
    max_paragraphs: int = 3,
    max_chars: int = 800,
) -> tuple[str, str]:
    """
    Returns (response, justification).

    For escalated tickets: returns a standard escalation message.
    For replied tickets: extracts the most relevant paragraphs from the top doc.
    For invalid tickets (detected by caller): returns an out-of-scope message.
    """
    if status == "escalated":
        top = docs[0] if docs else {}
        area = top.get("product_area", "support")
        title = top.get("title", "")
        response = (
            "This issue requires immediate attention from our support team. "
            "Please contact support directly so a specialist can assist you promptly."
        )
        justification = (
            f"Escalated because the ticket indicates a critical issue "
            f"(outage, fraud, or security concern) that requires human intervention. "
            f"Most relevant document: '{title}'."
        ) if title else (
            "Escalated: issue is critical or cannot be resolved from available documentation."
        )
        return response, justification

    if not docs:
        return (
            "I'm sorry, I couldn't find relevant information for your query. "
            "Please contact support for further assistance.",
            "No relevant documentation found for this ticket.",
        )

    top_doc = docs[0]
    title = top_doc.get("title", "")
    body = top_doc.get("body", "")
    area = top_doc.get("product_area", "")

    query_tokens = set(re.findall(r"\w+", issue.lower()))
    paragraphs = _split_paragraphs(body)

    if not paragraphs:
        snippet = _clean(body[:max_chars])
        return (
            snippet,
            f"Based on '{title}' ({area}) from the support documentation.",
        )

    # Score and pick best paragraphs
    scored = sorted(
        enumerate(paragraphs),
        key=lambda x: _score_paragraph(x[1], query_tokens),
        reverse=True,
    )

    # Always include the first paragraph (usually the overview) + top-scored ones
    selected_indices = {0}
    for i, _ in scored[:max_paragraphs]:
        selected_indices.add(i)

    # Re-order by original position for readable flow
    ordered = sorted(selected_indices)
    selected = [paragraphs[i] for i in ordered if i < len(paragraphs)]

    response = _clean("\n\n".join(selected))
    if len(response) > max_chars:
        response = response[:max_chars].rsplit(" ", 1)[0] + "..."

    justification = (
        f"Replied using '{title}' ({area}). "
        f"Answer extracted directly from the support documentation "
        f"(similarity score: {top_doc.get('score', 0):.2f})."
    )

    return response, justification


def out_of_scope_response(issue: str) -> tuple[str, str]:
    """Standard response for invalid/out-of-scope tickets."""
    response = (
        "I'm sorry, your request is outside the scope of what I can help with. "
        "I can only assist with questions related to HackerRank, Claude, or Visa support. "
        "Please reach out to the appropriate team for other queries."
    )
    justification = (
        "Ticket is unrelated to any supported product domain or contains a "
        "request that cannot be fulfilled by this support agent."
    )
    return response, justification
