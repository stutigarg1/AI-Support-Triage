"""
Rule-based classifier for status and request_type.
Uses keyword signals + retrieval confidence score — no LLM needed.
"""

import re

# --- Escalation signals ---

# System-wide outage / service down patterns
OUTAGE_PATTERNS = [
    r"\bsite\s+is\s+down\b",
    r"\bpages?\s+(are\s+)?not\s+accessible\b",
    r"\bcompletely\s+(down|broken|unavailable)\b",
    r"\bnone\s+of\s+the\s+(pages?|requests?|submissions?)\b",
    r"\bstopped\s+working\s+completely\b",
    r"\ball\s+requests?\s+are\s+fail(ing)?\b",
    r"\bresume\s+builder\s+is\s+down\b",
    r"\bplatform\s+is\s+(down|unavailable|unreachable)\b",
    r"\bdown\s+for\s+everyone\b",
    r"\bglobal\s+outage\b",
]

# Active fraud / security incidents requiring immediate human escalation
FRAUD_PATTERNS = [
    r"\bidentity\s+(has\s+been\s+)?stolen\b",
    r"\bidentity\s+theft\b",
    r"\bmy\s+account\s+was\s+hacked\b",
    r"\bunauthorized\s+(access|transaction|charge)\b",
    r"\bsecurity\s+vulnerability\b",
    r"\bsecurity\s+bug\b",
    r"\bfound\s+a\s+(major\s+)?vulnerability\b",
]

# Malicious / clearly out-of-scope requests
MALICIOUS_PATTERNS = [
    r"\bdelete\s+all\s+files\b",
    r"\brm\s+-rf\b",
    r"\bformat\s+(the\s+)?(hard\s+)?drive\b",
    r"\bhow\s+to\s+hack\b",
    r"\bsql\s+injection\b",
]

# --- request_type signals ---

BUG_PATTERNS = [
    r"\bnot\s+working\b",
    r"\bisn'?t\s+working\b",
    r"\bbroken\b",
    r"\bbug\b",
    r"\bcrash(ing|ed)?\b",
    r"\berror\b",
    r"\bfail(ing|ed|ure)?\b",
    r"\bglitch\b",
    r"\bdown\b",
    r"\bstopped\s+working\b",
    r"\bcan'?t\s+(access|load|open|log\s+in|submit)\b",
    r"\bunable\s+to\b",
    r"\bblocker\b",
]

FEATURE_REQUEST_PATTERNS = [
    r"\bwould\s+(like|love)\s+to\b",
    r"\bfeature\s+request\b",
    r"\bcan\s+you\s+add\b",
    r"\bplease\s+add\b",
    r"\bwish\s+(you\s+had|there\s+was)\b",
    r"\bsuggestion\b",
    r"\benhancement\b",
    r"\bnew\s+feature\b",
    r"\bin\s+the\s+future\b",
    r"\bit\s+would\s+be\s+(great|nice|helpful)\s+if\b",
]

# Low similarity threshold — below this we consider the ticket unresolvable
SIMILARITY_THRESHOLD_REPLY = 0.25
# Below this + vague text → escalate as unresolvable
SIMILARITY_THRESHOLD_ESCALATE = 0.15


def _matches_any(text: str, patterns: list[str]) -> bool:
    t = text.lower()
    return any(re.search(p, t) for p in patterns)


def is_too_vague(issue: str, subject: str) -> bool:
    """True if the ticket body is too short/vague to triage meaningfully.
    Checks issue text alone — subject is often noisy boilerplate."""
    issue_words = len(re.findall(r"\w+", issue.strip()))
    # Also treat generic filler subjects + short issue as vague
    generic_subjects = {"help needed", "help", "urgent", "issue", "problem", "hi", "hello", ""}
    subject_is_filler = subject.strip().lower() in generic_subjects
    return issue_words < 6 or (issue_words < 10 and subject_is_filler)


def classify_status(issue: str, subject: str, top_doc_score: float) -> str:
    """
    Returns 'escalated' or 'replied'.

    Escalation rules (in priority order):
    1. Malicious request → replied (with out-of-scope message, handled by request_type)
    2. Active fraud / security vulnerability → escalated
    3. System-wide outage detected → escalated
    4. Top retrieved doc score too low → escalated (can't answer safely)
    5. Everything else → replied
    """
    combined = f"{subject} {issue}".lower()

    # Malicious → we reply with out-of-scope, not escalate
    if _matches_any(combined, MALICIOUS_PATTERNS):
        return "replied"

    # Active fraud / security vulnerability → must escalate to human
    if _matches_any(combined, FRAUD_PATTERNS):
        return "escalated"

    # System-wide outage → escalate
    if _matches_any(combined, OUTAGE_PATTERNS):
        return "escalated"

    # Too vague to safely answer → escalate for human follow-up
    if is_too_vague(issue, subject):
        return "escalated"

    # No relevant doc found → can't safely answer
    if top_doc_score < SIMILARITY_THRESHOLD_ESCALATE:
        return "escalated"

    return "replied"


def classify_request_type(issue: str, subject: str, top_doc_score: float) -> str:
    """
    Returns one of: product_issue, feature_request, bug, invalid.
    """
    combined = f"{subject} {issue}".lower()

    # Malicious or very off-topic → invalid
    if _matches_any(combined, MALICIOUS_PATTERNS):
        return "invalid"

    # Very low similarity + not matching any product domain → invalid
    if top_doc_score < SIMILARITY_THRESHOLD_REPLY:
        return "invalid"

    # Feature request signals
    if _matches_any(combined, FEATURE_REQUEST_PATTERNS):
        return "feature_request"

    # Bug signals
    if _matches_any(combined, BUG_PATTERNS):
        return "bug"

    return "product_issue"


def infer_company(issue: str, subject: str) -> str | None:
    """Infer company from ticket text when company field is None."""
    combined = f"{subject} {issue}".lower()
    if "hackerrank" in combined:
        return "hackerrank"
    if "claude" in combined or "anthropic" in combined:
        return "claude"
    if "visa" in combined:
        return "visa"
    return None
