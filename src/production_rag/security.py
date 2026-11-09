from __future__ import annotations

import re
from dataclasses import dataclass, field

from .telemetry import SECURITY_EVENTS, log_event


EMAIL_RE = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.I)
SSN_RE = re.compile(r"\b(?!000|666|9\d\d)\d{3}-\d{2}-\d{4}\b")
CARD_RE = re.compile(r"(?<!\d)(?:\d[ -]?){13,19}(?!\d)")
PHONE_RE = re.compile(r"(?<!\w)(?:\+?\d{1,3}[ .-]?)?(?:\(?\d{3,5}\)?[ .-]?)\d{3,5}[ .-]?\d{3,5}(?!\w)")
AADHAAR_RE = re.compile(r"(?<!\d)(?:\d{4}[ -]?){2}\d{4}(?!\d)")
PAN_RE = re.compile(r"\b[A-Z]{5}[0-9]{4}[A-Z]\b", re.I)
INJECTION_PATTERNS = (
    re.compile(r"ignore\s+(?:all\s+)?(?:previous|prior|above)\s+instructions", re.I),
    re.compile(r"reveal\s+(?:the\s+)?(?:system|developer)\s+prompt", re.I),
    re.compile(r"(?:show|print|dump|exfiltrate)\s+(?:(?:your|the)\s+)?(?:hidden\s+)?(?:system|developer)\s+prompt", re.I),
    re.compile(r"(?:you\s+are\s+now|pretend\s+(?:you\s+are\s+)?|act\s+as\s+)\s*(?:unrestricted|a\s+different\s+assistant)", re.I),
    re.compile(r"disregard\s+(?:all\s+)?(?:safety|security|policy)\s+(?:rules|instructions)", re.I),
    re.compile(r"(?:developer|system)\s+message\s*:\s*", re.I),
    re.compile(r"\b(?:jailbreak|prompt\s+injection)\b.{0,100}\b(?:bypass|override|ignore|reveal|disclose)\b", re.I),
    re.compile(r"\bpretend\b.{0,60}\b(?:unrestricted|ignore\s+(?:all\s+)?instructions|reveal\s+secrets)\b", re.I),
)


@dataclass
class SecurityResult:
    allowed: bool
    sanitized_text: str
    blocked_reason: str | None = None
    redactions: list[str] = field(default_factory=list)
    events: list[str] = field(default_factory=list)


class SecurityInterceptor:
    def sanitize(self, text: str) -> tuple[str, list[str]]:
        redactions: list[str] = []
        replacements = (
            (EMAIL_RE, "[REDACTED_EMAIL]", "email"),
            (SSN_RE, "[REDACTED_SSN]", "ssn"),
            (CARD_RE, "[REDACTED_PAYMENT_CARD]", "payment_card"),
            (AADHAAR_RE, "[REDACTED_AADHAAR]", "aadhaar"),
            (PAN_RE, "[REDACTED_PAN]", "pan"),
            (PHONE_RE, "[REDACTED_PHONE]", "phone"),
        )
        output = text
        for pattern, replacement, label in replacements:
            before = output
            if label == "payment_card":
                output = pattern.sub(lambda m: replacement if sum(c.isdigit() for c in m.group()) >= 13 else m.group(), output)
            else:
                output = pattern.sub(replacement, output)
            if output != before:
                redactions.append(label)
        return output, redactions

    def inspect(self, text: str) -> SecurityResult:
        normalized = re.sub(r"\s+", " ", text).strip()
        for pattern in INJECTION_PATTERNS:
            match = pattern.search(normalized)
            if match:
                SECURITY_EVENTS.labels(kind="prompt_injection").inc()
                log_event("security_block", kind="prompt_injection", pattern=pattern.pattern[:48])
                sanitized, redactions = self.sanitize(text)
                return SecurityResult(False, sanitized, "prompt_injection", redactions, ["prompt_injection"])
        sanitized, redactions = self.sanitize(text)
        if redactions:
            for label in set(redactions):
                SECURITY_EVENTS.labels(kind=label).inc()
            log_event("pii_redacted", categories=sorted(set(redactions)))
        return SecurityResult(True, sanitized, redactions=redactions, events=redactions)

    @staticmethod
    def sanitize_untrusted_document(text: str) -> str:
        """Mark retrieved text as evidence, never as instructions."""
        return f"<untrusted_evidence>\n{text}\n</untrusted_evidence>"
