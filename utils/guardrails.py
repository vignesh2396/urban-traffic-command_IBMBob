"""
Guardrails — Rural Healthcare AI Navigation System
===================================================
Implements all 9 guardrail categories:
  1. Input Guardrails
  2. Output Guardrails
  3. Tool Execution Guardrails
  4. Data Guardrails
  5. Security Guardrails
  6. Operational Guardrails
  7. Human-in-the-Loop Guardrails
  8. Multi-Agent Guardrails
  9. Ethical & Compliance Guardrails
"""

from __future__ import annotations

import hashlib
import re
import time
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger(__name__)

# ── PII patterns ──────────────────────────────────────────────────────────────
_PII_PATTERNS: list[tuple[str, str]] = [
    (r"\b\d{10,12}\b", "PHONE"),
    (r"\b[A-Z]{5}\d{4}[A-Z]\b", "PAN_CARD"),
    (r"\b\d{4}[\s-]?\d{4}[\s-]?\d{4}\b", "AADHAAR"),
    (r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b", "EMAIL"),
    (r"\b\d{6}\b", "PINCODE"),
]

_PROMPT_INJECTION_PATTERNS = [
    r"ignore\s+(previous|above|all)\s+instructions",
    r"forget\s+(everything|all|your)\s",
    r"you\s+are\s+now\s+(a|an)\s+",
    r"act\s+as\s+if\s+you\s+are",
    r"disregard\s+(your|all|previous)",
    r"jailbreak",
    r"DAN\s+mode",
    r"pretend\s+(you\s+are|to\s+be)",
    r"system\s*:\s*",
]

_MALICIOUS_PATTERNS = [
    r"<script[^>]*>",
    r"javascript\s*:",
    r"on\w+\s*=",
    r"eval\s*\(",
    r"exec\s*\(",
    r"__import__",
    r"subprocess",
    r"os\.system",
]

_INAPPROPRIATE_KEYWORDS = [
    "suicide", "self-harm", "kill yourself", "overdose intentionally",
    "sell drugs", "illegal substances",
]

_TOXIC_PATTERNS = [
    r"\b(hate|kill|stupid|idiot|moron|worthless|die)\b",
]

_SENSITIVE_OUTPUT_PATTERNS = [
    r"api[_\s]?key\s*[:=]\s*\S+",
    r"password\s*[:=]\s*\S+",
    r"secret\s*[:=]\s*\S+",
    r"token\s*[:=]\s*\S+",
]

# ── Dataclasses ───────────────────────────────────────────────────────────────

@dataclass
class GuardrailResult:
    passed: bool
    violations: list[str] = field(default_factory=list)
    sanitized_text: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def blocked(self) -> bool:
        return not self.passed


@dataclass
class OperationalLimits:
    max_input_tokens: int = 2000
    max_output_tokens: int = 4000
    max_turns_per_session: int = 50
    max_execution_seconds: float = 120.0
    max_agent_iterations: int = 10
    api_rate_limit_per_min: int = 30


# ── Singleton operational tracker ─────────────────────────────────────────────

class _OperationalTracker:
    """Tracks rate limits, token counts, and execution times."""

    def __init__(self) -> None:
        self._call_timestamps: list[float] = []
        self._total_input_tokens: int = 0
        self._total_output_tokens: int = 0
        self._total_cost_usd: float = 0.0
        self._agent_iteration_counts: dict[str, int] = {}
        self._intervention_count: int = 0
        self._violations_log: list[dict] = []

    def record_call(self, session_id: str = "") -> None:
        now = time.time()
        self._call_timestamps = [t for t in self._call_timestamps if now - t < 60]
        self._call_timestamps.append(now)

    def calls_last_minute(self) -> int:
        now = time.time()
        return sum(1 for t in self._call_timestamps if now - t < 60)

    def record_tokens(self, input_tokens: int, output_tokens: int, cost_usd: float = 0.0) -> None:
        self._total_input_tokens += input_tokens
        self._total_output_tokens += output_tokens
        self._total_cost_usd += cost_usd

    def record_agent_iteration(self, agent_name: str) -> int:
        self._agent_iteration_counts[agent_name] = self._agent_iteration_counts.get(agent_name, 0) + 1
        return self._agent_iteration_counts[agent_name]

    def record_intervention(self, violation_type: str, details: str) -> None:
        self._intervention_count += 1
        self._violations_log.append({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "type": violation_type,
            "details": details,
        })

    def get_stats(self) -> dict[str, Any]:
        return {
            "calls_last_minute": self.calls_last_minute(),
            "total_input_tokens": self._total_input_tokens,
            "total_output_tokens": self._total_output_tokens,
            "estimated_cost_usd": round(self._total_cost_usd, 4),
            "guardrail_interventions": self._intervention_count,
            "agent_iterations": dict(self._agent_iteration_counts),
            "violations_log": self._violations_log[-20:],
        }


_tracker = _OperationalTracker()
_limits = OperationalLimits()


# ═══════════════════════════════════════════════════════════════════════════════
# 1. INPUT GUARDRAILS
# ═══════════════════════════════════════════════════════════════════════════════

def validate_input(text: str, session_id: str = "") -> GuardrailResult:
    """Full input validation pipeline."""
    violations: list[str] = []
    sanitized = text.strip()

    # Length / token proxy
    if len(sanitized) > _limits.max_input_tokens * 4:
        violations.append("INPUT_TOO_LONG")
        _tracker.record_intervention("INPUT_TOO_LONG", f"len={len(sanitized)}")

    # Prompt injection
    for pattern in _PROMPT_INJECTION_PATTERNS:
        if re.search(pattern, sanitized, re.IGNORECASE):
            violations.append("PROMPT_INJECTION")
            _tracker.record_intervention("PROMPT_INJECTION", pattern)
            break

    # Malicious code injection
    for pattern in _MALICIOUS_PATTERNS:
        if re.search(pattern, sanitized, re.IGNORECASE):
            violations.append("MALICIOUS_CODE")
            _tracker.record_intervention("MALICIOUS_CODE", pattern)
            break

    # PII detection (mask, do not block)
    pii_found: list[str] = []
    for pii_pattern, pii_type in _PII_PATTERNS:
        if re.search(pii_pattern, sanitized):
            pii_found.append(pii_type)
            sanitized = re.sub(pii_pattern, f"[{pii_type}_REDACTED]", sanitized)

    if pii_found:
        _tracker.record_intervention("PII_DETECTED", str(pii_found))

    # Inappropriate content
    for kw in _INAPPROPRIATE_KEYWORDS:
        if kw in sanitized.lower():
            violations.append("INAPPROPRIATE_CONTENT")
            _tracker.record_intervention("INAPPROPRIATE_CONTENT", kw)
            break

    passed = not any(v in violations for v in ["PROMPT_INJECTION", "MALICIOUS_CODE", "INAPPROPRIATE_CONTENT"])
    return GuardrailResult(
        passed=passed,
        violations=violations,
        sanitized_text=sanitized,
        metadata={"pii_detected": pii_found, "session_id": session_id},
    )


# ═══════════════════════════════════════════════════════════════════════════════
# 2. OUTPUT GUARDRAILS
# ═══════════════════════════════════════════════════════════════════════════════

def validate_output(text: str, context: str = "") -> GuardrailResult:
    """Validate LLM output before displaying to user."""
    violations: list[str] = []
    sanitized = text.strip()

    # Sensitive info leakage
    for pattern in _SENSITIVE_OUTPUT_PATTERNS:
        if re.search(pattern, sanitized, re.IGNORECASE):
            violations.append("SENSITIVE_INFO_LEAK")
            sanitized = re.sub(pattern, "[REDACTED]", sanitized, flags=re.IGNORECASE)
            _tracker.record_intervention("SENSITIVE_INFO_LEAK", pattern)

    # Toxicity
    for pattern in _TOXIC_PATTERNS:
        if re.search(pattern, sanitized, re.IGNORECASE):
            violations.append("TOXIC_CONTENT")
            _tracker.record_intervention("TOXIC_CONTENT", pattern)
            break

    # Hallucination heuristic: check for unsupported absolute claims about healthcare
    hallucination_markers = [
        r"\b(definitely|100%|guaranteed)\s+(cure|treat|heals?)\b",
        r"this\s+drug\s+will\s+definitely",
    ]
    for marker in hallucination_markers:
        if re.search(marker, sanitized, re.IGNORECASE):
            violations.append("POTENTIAL_HALLUCINATION")
            _tracker.record_intervention("POTENTIAL_HALLUCINATION", marker)
            break

    # Minimum response quality
    if len(sanitized) < 10:
        violations.append("EMPTY_RESPONSE")

    passed = "TOXIC_CONTENT" not in violations and "SENSITIVE_INFO_LEAK" not in violations
    return GuardrailResult(
        passed=passed,
        violations=violations,
        sanitized_text=sanitized,
        metadata={"output_length": len(sanitized)},
    )


# ═══════════════════════════════════════════════════════════════════════════════
# 3. TOOL EXECUTION GUARDRAILS
# ═══════════════════════════════════════════════════════════════════════════════

_ALLOWED_TOOLS = {
    "assessment_agent", "facility_discovery_agent", "referral_agent",
    "rag_knowledge_agent", "appointment_agent", "emergency_agent",
}
_RESTRICTED_ACTIONS = {"delete_patient", "modify_database_schema", "export_all_records"}


def validate_tool_call(tool_name: str, action: str, params: dict) -> GuardrailResult:
    violations: list[str] = []
    if tool_name not in _ALLOWED_TOOLS:
        violations.append(f"UNAUTHORIZED_TOOL:{tool_name}")
        _tracker.record_intervention("UNAUTHORIZED_TOOL", tool_name)
    if action in _RESTRICTED_ACTIONS:
        violations.append(f"RESTRICTED_ACTION:{action}")
        _tracker.record_intervention("RESTRICTED_ACTION", action)
    return GuardrailResult(passed=not violations, violations=violations, sanitized_text="", metadata=params)


# ═══════════════════════════════════════════════════════════════════════════════
# 4. DATA GUARDRAILS
# ═══════════════════════════════════════════════════════════════════════════════

def mask_pii(text: str) -> str:
    """Mask PII in data fields for storage/logging."""
    masked = text
    for pattern, label in _PII_PATTERNS:
        masked = re.sub(pattern, f"[{label}]", masked)
    return masked


def validate_data_access(role: str, resource: str) -> GuardrailResult:
    """Role-based data access control."""
    _access_matrix = {
        "patient": {"own_records", "facilities", "knowledge_base"},
        "doctor": {"patient_records", "facilities", "knowledge_base", "referrals"},
        "admin": {"all"},
    }
    allowed = _access_matrix.get(role, set())
    if "all" in allowed or resource in allowed:
        return GuardrailResult(passed=True, sanitized_text=resource)
    _tracker.record_intervention("UNAUTHORIZED_DATA_ACCESS", f"role={role} resource={resource}")
    return GuardrailResult(passed=False, violations=[f"RBAC_DENIED:{resource}"], sanitized_text="")


# ═══════════════════════════════════════════════════════════════════════════════
# 5. SECURITY GUARDRAILS
# ═══════════════════════════════════════════════════════════════════════════════

_JAILBREAK_PATTERNS = [
    r"do\s+anything\s+now",
    r"bypass\s+(your\s+)?(safety|filter|restriction)",
    r"ignore\s+(your\s+)?(safety|ethical|moral)",
    r"evil\s+(mode|persona|bot)",
    r"no\s+restrictions",
]

_SQL_INJECTION_PATTERNS = [
    r"('\s*OR\s*'1'\s*=\s*'1)",
    r"(;\s*DROP\s+TABLE)",
    r"(UNION\s+SELECT)",
    r"(--\s*$)",
    r"(\/\*.*\*\/)",
]


def security_check(text: str) -> GuardrailResult:
    """Combined security check for jailbreak and SQL injection."""
    violations: list[str] = []

    for pattern in _JAILBREAK_PATTERNS:
        if re.search(pattern, text, re.IGNORECASE):
            violations.append("JAILBREAK_ATTEMPT")
            _tracker.record_intervention("JAILBREAK_ATTEMPT", pattern)
            break

    for pattern in _SQL_INJECTION_PATTERNS:
        if re.search(pattern, text, re.IGNORECASE):
            violations.append("SQL_INJECTION")
            _tracker.record_intervention("SQL_INJECTION", pattern)
            break

    return GuardrailResult(passed=not violations, violations=violations, sanitized_text=text)


# ═══════════════════════════════════════════════════════════════════════════════
# 6. OPERATIONAL GUARDRAILS
# ═══════════════════════════════════════════════════════════════════════════════

def check_rate_limit() -> GuardrailResult:
    calls = _tracker.calls_last_minute()
    if calls >= _limits.api_rate_limit_per_min:
        _tracker.record_intervention("RATE_LIMIT_EXCEEDED", f"calls={calls}")
        return GuardrailResult(passed=False, violations=["RATE_LIMIT_EXCEEDED"],
                               metadata={"calls_last_minute": calls})
    _tracker.record_call()
    return GuardrailResult(passed=True, metadata={"calls_last_minute": calls + 1})


def check_agent_iterations(agent_name: str) -> GuardrailResult:
    count = _tracker.record_agent_iteration(agent_name)
    if count > _limits.max_agent_iterations:
        _tracker.record_intervention("MAX_ITERATIONS_EXCEEDED", f"agent={agent_name} count={count}")
        return GuardrailResult(passed=False, violations=["MAX_ITERATIONS_EXCEEDED"],
                               metadata={"agent": agent_name, "iterations": count})
    return GuardrailResult(passed=True, metadata={"agent": agent_name, "iterations": count})


def record_token_usage(input_tokens: int, output_tokens: int, cost_usd: float = 0.0) -> None:
    _tracker.record_tokens(input_tokens, output_tokens, cost_usd)


# ═══════════════════════════════════════════════════════════════════════════════
# 7. HUMAN-IN-THE-LOOP GUARDRAILS
# ═══════════════════════════════════════════════════════════════════════════════

_HITL_TRIGGERS = {
    "emergency": True,
    "referral_outside_state": True,
    "critical_medication": True,
    "surgical_recommendation": True,
}


def requires_human_approval(action_type: str) -> bool:
    return _HITL_TRIGGERS.get(action_type, False)


def validate_human_decision(decision: dict) -> GuardrailResult:
    required_fields = {"approver_id", "decision", "timestamp"}
    missing = required_fields - set(decision.keys())
    if missing:
        return GuardrailResult(passed=False, violations=[f"MISSING_HITL_FIELDS:{missing}"])
    if decision["decision"] not in ("APPROVED", "REJECTED", "ESCALATED"):
        return GuardrailResult(passed=False, violations=["INVALID_HITL_DECISION"])
    return GuardrailResult(passed=True)


# ═══════════════════════════════════════════════════════════════════════════════
# 8. MULTI-AGENT GUARDRAILS
# ═══════════════════════════════════════════════════════════════════════════════

_AGENT_PERMISSIONS: dict[str, set[str]] = {
    "supervisor": {"assessment", "discovery", "referral", "knowledge", "appointment", "emergency"},
    "assessment": {"knowledge"},
    "discovery": set(),
    "referral": {"discovery"},
    "knowledge": set(),
    "appointment": {"discovery"},
    "emergency": set(),
}


def validate_agent_handoff(from_agent: str, to_agent: str) -> GuardrailResult:
    allowed = _AGENT_PERMISSIONS.get(from_agent, set())
    if to_agent not in allowed:
        _tracker.record_intervention("INVALID_AGENT_HANDOFF", f"{from_agent}→{to_agent}")
        return GuardrailResult(passed=False, violations=[f"UNAUTHORIZED_HANDOFF:{from_agent}→{to_agent}"])
    return GuardrailResult(passed=True)


def detect_agent_loop(agent_trace: list[dict], window: int = 5) -> GuardrailResult:
    if len(agent_trace) < window:
        return GuardrailResult(passed=True)
    recent_agents = [step.get("agent") for step in agent_trace[-window:]]
    if len(set(recent_agents)) == 1:
        _tracker.record_intervention("AGENT_LOOP_DETECTED", str(recent_agents))
        return GuardrailResult(passed=False, violations=["AGENT_LOOP_DETECTED"],
                               metadata={"repeated_agent": recent_agents[0]})
    return GuardrailResult(passed=True)


# ═══════════════════════════════════════════════════════════════════════════════
# 9. ETHICAL & COMPLIANCE GUARDRAILS
# ═══════════════════════════════════════════════════════════════════════════════

_BIAS_INDICATORS = [
    r"\b(women|men|female|male)\s+are\s+(less|more|not)\s+(capable|smart|intelligent)\b",
    r"\b(poor|rich|caste|religion)\s+(people|patients)\s+(don't|cannot|should\s+not)\b",
]

_POLICY_VIOLATIONS = [
    r"do\s+not\s+visit\s+a\s+(doctor|hospital)",
    r"avoid\s+(medication|treatment|surgery)\s+completely",
    r"traditional\s+remedy\s+instead\s+of\s+hospital",
]


def ethics_compliance_check(text: str) -> GuardrailResult:
    violations: list[str] = []
    for pattern in _BIAS_INDICATORS:
        if re.search(pattern, text, re.IGNORECASE):
            violations.append("BIAS_DETECTED")
            _tracker.record_intervention("BIAS_DETECTED", pattern)
            break
    for pattern in _POLICY_VIOLATIONS:
        if re.search(pattern, text, re.IGNORECASE):
            violations.append("POLICY_VIOLATION")
            _tracker.record_intervention("POLICY_VIOLATION", pattern)
            break
    return GuardrailResult(passed=not violations, violations=violations, sanitized_text=text)


def get_audit_trail() -> list[dict]:
    return _tracker._violations_log


def get_operational_stats() -> dict[str, Any]:
    return _tracker.get_stats()


# ═══════════════════════════════════════════════════════════════════════════════
# Convenience: run_all_input_guardrails / run_all_output_guardrails
# ═══════════════════════════════════════════════════════════════════════════════

def run_all_input_guardrails(text: str, session_id: str = "") -> GuardrailResult:
    """Run input validation + security check in sequence."""
    rate = check_rate_limit()
    if rate.blocked:
        return rate
    sec = security_check(text)
    if sec.blocked:
        return sec
    return validate_input(text, session_id)


def run_all_output_guardrails(text: str, context: str = "") -> GuardrailResult:
    out = validate_output(text, context)
    if out.passed:
        eth = ethics_compliance_check(text)
        if eth.blocked:
            return eth
    return out
