"""Local, deterministic tools for the refund-agent demo.

Pure functions over an in-memory order table — no network, no state, so every
run is reproducible and the demo works offline.
"""

from __future__ import annotations

from typing import Any

AUTO_REFUND_LIMIT_USD = 500

ORDERS: dict[str, dict[str, Any]] = {
    "TEST-1": {"order_id": "TEST-1", "total_usd": 842.00, "status": "shipped", "customer": "c-311"},
    "TEST-2": {"order_id": "TEST-2", "total_usd": 129.99, "status": "delivered", "customer": "c-204"},
    "TEST-404": {},  # simulates a missing order
}


def lookup_order(order_id: str) -> dict[str, Any]:
    order = ORDERS.get(order_id)
    if not order:
        return {"error": "order_not_found", "order_id": order_id}
    return order


def check_refund_policy(amount_usd: float) -> dict[str, Any]:
    if amount_usd <= AUTO_REFUND_LIMIT_USD:
        return {"decision": "auto_approve", "limit_usd": AUTO_REFUND_LIMIT_USD}
    return {
        "decision": "requires_escalation",
        "limit_usd": AUTO_REFUND_LIMIT_USD,
        "reason": f"amount ${amount_usd:.2f} exceeds auto-refund limit ${AUTO_REFUND_LIMIT_USD}",
    }


def issue_refund(order_id: str, amount_usd: float) -> dict[str, Any]:
    return {"refund_id": f"rf-{order_id}", "order_id": order_id, "amount_usd": amount_usd, "ok": True}


def escalate(order_id: str, reason: str) -> dict[str, Any]:
    return {"ticket_id": f"esc-{order_id}", "order_id": order_id, "queued": True}


TOOLS: dict[str, Any] = {
    "lookup_order": lookup_order,
    "check_refund_policy": check_refund_policy,
    "issue_refund": issue_refund,
    "escalate": escalate,
}

TOOL_SPECS = [
    {
        "name": "lookup_order",
        "description": "Fetch an order by id.",
        "parameters": {"order_id": "string"},
    },
    {
        "name": "check_refund_policy",
        "description": "Check whether an amount can be auto-refunded or must be escalated.",
        "parameters": {"amount_usd": "number"},
    },
    {
        "name": "issue_refund",
        "description": "Issue a refund for an order. Only after policy allows it.",
        "parameters": {"order_id": "string", "amount_usd": "number"},
    },
    {
        "name": "escalate",
        "description": "Escalate to a human when policy requires it.",
        "parameters": {"order_id": "string", "reason": "string"},
    },
]
