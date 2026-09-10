"""Evaluate the tool-calling and structured-output capability probes.

These are network-free judges over a ``CapabilityResult`` (duck-typed here so the
tests don't need to import the OpenAI SDK). Each returns a (status, detail) pair
using the same status vocabulary as the rest of the report:

    PASS  — the model performed the capability correctly
    FAIL  — the model supports it but did it wrong (bad/missing tool call, bad JSON)
    WARN  — the endpoint reports the feature is unsupported for this model
    ERROR — the request failed for an unrelated reason (auth, timeout, …)
"""

from __future__ import annotations

import json

from .report import ERROR, FAIL, PASS, WARN

EXPECTED_TOOL = "get_weather"
REQUIRED_JSON_KEYS = ("name", "age", "city")


def evaluate_tool_call(result) -> tuple[str, str]:
    if result.is_error:
        if result.unsupported:
            return WARN, f"Tool calling not supported by this model. ({result.error})"
        return ERROR, result.error

    calls = result.tool_calls or []
    if not calls:
        return FAIL, "Model returned no tool call although one was clearly required."

    call = calls[0]
    if call.get("name") != EXPECTED_TOOL:
        return FAIL, f"Called unexpected function '{call.get('name')}' (expected '{EXPECTED_TOOL}')."

    try:
        args = json.loads(call.get("arguments") or "{}")
    except json.JSONDecodeError as exc:
        return FAIL, f"Tool-call arguments were not valid JSON: {exc}."

    location = args.get("location")
    if not isinstance(location, str) or not location.strip():
        return FAIL, "Tool call is missing the required 'location' string argument."

    extra = f", unit={args['unit']}" if "unit" in args else ""
    return PASS, f"Correctly called {EXPECTED_TOOL}(location={location!r}{extra})."


def evaluate_structured_output(result) -> tuple[str, str]:
    if result.is_error:
        if result.unsupported:
            return WARN, f"Structured output (json_schema) not supported by this model. ({result.error})"
        return ERROR, result.error

    text = (result.text or "").strip()
    if not text:
        return FAIL, "Structured-output response was empty."

    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        return FAIL, f"Response was not valid JSON: {exc}."

    if not isinstance(data, dict):
        return FAIL, f"Expected a JSON object, got {type(data).__name__}."

    missing = [k for k in REQUIRED_JSON_KEYS if k not in data]
    if missing:
        return FAIL, f"JSON is missing required key(s): {', '.join(missing)}."

    if not isinstance(data.get("age"), int) or isinstance(data.get("age"), bool):
        return FAIL, f"'age' should be an integer, got {type(data.get('age')).__name__}."
    if not isinstance(data.get("name"), str) or not isinstance(data.get("city"), str):
        return FAIL, "'name' and 'city' should both be strings."

    return PASS, f"Returned valid schema-conforming JSON: {json.dumps(data)}."


def tool_body(result) -> str | None:
    """Human-readable rendering of the tool-call response for the report <pre>."""
    if result.is_error:
        return None
    calls = result.tool_calls or []
    if not calls:
        return "(no tool calls returned)"
    return "\n".join(f"{c.get('name')}({c.get('arguments')})" for c in calls)
