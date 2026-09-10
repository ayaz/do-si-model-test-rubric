#!/usr/bin/env python3
"""Entry point: probe every configured model with the built-in prompts and write report.html.

Usage:
    python run.py                 # uses models.yaml + $DIGITALOCEAN_INFERENCE_KEY
    python run.py --config x.yaml --output out.html
"""

from __future__ import annotations

import argparse
import datetime
import sys

from rubric.capabilities import (
    evaluate_structured_output,
    evaluate_tool_call,
    tool_body,
)
from rubric.client import (
    make_client,
    run_prompt,
    run_structured_output,
    run_tool_call,
)
from rubric.config import get_api_key, load_config
from rubric.heuristics import evaluate
from rubric.prompts import PROMPTS
from rubric.report import (
    CapabilityOutcome,
    ModelOutcome,
    PromptOutcome,
    build_report,
    classify,
)


def _load_dotenv() -> None:
    """Minimal .env loader (no external dependency) — only sets keys not already present."""
    from pathlib import Path

    env_file = Path(".env")
    if not env_file.exists():
        return
    import os

    for line in env_file.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip("'\""))


def _run_capability_checks(client, model) -> list[CapabilityOutcome]:
    """Probe tool calling and structured output for one model."""
    tool_res = run_tool_call(client, model)
    tool_status, tool_detail = evaluate_tool_call(tool_res)

    struct_res = run_structured_output(client, model)
    struct_status, struct_detail = evaluate_structured_output(struct_res)

    return [
        CapabilityOutcome(
            name="tool-calling",
            label="Tool calling",
            description="asked to call get_weather(location) for Paris",
            status=tool_status,
            detail=tool_detail,
            body=tool_body(tool_res),
            finish_reason=tool_res.finish_reason,
            temperature=tool_res.temperature,
            is_error=tool_res.is_error and not tool_res.unsupported,
        ),
        CapabilityOutcome(
            name="structured-output",
            label="Structured output (JSON schema)",
            description="asked to extract {name, age, city} as strict-schema JSON",
            status=struct_status,
            detail=struct_detail,
            body=None if struct_res.is_error else (struct_res.text or ""),
            finish_reason=struct_res.finish_reason,
            temperature=struct_res.temperature,
            is_error=struct_res.is_error and not struct_res.unsupported,
        ),
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="models.yaml", help="Path to the models config.")
    parser.add_argument("--output", default="report.html", help="Path to write the HTML report.")
    args = parser.parse_args()

    _load_dotenv()

    try:
        config = load_config(args.config)
        api_key = get_api_key()
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    client = make_client(api_key)
    outcomes: list[ModelOutcome] = []

    for model in config.models:
        print(f"Checking {model} …", flush=True)
        prompt_outcomes: list[PromptOutcome] = []
        for prompt in PROMPTS:
            result = run_prompt(client, model, prompt.text)
            verdict = None if result.is_error else evaluate(result.text or "", config.thresholds)
            status, note = classify(result, verdict)
            prompt_outcomes.append(
                PromptOutcome(
                    prompt_id=prompt.id,
                    prompt_text=prompt.text,
                    result=result,
                    verdict=verdict,
                    status=status,
                    note=note,
                )
            )
        capabilities = _run_capability_checks(client, model)
        outcomes.append(
            ModelOutcome(model=model, prompts=prompt_outcomes, capabilities=capabilities)
        )

    timestamp = datetime.datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")
    path = build_report(outcomes, timestamp=timestamp, api_key=api_key, output_path=args.output)

    counts = {"PASS": 0, "FAIL": 0, "WARN": 0, "ERROR": 0}
    for m in outcomes:
        counts[m.status] += 1
    print(
        f"\nDone: {counts['PASS']} PASS · {counts['FAIL']} FAIL · "
        f"{counts['WARN']} WARN · {counts['ERROR']} ERROR  →  wrote {path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
