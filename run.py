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

from rubric.client import make_client, run_prompt
from rubric.config import get_api_key, load_config
from rubric.heuristics import evaluate
from rubric.prompts import PROMPTS
from rubric.report import ModelOutcome, PromptOutcome, build_report, classify


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
        outcomes.append(ModelOutcome(model=model, prompts=prompt_outcomes))

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
