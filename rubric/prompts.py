"""A small, curated set of unambiguously-English prompts used to probe each model.

Kept deterministic (short, factual/simple) so runs are comparable over time.
Each prompt has a stable `id` used as a key in the report.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Prompt:
    id: str
    text: str


PROMPTS: list[Prompt] = [
    Prompt(
        "factual",
        "Explain in a short paragraph (about 4-5 sentences) why the sky appears "
        "blue during the day and often red or orange at sunset.",
    ),
    Prompt(
        "reasoning",
        "A train travels 60 miles in 1.5 hours, then 90 miles in the next 2 hours. "
        "Walk through the steps to find its average speed for the whole trip, and "
        "state the final answer in miles per hour.",
    ),
    Prompt(
        "summary",
        "In one full paragraph of plain English, explain what a web browser is and "
        "describe the main things it does when you open a web page.",
    ),
    Prompt(
        "instruction",
        "Give three practical tips for someone starting to run for the first time. "
        "Write one or two sentences explaining each tip.",
    ),
    Prompt(
        "creative",
        "Write a short, friendly paragraph (3-4 sentences) welcoming a new employee "
        "to a software team and encouraging them to ask questions.",
    ),
]
