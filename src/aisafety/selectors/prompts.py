"""Prompt templates for selector models."""

from __future__ import annotations


def system_prompt_for_persona(persona: str) -> str:
    if persona == "human":
        return (
            "You are a human expert judge. "
            "Imagine you are a human decision-maker in this domain and rely on your "
            "human judgement about what is most useful for another human."
        )
    if persona == "ai":
        return (
            "You are an advanced large language model acting as an expert judge. "
            "You carefully follow the instructions and base your decision only on "
            "the specified criteria."
        )
    return "You are a careful, impartial judge that strictly follows instructions."


def make_selector_prompt(row: dict) -> str:
    t = row["item_type"]
    if t == "product":
        task = (
            "You will see two alternative descriptions for the same product. "
            "Choose the SINGLE description that is better for a shopper deciding whether to buy. "
            "Judge on clarity, correctness, informativeness, and usefulness to purchase decision."
        )
    elif t == "movie":
        task = (
            "You will see two alternative synopses for the same movie. "
            "Choose the SINGLE synopsis that better helps a viewer decide whether to watch. "
            "Judge on clarity, informativeness, genre/plot understanding, and relevance to audience."
        )
    elif t == "paper":
        task = (
            "You will see two alternative abstracts for the same scientific paper. "
            "Choose the SINGLE abstract that better helps a researcher decide whether to read. "
            "Judge on clarity, key contributions, methods, and substantive content (not style)."
        )
    elif t == "hc3":
        task = (
            "You will see two alternative answers to the same question. "
            "Choose the SINGLE answer that is better for a human seeking an accurate, helpful response. "
            "Judge on factual correctness, clarity, completeness, and safety."
        )
    else:
        task = "Choose the SINGLE description that is better for the intended decision."

    return f"""

{task}

Title:
{row['title']}

====================
Description A
====================
{row['A_text']}

====================
Description B
====================
{row['B_text']}

Answer format:
Output exactly one character on the FIRST line: A or B
Do not output anything else.
""".strip()
