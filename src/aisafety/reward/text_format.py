"""Canonical text formatting for reward-model inputs."""

from __future__ import annotations


def format_prompt_response(prompt: str | None, response: str, *, sep: str = "\n\n") -> str:
    prompt = "" if prompt is None else str(prompt)
    response = str(response)
    if prompt.strip():
        return prompt.rstrip() + sep + response.lstrip()
    return response

