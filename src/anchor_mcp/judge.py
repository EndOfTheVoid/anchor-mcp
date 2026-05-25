import json
import re
from typing import Any, Literal

import httpx
from pydantic import BaseModel, ValidationError

from anchor_mcp.backends.base import VectorBackend
from anchor_mcp.errors import VerificationError

_OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
_TIMEOUT = 60.0

# Single source of truth for the judge's instructions. Kept verbatim as a module
# constant so it can be reviewed and tuned in one place.
JUDGE_SYSTEM_PROMPT = """You are a strict faithfulness judge evaluating whether a CLAIM is supported
by retrieved EVIDENCE from a knowledge base.

Verdicts:
- supported: every factual assertion in the claim is directly stated in the evidence.
- partially_supported: some assertions are in the evidence; others are absent or contradicted.
- not_supported: the claim's key assertions are absent from or contradicted by the evidence.

Output ONLY valid JSON, no explanation outside the JSON:
{"verdict": "supported|partially_supported|not_supported",
 "rationale": "<1-2 sentences>",
 "evaluated_chunk_ids": ["id1", "id2"]}"""

_RETRY_PROMPT = "Your last response was not valid JSON. Return only the JSON object."


class VerifyResult(BaseModel):
    verdict: Literal["supported", "partially_supported", "not_supported"]
    rationale: str
    evaluated_chunk_ids: list[str]


def _build_user_prompt(claim: str, evidence_block: str) -> str:
    return f"CLAIM: {claim}\n\nEVIDENCE:\n{evidence_block}"


def _parse_verdict(raw: str) -> VerifyResult | None:
    """Parse the judge's response into a VerifyResult, or None if it isn't valid."""
    text = raw.strip()
    # Strip a fenced code block (```json ... ```), which models often add.
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\n?", "", text)
        text = re.sub(r"\n?```$", "", text).strip()

    candidates = [text]
    # Fall back to the first {...} block if there is surrounding prose.
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match is not None and match.group(0) != text:
        candidates.append(match.group(0))

    for candidate in candidates:
        try:
            obj: object = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        try:
            return VerifyResult.model_validate(obj)
        except ValidationError:
            continue
    return None


class OpenRouterJudge:
    def __init__(self, api_key: str, model: str) -> None:
        self._api_key = api_key
        self._model = model

    def verify(self, claim: str, chunk_ids: list[str], backend: VectorBackend) -> VerifyResult:
        chunks = backend.get_chunks_by_ids(chunk_ids)
        if not chunks:
            raise VerificationError(
                "None of the provided chunk_ids were found in the knowledge base."
            )

        evidence_block = "\n\n".join(f"[{c.id}]\n{c.text}" for c in chunks)
        messages: list[dict[str, str]] = [
            {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
            {"role": "user", "content": _build_user_prompt(claim, evidence_block)},
        ]

        raw = self._call(messages)
        result = _parse_verdict(raw)
        if result is not None:
            return result

        # Retry once with an explicit correction.
        messages.append({"role": "assistant", "content": raw})
        messages.append({"role": "user", "content": _RETRY_PROMPT})
        raw = self._call(messages)
        result = _parse_verdict(raw)
        if result is None:
            raise VerificationError(
                "The faithfulness judge did not return valid JSON after a retry."
            )
        return result

    def _call(self, messages: list[dict[str, str]]) -> str:
        try:
            response = httpx.post(
                _OPENROUTER_URL,
                headers={
                    "Authorization": f"Bearer {self._api_key}",
                    "Content-Type": "application/json",
                },
                json={"model": self._model, "messages": messages, "temperature": 0},
                timeout=_TIMEOUT,
            )
        except httpx.HTTPError as exc:
            raise VerificationError(f"OpenRouter request failed: {exc}") from exc

        if response.status_code != 200:
            raise VerificationError(
                f"OpenRouter returned status {response.status_code}: {response.text}"
            )

        try:
            data: Any = response.json()
            return str(data["choices"][0]["message"]["content"])
        except (json.JSONDecodeError, KeyError, IndexError, TypeError) as exc:
            raise VerificationError(f"Unexpected OpenRouter response shape: {exc}") from exc
