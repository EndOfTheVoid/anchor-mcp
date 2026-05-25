from unittest.mock import MagicMock

import pytest

from anchor_mcp.chunk import Chunk
from anchor_mcp.errors import VerificationError
from anchor_mcp.judge import OpenRouterJudge, _parse_verdict

# ── helpers ───────────────────────────────────────────────────────────────────


def _chunk(cid: str = "c1", text: str = "The sky is blue.") -> Chunk:
    return Chunk(
        id=cid,
        text=text,
        file_id="f1",
        file_name="doc.txt",
        chunk_index=0,
        token_count=4,
        modified_time="2024-01-01T00:00:00Z",
        source_url=None,
    )


def _backend(chunks: list[Chunk]) -> MagicMock:
    backend = MagicMock()
    backend.get_chunks_by_ids.return_value = chunks
    return backend


def _resp(content: str, status: int = 200) -> MagicMock:
    resp = MagicMock()
    resp.status_code = status
    resp.text = content
    resp.json.return_value = {"choices": [{"message": {"content": content}}]}
    return resp


def _verdict_json(verdict: str, ids: list[str] | None = None) -> str:
    chunk_ids = ids if ids is not None else ["c1"]
    id_list = ", ".join(f'"{i}"' for i in chunk_ids)
    return f'{{"verdict": "{verdict}", "rationale": "because", "evaluated_chunk_ids": [{id_list}]}}'


# ── verdicts ──────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("verdict", ["supported", "partially_supported", "not_supported"])
def test_verify_returns_each_verdict(monkeypatch: pytest.MonkeyPatch, verdict: str) -> None:
    monkeypatch.setattr(
        "anchor_mcp.judge.httpx.post", lambda *a, **k: _resp(_verdict_json(verdict))
    )
    judge = OpenRouterJudge("key", "model")
    result = judge.verify("The sky is blue.", ["c1"], _backend([_chunk()]))
    assert result.verdict == verdict
    assert result.evaluated_chunk_ids == ["c1"]


def test_verify_builds_evidence_from_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    def fake_post(*_args: object, **kwargs: object) -> MagicMock:
        captured["json"] = kwargs.get("json")
        return _resp(_verdict_json("supported"))

    monkeypatch.setattr("anchor_mcp.judge.httpx.post", fake_post)
    judge = OpenRouterJudge("key", "model")
    judge.verify("claim", ["c1"], _backend([_chunk(text="Evidence text here.")]))

    payload = captured["json"]
    assert isinstance(payload, dict)
    messages = payload["messages"]
    assert messages[0]["role"] == "system"
    assert "Evidence text here." in messages[1]["content"]
    assert payload["model"] == "model"


# ── retry / failure paths ───────────────────────────────────────────────────


def test_verify_retries_once_on_malformed_json(monkeypatch: pytest.MonkeyPatch) -> None:
    responses = iter([_resp("not json at all"), _resp(_verdict_json("not_supported"))])
    calls: list[object] = []

    def fake_post(*_args: object, **kwargs: object) -> MagicMock:
        calls.append(kwargs.get("json"))
        return next(responses)

    monkeypatch.setattr("anchor_mcp.judge.httpx.post", fake_post)
    judge = OpenRouterJudge("key", "model")
    result = judge.verify("claim", ["c1"], _backend([_chunk()]))

    assert result.verdict == "not_supported"
    assert len(calls) == 2  # retried exactly once


def test_verify_raises_after_two_malformed_responses(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("anchor_mcp.judge.httpx.post", lambda *a, **k: _resp("still garbage"))
    judge = OpenRouterJudge("key", "model")
    with pytest.raises(VerificationError, match="valid JSON"):
        judge.verify("claim", ["c1"], _backend([_chunk()]))


def test_verify_raises_when_no_chunks_found(monkeypatch: pytest.MonkeyPatch) -> None:
    called = MagicMock()
    monkeypatch.setattr("anchor_mcp.judge.httpx.post", called)
    judge = OpenRouterJudge("key", "model")
    with pytest.raises(VerificationError, match="chunk_ids"):
        judge.verify("claim", ["missing"], _backend([]))
    called.assert_not_called()  # never hits the network


def test_verify_raises_on_http_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("anchor_mcp.judge.httpx.post", lambda *a, **k: _resp("boom", status=500))
    judge = OpenRouterJudge("key", "model")
    with pytest.raises(VerificationError, match="status 500"):
        judge.verify("claim", ["c1"], _backend([_chunk()]))


# ── _parse_verdict ────────────────────────────────────────────────────────────


def test_parse_verdict_strips_code_fence() -> None:
    raw = f"```json\n{_verdict_json('supported')}\n```"
    result = _parse_verdict(raw)
    assert result is not None
    assert result.verdict == "supported"


def test_parse_verdict_extracts_embedded_json() -> None:
    raw = f"Here is the verdict: {_verdict_json('not_supported')} -- done."
    result = _parse_verdict(raw)
    assert result is not None
    assert result.verdict == "not_supported"


def test_parse_verdict_invalid_returns_none() -> None:
    assert _parse_verdict("totally not json") is None


def test_parse_verdict_rejects_unknown_verdict() -> None:
    raw = '{"verdict": "maybe", "rationale": "x", "evaluated_chunk_ids": []}'
    assert _parse_verdict(raw) is None
