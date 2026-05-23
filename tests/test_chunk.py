import tiktoken

from anchor_mcp.chunk import _count, _get_encoder, _merge, _split, chunk_text
from anchor_mcp.drive import DriveFile


def _file(file_id: str = "f1") -> DriveFile:
    return DriveFile(
        id=file_id,
        name="doc.txt",
        mime_type="text/plain",
        modified_time="2024-01-01T00:00:00.000Z",
        web_view_link="https://drive.google.com/file/d/f1/view",
    )


def _enc() -> tiktoken.Encoding:
    return _get_encoder()


# ── _count ────────────────────────────────────────────────────────────────────


def test_count_empty() -> None:
    assert _count("", _enc()) == 0


def test_count_nonempty() -> None:
    assert _count("hello world", _enc()) > 0


# ── _split ────────────────────────────────────────────────────────────────────


def test_split_short_text_unchanged() -> None:
    enc = _enc()
    text = "Short text."
    pieces = _split(text, ["\n\n", "\n", " ", ""], 800, enc)
    assert pieces == [text]


def test_split_produces_pieces_within_limit() -> None:
    enc = _enc()
    # 200 words per paragraph * 3 paragraphs — will need splitting at chunk_size=50
    para = " ".join(["word"] * 60)
    text = f"{para}\n\n{para}\n\n{para}"
    pieces = _split(text, ["\n\n", "\n", " ", ""], 50, enc)
    for piece in pieces:
        assert _count(piece, enc) <= 50, f"Piece too long: {_count(piece, enc)} tokens"


def test_split_token_boundary_fallback() -> None:
    enc = _enc()
    # A single very long "word" (no spaces) that needs character-level splitting
    long_word = "x" * 2000
    pieces = _split(long_word, ["\n\n", "\n", " ", ""], 100, enc)
    for piece in pieces:
        assert _count(piece, enc) <= 100


# ── _merge ────────────────────────────────────────────────────────────────────


def test_merge_short_pieces_into_one_chunk() -> None:
    enc = _enc()
    pieces = ["Hello", "world", "foo", "bar"]
    chunks = _merge(pieces, chunk_size=800, overlap=100, enc=enc)
    assert len(chunks) == 1


def test_merge_respects_chunk_size() -> None:
    enc = _enc()
    # Each piece ~10 tokens, chunk_size=20 → multiple chunks
    piece = "The quick brown fox jumps over the lazy dog"  # ~9 tokens
    pieces = [piece] * 20
    chunks = _merge(pieces, chunk_size=20, overlap=5, enc=enc)
    for chunk in chunks:
        assert _count(chunk, enc) <= 20


def test_merge_overlap_shares_content() -> None:
    enc = _enc()
    pieces = [f"sentence {i}" for i in range(30)]
    chunks = _merge(pieces, chunk_size=30, overlap=10, enc=enc)
    assert len(chunks) >= 2
    # Adjacent chunks must share some content (overlap)
    words_in_first = set(chunks[0].split())
    words_in_second = set(chunks[1].split())
    assert words_in_first & words_in_second, "Adjacent chunks share no content — overlap is broken"


# ── chunk_text ────────────────────────────────────────────────────────────────


def test_chunk_text_single_chunk_for_short_text() -> None:
    chunks = chunk_text("Hello, world!", _file(), chunk_size=800, overlap=100)
    assert len(chunks) == 1
    assert chunks[0].text == "Hello, world!"
    assert chunks[0].chunk_index == 0
    assert chunks[0].file_id == "f1"
    assert chunks[0].file_name == "doc.txt"
    assert chunks[0].token_count > 0


def test_chunk_text_multiple_chunks_for_long_text() -> None:
    # ~150 tokens per paragraph, 10 paragraphs → needs splitting at 800
    para = ("The quick brown fox jumps over the lazy dog. " * 10).strip()
    text = "\n\n".join([para] * 10)
    chunks = chunk_text(text, _file(), chunk_size=200, overlap=30)
    assert len(chunks) > 1
    for c in chunks:
        assert c.token_count <= 200


def test_chunk_text_ids_are_deterministic() -> None:
    text = "Some content.\n\nMore content here."
    file = _file("f1")
    chunks1 = chunk_text(text, file, chunk_size=800, overlap=100)
    chunks2 = chunk_text(text, file, chunk_size=800, overlap=100)
    assert [c.id for c in chunks1] == [c.id for c in chunks2]


def test_chunk_text_ids_differ_by_file() -> None:
    text = "Same text."
    chunks_a = chunk_text(text, _file("fileA"), chunk_size=800, overlap=100)
    chunks_b = chunk_text(text, _file("fileB"), chunk_size=800, overlap=100)
    assert chunks_a[0].id != chunks_b[0].id


def test_chunk_text_source_url_populated() -> None:
    chunks = chunk_text("content", _file(), chunk_size=800, overlap=100)
    assert chunks[0].source_url == "https://drive.google.com/file/d/f1/view"


def test_chunk_text_source_url_none_when_missing() -> None:
    file = DriveFile(
        id="f2",
        name="doc.txt",
        mime_type="text/plain",
        modified_time="2024-01-01T00:00:00.000Z",
    )
    chunks = chunk_text("content", file, chunk_size=800, overlap=100)
    assert chunks[0].source_url is None


def test_chunk_text_indices_sequential() -> None:
    para = ("word " * 50).strip()
    text = "\n\n".join([para] * 20)
    chunks = chunk_text(text, _file(), chunk_size=100, overlap=20)
    for i, c in enumerate(chunks):
        assert c.chunk_index == i
