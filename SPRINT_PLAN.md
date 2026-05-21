# Anchor — Master Sprint Plan

> **Project**: `anchor-mcp` — a local-first MCP server that grounds LLMs in a Google Drive folder, with source-cited retrieval, append-only knowledge capture, and BYOK faithfulness verification.
>
> **Owner**: Atharva. Implementation by Claude Code. Default model: **Sonnet**. Opus only on sprints marked `[OPUS]`.
>
> **Transport**: stdio (local, single-user).
> **Judge LLM**: BYOK via OpenRouter.
> **Vector backend**: ChromaDB local (default), Pinecone optional via env.
> **Embeddings**: bge-m3 local via sentence-transformers (free, multilingual, 8192-token context).
>
> This document is the single source of truth. If a sprint disagrees with this header, the header wins.

---

## Standing rules for every sprint (Claude Code: read this before each sprint)

1. **Do not synthesize or fabricate inputs.** If a required file, env var, OAuth token, or upstream artifact is missing, **stop and ask**. Never generate fake test data or stub a missing dependency to make a test "pass." This rule overrides any instinct to keep momentum.
2. **No scope creep.** If a sprint says "implement X," do not also implement Y, Z, or "while we're here, let's add..." Anything not in the sprint goes into `BACKLOG.md`.
3. **Stay within the locked architectural decisions** (see header). Do not propose alternative transports, chunking strategies, or providers mid-sprint. If you believe the locked decision is wrong, stop and surface it as a question.
4. **Token discipline.** Each sprint has a target token budget. If you're at 80% of budget and not done, stop, write what's done into `SPRINT_NOTES.md`, and ask for direction.
5. **Tests are not optional**, but they are minimal: unit tests for pure functions, one integration test per MCP tool, no test for things that require live OAuth (mock the Drive client).
6. **Type-check with pyright strict.** Lint with ruff. Format with ruff format. Every sprint ends green.
7. **Commit at the end of every sprint** with message `sprint-N: <one-line summary>`. Do not commit mid-sprint unless explicitly told.
8. **Refuse anti-patterns** listed in each sprint. These are explicit "do not do this" items based on common Claude Code drift.

---

## Architecture summary (pre-thinking, applies to all sprints)

```
┌─────────────────────┐
│  Claude Desktop /   │   stdio
│  Claude Code CLI    │◄─────────┐
└─────────────────────┘          │
                                 │
                    ┌────────────▼──────────────┐
                    │   anchor-mcp (FastMCP)    │
                    │  ┌─────────────────────┐  │
                    │  │ MCP Tools:          │  │
                    │  │  - search           │  │
                    │  │  - get_document     │  │
                    │  │  - list_sources     │  │
                    │  │  - add_note         │  │
                    │  │  - verify_claim     │  │
                    │  │  - sync_drive       │  │
                    │  └─────────────────────┘  │
                    └─┬───────────┬─────────┬───┘
                      │           │         │
              ┌───────▼──┐  ┌─────▼─────┐  ┌▼──────────┐
              │ Google   │  │ Vector    │  │ OpenRouter│
              │ Drive    │  │ Backend   │  │ (judge)   │
              │ API      │  │ (Chroma   │  │           │
              │          │  │ /Pinecone)│  │           │
              └──────────┘  └───────────┘  └───────────┘
```

**Local state directory** (`~/.anchor/`):
- `config.json` — folder ID, backend choice, embedding model
- `oauth_token.json` — Drive OAuth refresh token (encrypted at rest)
- `chroma/` — ChromaDB persistent storage (if local backend)
- `cache/sync_state.json` — last sync timestamp, file modified-time map
- `notes/` — sidecar markdown for `add_note` (auditable, human-readable)
- `logs/anchor.log` — rotating log

**Locked dependencies** (pin in `pyproject.toml`):
- `mcp[cli]>=1.27` (FastMCP)
- `google-api-python-client`, `google-auth-oauthlib`, `google-auth-httplib2`
- `chromadb>=0.5`
- `sentence-transformers>=3.0` (for bge-m3)
- `pypdf>=4.0`
- `httpx` (for OpenRouter)
- `pydantic>=2`
- `click` (CLI)
- `cryptography` (token encryption at rest)
- `pinecone-client` (optional, behind extras)

Dev: `pytest`, `pytest-asyncio`, `ruff`, `pyright`.

---

# Sprint 0 — README and project skeleton  [SONNET] [~12k tokens]

**Goal**: produce a complete, CV-quality README that describes the *intended* product, plus the empty Python project skeleton. Atharva can list this on his CV and apply immediately while the rest of the build proceeds asynchronously.

## Pre-thinking checklist
- [ ] Read this entire SPRINT_PLAN.md before starting.
- [ ] Confirm: project name is `anchor-mcp` (PyPI), product name is "Anchor."
- [ ] Confirm: the README describes intended capabilities at v1, not "TODO" or "coming soon."

## Implementation steps
1. Create repo structure:
   ```
   anchor-mcp/
     README.md
     LICENSE                # MIT
     pyproject.toml         # uv-managed
     ruff.toml
     pyrightconfig.json
     .gitignore             # incl. .anchor/, *.json oauth tokens, __pycache__
     .env.example
     src/anchor_mcp/
       __init__.py          # __version__ = "0.1.0"
       __main__.py          # CLI entrypoint
     tests/
       __init__.py
     docs/
       ARCHITECTURE.md      # stub, populated in Sprint 8
       DESIGN.md            # stub with "Future Work: multi-tenant ACL mirroring" section
       BACKLOG.md           # empty
       SPRINT_NOTES.md      # empty
   ```
2. Write `pyproject.toml` with all locked deps under `[project.dependencies]`, Pinecone under `[project.optional-dependencies].pinecone`, dev deps under `dev`.
3. Write `README.md` covering:
   - One-line tagline
   - The problem (3 sentences max — meeting transcripts + notes + docs scattered in Drive, LLMs hallucinate, Glean is enterprise-only)
   - What Anchor does (bullet list of the 6 MCP tools with one-line descriptions each)
   - Quickstart (install via pip, `anchor init`, OAuth flow, config in Claude Desktop)
   - Architecture diagram (use the ASCII diagram above)
   - Configuration (env vars: `OPENROUTER_API_KEY`, `ANCHOR_VECTOR_BACKEND`, `ANCHOR_DRIVE_FOLDER_ID`, optional `PINECONE_API_KEY`)
   - Supported file types (PDF text-copyable, .txt, .md, Google Docs, Google Meet transcripts; explicit "not supported in v1" list)
   - Demo section (link placeholder for Loom — leave as `[Loom link: TBA]`)
   - Limitations and honest scope (single-user, append-only vector store, multi-tenant ACL is future work, OCR is future work)
   - Future work (link to `docs/DESIGN.md`)
   - License
4. Write `LICENSE` (MIT, Atharva as copyright holder).
5. Write `.env.example` with every required and optional env var, commented.
6. Initialize git repo, first commit `sprint-0: project skeleton and README`.

## Anti-patterns (refuse)
- Do NOT write any actual Python implementation in this sprint. The `__main__.py` is a stub that prints "Anchor v0.1.0 — not yet implemented." That's it.
- Do NOT add CI/CD, Docker, Cloud Run configs, or GitHub Actions in this sprint.
- Do NOT add a logo, badges, or marketing fluff to the README. Plain, technical, defensible.
- Do NOT promise features not in the locked scope (no "multi-user support," no "OCR," no "ChatGPT compatibility").

## Acceptance criteria
- `pip install -e .` succeeds in a fresh venv.
- `python -m anchor_mcp` prints the stub message and exits cleanly.
- `ruff check` and `pyright` both pass on the empty skeleton.
- README renders cleanly on GitHub (Atharva eyeballs it).
- Atharva can copy-paste a CV bullet from the README without embarrassment.

## CV bullet generated by this sprint
> "Designed and shipped `anchor-mcp`, a local-first Model Context Protocol server that grounds LLMs (Claude, Cursor, any MCP-compatible client) in a user's Google Drive corpus. Source-cited retrieval, BYOK faithfulness verification, multilingual embeddings via bge-m3. Python 3.11, FastMCP, ChromaDB / Pinecone."

---

# Sprint 1 — Configuration, CLI, and state directory  [SONNET] [~15k tokens]

**Goal**: a working `anchor init` and `anchor config` CLI that bootstraps `~/.anchor/`, validates env vars, and stores a typed config the rest of the codebase reads from.

## Pre-thinking checklist
- [ ] All config is pydantic-validated. No raw dict access elsewhere in the codebase.
- [ ] `~/.anchor/` must respect `XDG_CONFIG_HOME` if set (Linux convention).
- [ ] Decide config file format: JSON (for human-readability) — not TOML, not YAML.

## Implementation steps
1. Create `src/anchor_mcp/config.py`:
   - `AnchorConfig` pydantic model with fields: `drive_folder_id`, `vector_backend` (Literal["chroma", "pinecone"]), `embedding_model` (default "BAAI/bge-m3"), `judge_model` (default "anthropic/claude-haiku-4-5"), `chunk_size` (default 800), `chunk_overlap` (default 100), `state_dir` (Path).
   - `load_config()` reads `~/.anchor/config.json`, returns `AnchorConfig`. Raises `ConfigNotFoundError` with a helpful message if missing.
   - `save_config(cfg)` writes atomically (write to tmp, rename).
2. Create `src/anchor_mcp/cli.py` using Click:
   - `anchor init` — interactive prompt for Drive folder ID, asks for backend choice (default chroma), validates `OPENROUTER_API_KEY` env var is set, creates `~/.anchor/` tree, writes initial config.
   - `anchor config show` — prints current config (redacts secrets).
   - `anchor config set <key> <value>` — updates one field, re-validates.
   - `anchor doctor` — runs sanity checks: state dir exists, config valid, `OPENROUTER_API_KEY` set, OAuth token present (just check file exists; don't validate it yet — that's Sprint 2).
3. Wire `cli.py` into `__main__.py` so `python -m anchor_mcp` and `anchor` (via entrypoint in pyproject) both work.
4. Add `anchor_mcp.errors` module with custom exceptions: `ConfigNotFoundError`, `AuthError`, `SyncError`, `BackendError`.
5. Unit tests for config load/save round-trip, atomic write behavior, CLI smoke tests via Click's `CliRunner`.

## Anti-patterns (refuse)
- No global mutable state. Config is loaded per-call or injected.
- No `os.environ` reads outside `config.py` and a single `secrets.py` module that wraps env access.
- No "fallback to defaults if config is malformed" — fail loudly with a clear error.

## Acceptance criteria
- `anchor init` in a fresh home directory produces a valid `~/.anchor/config.json`.
- `anchor doctor` reports all green when env is set up correctly, red with specific remediation when not.
- Tests pass, ruff + pyright clean.

---

# Sprint 2 — Google Drive OAuth and file enumeration  [SONNET] [~20k tokens]

**Goal**: OAuth flow that gets a refresh token, stores it encrypted, and a `DriveClient` that lists + downloads files from the configured folder (recursively).

## Pre-thinking checklist
- [ ] Atharva has a GCP project and OAuth credentials JSON from Food Explorer experience — assume he can produce one. README documents how.
- [ ] Token encryption at rest: use `cryptography.fernet` with a key derived from a machine-local secret (e.g., a generated key stored at `~/.anchor/.key` with 0600 perms). Not Fort-Knox, but better than plaintext.
- [ ] OAuth scope: `https://www.googleapis.com/auth/drive.readonly`. Read-only. We never write to Drive.

## Implementation steps
1. Create `src/anchor_mcp/auth.py`:
   - `run_oauth_flow(credentials_json_path)` — runs `InstalledAppFlow` with local redirect (port 0, auto-picked), saves token encrypted.
   - `load_credentials()` — loads encrypted token, refreshes if expired, returns `google.oauth2.credentials.Credentials`.
   - `is_authenticated()` — boolean check.
2. Add `anchor auth login --credentials <path>` CLI command that calls `run_oauth_flow`.
3. Add `anchor auth status` that calls `is_authenticated()` and reports.
4. Create `src/anchor_mcp/drive.py`:
   - `DriveClient` class wrapping `googleapiclient.discovery.build("drive", "v3", credentials=...)`.
   - `list_files(folder_id: str, recursive: bool = True) -> list[DriveFile]` — paginated, returns `DriveFile` pydantic model with `id, name, mime_type, modified_time, parents, web_view_link, md5_checksum`.
   - `download_file(file_id, mime_type) -> bytes` — handles Google Docs export (export as `text/plain` for v1; we don't preserve formatting in v1), PDFs and other binaries via `get_media`.
   - Rate-limit handling: retry with exponential backoff on 429/5xx using `tenacity` or hand-rolled. 5 retries max.
5. Mock-based unit tests for `DriveClient` using `unittest.mock`. No live API calls in tests.

## Anti-patterns (refuse)
- Do NOT use a service account. Installed-app OAuth only for v1.
- Do NOT request broader scopes "in case we need them later." `drive.readonly` only.
- Do NOT store the OAuth client secret in code. It's loaded from the user's `credentials.json` path.
- Do NOT swallow Google API errors silently. Wrap into `AuthError` or `SyncError` with the original as `__cause__`.

## Acceptance criteria
- `anchor auth login --credentials creds.json` opens browser, completes OAuth, writes encrypted token.
- `anchor auth status` reports authenticated.
- Manual smoke test: in a Python REPL, `DriveClient.list_files(<atharva_test_folder>)` returns the expected files. (Document this manual check in `SPRINT_NOTES.md`.)

---

# Sprint 3 — Text extraction and chunking  [SONNET] [~15k tokens]

**Goal**: a pure-function pipeline that takes a `DriveFile` + raw bytes and produces a list of `Chunk` objects ready for embedding.

## Pre-thinking checklist
- [ ] This is mostly boring deterministic work. Pure functions, easy to test.
- [ ] Chunk size 800 tokens, overlap 100. Use `tiktoken` for token counting (cl100k_base is fine as a proxy).
- [ ] One file → many chunks. Each chunk gets full metadata for citation.

## Implementation steps
1. Create `src/anchor_mcp/extract.py`:
   - `extract_text(file: DriveFile, raw_bytes: bytes) -> str` — dispatches on `mime_type`:
     - `application/pdf` → pypdf, raise `ExtractError` if no text (scanned PDF detection)
     - `text/plain`, `text/markdown` → decode utf-8
     - `application/vnd.google-apps.document` → already text-plain after export in Sprint 2
     - Anything else → raise `UnsupportedMimeTypeError`
2. Create `src/anchor_mcp/chunk.py`:
   - `Chunk` pydantic model: `id` (deterministic hash of file_id + chunk_index + text), `text`, `file_id`, `file_name`, `chunk_index`, `token_count`, `modified_time`, `source_url`.
   - `chunk_text(text: str, file: DriveFile, chunk_size: int, overlap: int) -> list[Chunk]` — recursive character splitter on `\n\n`, `\n`, `. `, ` `, then character-level if needed. Token-counted, not char-counted.
3. Tests:
   - Round-trip: extract text from a fixture PDF, chunk it, verify chunk count + overlap math.
   - `Chunk.id` is deterministic across runs.
   - Unsupported MIME raises the right error.

## Anti-patterns (refuse)
- No "semantic chunking" via embeddings or LLM. Recursive character splitter only.
- No OCR. Scanned PDF = `ExtractError`. Documented limitation.
- No silent encoding fallbacks. UTF-8 only; on failure, raise.

## Acceptance criteria
- All file types in the supported list extract + chunk cleanly on real samples.
- 100% type coverage for `chunk.py` and `extract.py`.

---

# Sprint 4 — Vector backend abstraction + embedding  [SONNET] [~20k tokens]

**Goal**: a clean abstraction over ChromaDB (default) and Pinecone (optional), plus the embedding pipeline using bge-m3.

## Pre-thinking checklist
- [ ] bge-m3 via `sentence-transformers` — first run will download ~2GB to HuggingFace cache. Document this in README and surface a progress bar via the CLI.
- [ ] Embedding dimension is 1024 for bge-m3. Hard-code as a constant; document.
- [ ] The backend interface is small: `upsert(chunks, embeddings)`, `query(embedding, top_k, filter)`, `delete(chunk_ids)`, `list_sources()`, `count()`.

## Implementation steps
1. Create `src/anchor_mcp/embed.py`:
   - `Embedder` class with lazy-loaded `SentenceTransformer("BAAI/bge-m3")`.
   - `embed_chunks(chunks: list[Chunk]) -> list[list[float]]` — batched (batch_size=32), returns dense vectors. Normalize for cosine sim.
   - `embed_query(text: str) -> list[float]` — single query.
2. Create `src/anchor_mcp/backends/base.py`:
   - `VectorBackend` Protocol with the 5 methods above. Use pydantic models for inputs/outputs, no dicts.
3. Create `src/anchor_mcp/backends/chroma_backend.py`:
   - `ChromaBackend` using `chromadb.PersistentClient(path=state_dir / "chroma")`.
   - Single collection `"anchor"`.
   - Stores: embedding, document (chunk.text), metadata (everything else).
4. Create `src/anchor_mcp/backends/pinecone_backend.py`:
   - `PineconeBackend` — only imported if `pinecone-client` is installed.
   - Reads `PINECONE_API_KEY`, `PINECONE_INDEX_NAME` from env.
   - Asserts index dimension matches embedder dimension on first use; raises clear error if not.
5. Create `src/anchor_mcp/backends/__init__.py` with `get_backend(config) -> VectorBackend` factory.
6. Tests:
   - In-memory Chroma round-trip: upsert 3 chunks, query, verify top-1 match.
   - Pinecone test is skipped unless `PINECONE_API_KEY` is set (`pytest.mark.skipif`).

## Anti-patterns (refuse)
- No support for "other" backends in v1. Two is enough.
- No re-ranking layer. v1 is single-stage dense retrieval.
- No background re-indexing. Embeddings are computed on `sync_drive` only.

## Acceptance criteria
- A fresh ChromaDB collection can be created, populated with 10+ chunks, queried, and returns sensible nearest neighbors.
- Switching `ANCHOR_VECTOR_BACKEND=pinecone` (with key set) routes through Pinecone with no other code changes.

---

# Sprint 5 — Sync engine  [SONNET] [~20k tokens]

**Goal**: the `sync_drive` flow that detects new/modified/deleted files, pipes them through extract → chunk → embed → upsert, and updates a local sync state.

## Pre-thinking checklist
- [ ] Sync state at `~/.anchor/cache/sync_state.json`: `{file_id: {modified_time, md5_checksum, chunk_ids: [...]}}`.
- [ ] **Append-only at the vector level, but we must handle file changes.** Decision: when a file's `modified_time` changes, delete its old chunks (using stored `chunk_ids`) and insert new ones. This isn't "editing a vector," it's "re-indexing a source document." That's a defensible distinction — document it in DESIGN.md.
- [ ] **Deletions:** if a file is no longer in Drive, its chunks are deleted from the vector store. (Otherwise stale knowledge accumulates forever.) Document.
- [ ] User-initiated `add_note` content (Sprint 7) is *not* touched by sync — it's a separate logical namespace.

## Implementation steps
1. Create `src/anchor_mcp/sync.py`:
   - `SyncState` pydantic model + JSON persistence helpers.
   - `Syncer` class with `__init__(drive, embedder, backend, state)`.
   - `sync(folder_id: str, progress_cb: Callable | None = None) -> SyncReport` where `SyncReport = {added, updated, deleted, skipped, errors}`.
   - Diff algorithm:
     1. List files in folder via `DriveClient`.
     2. For each: if `file_id not in state` → ADD. If `state[file_id].modified_time < drive_modified` → UPDATE. Else SKIP.
     3. For each `file_id in state` but not in Drive → DELETE.
   - Errors per-file don't abort the sync; collected in `SyncReport.errors`.
2. Add `anchor sync` CLI command that runs the sync and prints the report.
3. Tests with mocked Drive + Chroma:
   - First sync (cold): all files added.
   - Second sync (no changes): all skipped.
   - File modified: 1 update, old chunks gone, new chunks present.
   - File deleted from Drive: chunks removed from backend.

## Anti-patterns (refuse)
- No partial-sync resume on failure mid-sprint. If sync crashes, next run starts over for un-finished files (since state is only updated post-success per file). Document this.
- No webhooks, no polling daemon. Manual `anchor sync` only.
- No "smart" diff that tries to chunk-level diff documents. Whole-file re-index on change. Simpler and correct.

## Acceptance criteria
- Full cold sync of a 20-file fixture folder completes without errors and reports correct counts.
- Modifying one file and re-syncing produces exactly 1 update, with old chunks gone.

---

# Sprint 6 — MCP server: read tools  [OPUS] [~20k tokens]

**Goal**: the FastMCP server exposing `search`, `get_document`, `list_sources`, and `sync_drive` as MCP tools. **Marked Opus** because the tool schemas, descriptions, and citation contract are the user-facing API — these need to be tight, clear, and resistant to misuse by an LLM caller.

## Pre-thinking checklist
- [ ] Tool descriptions are the *prompt* the LLM sees to decide when to call. They must be specific, with examples. This is the highest-leverage prose in the codebase.
- [ ] Every search result includes structured citation metadata. The LLM will be instructed (via tool descriptions) to cite these explicitly.
- [ ] No `stdout` logging in stdio mode — it corrupts the MCP transport. All logs to file or stderr.

## Implementation steps
1. Create `src/anchor_mcp/server.py`:
   - Initialize FastMCP server `mcp = FastMCP("anchor")`.
   - Tool: `search(query: str, top_k: int = 5, file_name_filter: str | None = None) -> list[SearchResult]`. `SearchResult` includes `text, file_name, file_id, chunk_index, source_url, relevance_score, modified_time`.
   - Tool: `get_document(file_id: str) -> DocumentView` — returns full text of a file (re-assembled from chunks), with metadata.
   - Tool: `list_sources(name_filter: str | None = None) -> list[SourceSummary]` — lists all indexed files with chunk counts.
   - Tool: `sync_drive() -> SyncReport` — triggers a sync, returns the report.
2. **Tool descriptions** — written carefully, with explicit instruction to the LLM:
   - `search`: "Search the user's Google Drive knowledge base by semantic similarity. Returns up to `top_k` chunks of text with full source metadata. **You MUST cite results using the format `[file_name, chunk N](source_url)` when using information from them.** If results have low relevance scores (< 0.5), state that the answer may not be in the user's documents."
   - `get_document`: "Retrieve the full reconstructed text of a single document by its `file_id`. Use this when `search` returned a useful snippet and you need broader context from the same file."
   - `list_sources`: "Enumerate documents available in the user's indexed Drive folder. Useful when the user asks 'what do you have access to?' or wants to know if a specific document is indexed."
   - `sync_drive`: "Re-sync the user's Drive folder, picking up new, modified, and deleted files. **Only call this when the user explicitly asks to refresh / sync / update / re-index.** Returns a summary of changes."
3. Configure logging to write to `~/.anchor/logs/anchor.log` with rotation. Stderr for warnings only. Stdout is reserved for MCP transport.
4. Add `anchor serve` CLI command that starts the FastMCP stdio server.
5. Write a `claude_desktop_config.json` example block in README:
   ```json
   {
     "mcpServers": {
       "anchor": {
         "command": "anchor",
         "args": ["serve"],
         "env": { "OPENROUTER_API_KEY": "sk-or-..." }
       }
     }
   }
   ```
6. Tests:
   - `mcp dev` (the FastMCP dev runner) loads the server without error.
   - Each tool callable with valid inputs returns the expected pydantic-serialized response.
   - `search` against a fixture-populated Chroma returns ranked results.

## Anti-patterns (refuse)
- Do NOT log to stdout. This breaks MCP stdio transport. All logs go to file or stderr.
- Do NOT skip the explicit citation instruction in tool descriptions — that's the entire faithfulness story for read tools.
- Do NOT add tools beyond the four listed. `add_note` and `verify_claim` are Sprint 7.

## Acceptance criteria
- Server starts cleanly via `anchor serve` and via Claude Desktop launch.
- All four tools are visible and callable via MCP Inspector.
- Atharva runs end-to-end: opens Claude Desktop, asks "what's in my ISB Strategy folder?" and gets a cited answer.

---

# Sprint 7 — MCP server: write tools (`add_note`, `verify_claim`)  [OPUS] [~18k tokens]

**Goal**: append-only knowledge capture and BYOK faithfulness verification. **Marked Opus** because `verify_claim`'s judge prompt is the second-highest-leverage prompt in the codebase — get it wrong and the tool produces confident-sounding wrong classifications.

## Pre-thinking checklist
- [ ] `add_note` writes a markdown file to `~/.anchor/notes/` AND ingests it into the vector store, namespaced with `source_type: "user_note"` so it's distinguishable from Drive content.
- [ ] `verify_claim` calls OpenRouter directly via httpx. No SDK dependency. Single POST, parse JSON response.
- [ ] If `OPENROUTER_API_KEY` is not set, `verify_claim` is **not registered** as an MCP tool. Server still starts; the tool simply doesn't exist for the LLM. Document this clearly.
- [ ] The judge prompt must be carefully written: it sees `(claim, evidence_chunks)` and must output structured JSON `{verdict: "supported"|"partially_supported"|"not_supported", rationale: str, evaluated_chunk_ids: [...]}`. Few-shot the verdict criteria.

## Implementation steps
1. Add `add_note` tool to `server.py`:
   - Signature: `add_note(content: str, title: str, tags: list[str] = []) -> NoteReceipt`.
   - Writes `~/.anchor/notes/{timestamp}_{slug(title)}.md` with frontmatter (title, tags, created_at).
   - Chunks + embeds + upserts to backend with `source_type: "user_note"`.
   - Tool description includes: "**This appends a permanent note to the user's knowledge base. Only call this when the user explicitly says 'remember this,' 'save this,' 'add a note,' or similar.** The note becomes searchable immediately."
2. Create `src/anchor_mcp/judge.py`:
   - `OpenRouterJudge` class.
   - Prompt template stored as a constant (single source of truth, no string templating across files):
     ```
     You are a strict faithfulness judge. Given a CLAIM and a set of EVIDENCE chunks from a knowledge base, determine whether the claim is supported.

     Return verdict from {supported, partially_supported, not_supported}:
     - supported: every factual assertion in the claim is directly stated in the evidence
     - partially_supported: some assertions are in the evidence, others are not present (or contradicted)
     - not_supported: the claim's key assertions are not present in the evidence (or are contradicted)

     Output strict JSON only:
     {"verdict": "...", "rationale": "<1-2 sentences>", "evaluated_chunk_ids": [...]}

     CLAIM: {claim}

     EVIDENCE:
     {evidence_block}
     ```
   - `verify(claim: str, chunk_ids: list[str]) -> VerifyResult`. Loads chunk texts from backend by ID, builds evidence block, posts to OpenRouter, parses strict JSON (use json.loads with strict mode; reject if not valid JSON; retry once with a "Your last response was not valid JSON, try again" follow-up).
3. Add `verify_claim` tool to `server.py` — registered conditionally on `OPENROUTER_API_KEY` presence. Tool description:
   > "Verify whether a factual claim is supported by specific evidence chunks from the user's knowledge base. Use this after making a substantive claim based on `search` results. Pass the claim text and the `chunk_id` values from the search results you used. Returns a verdict: supported / partially_supported / not_supported, with rationale. If verdict is not 'supported', you MUST tell the user your claim could not be fully verified against their documents."
4. Tests:
   - `add_note` round-trip: call tool, verify file written, verify chunk searchable.
   - `judge.verify` with mocked OpenRouter response for each verdict type.
   - Judge handles malformed JSON gracefully (retries once, then errors).

## Anti-patterns (refuse)
- Do NOT call OpenRouter from any other module. All judge logic in `judge.py`.
- Do NOT expand `verify_claim` to do retrieval itself ("verify this claim against my whole drive"). It only operates on chunks the LLM already retrieved. Keeps the contract small.
- Do NOT add a "score" or "confidence percentage" to the judge output. Three-state verdict + rationale only. Probabilistic scores from LLM judges are noise; don't pretend otherwise.

## Acceptance criteria
- `add_note` called with a fixture note appears in `list_sources` after the call, and surfaces in `search`.
- `verify_claim` with a supported claim returns `supported`; with a fabricated claim returns `not_supported`. Atharva manually runs both cases through Claude Desktop and documents in `SPRINT_NOTES.md`.

---

# Sprint 8 — Hardening, docs, demo  [SONNET] [~18k tokens]

**Goal**: production-quality polish, complete docs, and a recorded Loom demo. This is the sprint that converts a working tool into a CV-worthy artifact.

## Pre-thinking checklist
- [ ] The README written in Sprint 0 may be out of sync with shipped reality. Reconcile.
- [ ] DESIGN.md gets the real version with all the decision rationale.
- [ ] One full end-to-end demo run on the ISB coursework + fabricated Meet transcripts.

## Implementation steps
1. Read Sprint 0's README. Diff against current capabilities. Update anything that drifted. Add Loom link.
2. Write `docs/ARCHITECTURE.md`:
   - The diagram from this plan.
   - Module-by-module: what lives where and why.
   - Data flow: ingestion path, query path, verification path.
3. Write `docs/DESIGN.md`:
   - **Decision log**: stdio vs HTTP, append-only with re-index on change, single-user, no OCR, no semantic chunking, openrouter for judge. Each with one paragraph of rationale.
   - **Future work**: multi-tenant ACL mirroring (with the security argument from our conversation), Streamable HTTP transport for remote/Claude-web use, OCR via Tesseract, polling-based incremental sync, re-ranking layer, observability hooks.
4. Add a `Makefile` (or `justfile`) with targets: `install`, `lint`, `typecheck`, `test`, `serve`, `sync`, `doctor`.
5. Add GitHub Actions: `lint + test` on push to main.
6. Add a `CHANGELOG.md`, populate v0.1.0.
7. Atharva records Loom (5-7 min): config in Claude Desktop, `anchor sync`, query against ISB content, query against Meet transcript, `verify_claim` flow, `add_note` flow.
8. Add Loom link to README, commit, tag `v0.1.0`.

## Anti-patterns (refuse)
- Do NOT add new features in this sprint. Polish only.
- Do NOT publish to PyPI yet (separate decision, after Atharva eyeballs the final product).

## Acceptance criteria
- Fresh-clone, fresh-install on a new machine produces a working setup in <10 minutes following only the README.
- All docs render cleanly.
- Loom is shippable to a recruiter as-is.

---

# Sprint 9 (optional / stretch) — Pinecone deploy + observability  [SONNET] [~12k tokens]

**Goal**: optional polish. Only execute if all prior sprints came in under budget.

- Verify Pinecone backend on Atharva's free-tier index end-to-end.
- Add basic OpenTelemetry hooks (just spans for `search`, `sync`, `verify_claim`) — local-only exporter for now. Keyword on the CV: observability.
- Write a blog post draft: "Building a local-first Drive RAG MCP server in 2 weeks."

---

# Token budget summary

| Sprint | Model | Tokens | Cumulative |
|---|---|---|---|
| 0 — README + skeleton | Sonnet | 12k | 12k |
| 1 — Config + CLI | Sonnet | 15k | 27k |
| 2 — OAuth + Drive | Sonnet | 20k | 47k |
| 3 — Extract + chunk | Sonnet | 15k | 62k |
| 4 — Vector backend + embed | Sonnet | 20k | 82k |
| 5 — Sync engine | Sonnet | 20k | 102k |
| 6 — MCP read tools | **Opus** | 20k | 122k |
| 7 — `add_note` + `verify_claim` | **Opus** | 18k | 140k |
| 8 — Hardening + docs + demo | Sonnet | 18k | 158k |
| 9 — Stretch | Sonnet | 12k | 170k |

Two Opus sprints, seven Sonnet. Total target ~158k tokens (170k with stretch). Calendar time ~2 weeks at a comfortable pace.

---

# Standing instructions for Atharva between sprints

- After each sprint, **run the project yourself for 5 minutes** before starting the next. Catch drift early.
- Maintain `BACKLOG.md` with anything Claude Code surfaces as "future work" — don't let it sneak into the current sprint.
- If a sprint exceeds 1.3x budget, **stop and reassess**. Either the sprint was mis-scoped or the locked architecture is wrong. Either is worth a pause.
- Keep `SPRINT_NOTES.md` as a running diary. Useful for blog post in Sprint 9 and for interview storytelling later.
