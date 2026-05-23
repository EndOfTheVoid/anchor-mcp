# Anchor — Master Sprint Plan

> **Project**: `anchor-mcp` — an MCP server that grounds LLMs in a Google Drive corpus, with source-cited hybrid retrieval, server-side faithfulness verification, and a cloud-native multi-user deployment.
>
> **Owner**: Atharva. Implementation by Claude Code. Default model: **Sonnet**. Opus only on sprints marked `[OPUS]`.
>
> **Transport**: StreamableHTTP (Cloud Run). Local stdio mode retained for dev only.
> **Judge LLM**: OpenRouter (model-agnostic BYOK).
> **Vector backend**: Pinecone Serverless only (hybrid dense + sparse BM25).
> **Embeddings**: Pinecone inference API (server-side, no local model).
> **Drive auth**: Google Service Account (no user OAuth for Drive).
> **User auth**: Google OAuth as identity provider → server-issued JWT → email allowlist in GCS.
>
> This document is the single source of truth. If a sprint disagrees with this header, the header wins.

---

## Standing rules for every sprint (Claude Code: read this before each sprint)

1. **Do not synthesize or fabricate inputs.** If a required file, env var, OAuth token, or upstream artifact is missing, **stop and ask**. Never generate fake test data or stub a missing dependency to make a test "pass." This rule overrides any instinct to keep momentum.
2. **No scope creep.** If a sprint says "implement X," do not also implement Y, Z, or "while we're here, let's add..." Anything not in the sprint goes into `BACKLOG.md`.
3. **Stay within the locked architectural decisions** (see header). Do not propose alternative transports, chunking strategies, or providers mid-sprint. If you believe the locked decision is wrong, stop and surface it as a question.
4. **Token discipline.** Each sprint has a target token budget. If you're at 80% of budget and not done, stop, write what's done into `SPRINT_NOTES.md`, and ask for direction.
5. **Tests are not optional**, but they are minimal: unit tests for pure functions, one integration test per MCP tool, no test for things that require live external APIs (mock them).
6. **Type-check with pyright strict.** Lint with ruff. Format with ruff format. Every sprint ends green.
7. **Commit at the end of every sprint** with message `sprint-N: <one-line summary>`. Do not commit mid-sprint unless explicitly told.
8. **Refuse anti-patterns** listed in each sprint. These are explicit "do not do this" items based on common Claude Code drift.

---

## Architecture (current — cloud-native)

```
                        ┌──────────────────────────────────────────┐
                        │          Google Cloud Run                │
                        │          anchor-mcp service              │
                        │                                          │
  Claude Desktop        │  FastMCP (StreamableHTTP)                │
  any OS, any machine   │  ┌────────────────────────────────────┐  │
  ──HTTPS + JWT──────►  │  │ OAuth endpoints                    │  │
                        │  │  /.well-known/oauth-auth-server    │  │
                        │  │  /oauth/authorize  /oauth/token    │  │
                        │  │  /oauth/callback   /admin/users    │  │
                        │  ├────────────────────────────────────┤  │
                        │  │ MCP Tools (readers)                │  │
                        │  │  search        → Pinecone          │  │
                        │  │  get_document  → Pinecone          │  │
                        │  │  list_sources  → GCS sidecar       │  │
                        │  │  verify_claim  → OpenRouter        │  │
                        │  ├────────────────────────────────────┤  │
                        │  │ MCP Tools (admin only)             │  │
                        │  │  sync_drive    → Drive + Pinecone  │  │
                        │  │  add_note      → GCS + Pinecone    │  │
                        │  └────────────────────────────────────┘  │
                        │                                          │
                        │  Service Account → Google Drive          │
                        │  Secret Manager  → all API keys          │
                        └───────────┬──────────────────────────────┘
                                    │
              ┌─────────────────────┼──────────────────────┐
              │                     │                      │
       [Pinecone Serverless]    [Cloud Storage]      [OpenRouter]
       hybrid vectors           file_registry.json   judge LLM
       (dense + sparse BM25)    sync_state.json      (any model)
                                allowlist.json
```

**GCS bucket** (`anchor-{project}-state`):
- `file_registry.json` — indexed file listing (name, chunk count, modified time). Instant `list_sources`, no Pinecone call needed.
- `sync_state.json` — per-file modified time + chunk ID list. Drives incremental sync.
- `allowlist.json` — `{"readers": ["a@org.com"], "admins": ["you@org.com"]}`. Admin edits this to grant/revoke access.
- `notes/` — sidecar markdown for `add_note`.

**Secret Manager secrets** (all accessed by the Cloud Run service account):
- `PINECONE_API_KEY`
- `OPENROUTER_API_KEY`
- `GOOGLE_SERVICE_ACCOUNT_KEY` — JSON key for Drive read access
- `GOOGLE_OAUTH_CLIENT_ID` / `GOOGLE_OAUTH_CLIENT_SECRET` — for user auth dance
- `JWT_SECRET` — symmetric key for signing user JWTs

**Roles**:
- `reader` — search, get_document, list_sources, verify_claim
- `admin` — all reader tools + sync_drive, add_note, /admin/users endpoint

Role is embedded in the JWT claim and checked per-tool at runtime.

---

## Locked dependencies

```toml
# Runtime
mcp[cli]>=1.27            # FastMCP, StreamableHTTP support
google-api-python-client  # Drive API
google-auth-oauthlib      # OAuth flow (user auth dance)
google-auth-httplib2
google-cloud-storage      # GCS state files
pinecone>=3.0             # Pinecone client + inference API
pypdf>=4.0                # PDF extraction
httpx>=0.27               # OpenRouter judge calls
pydantic>=2.0
click>=8.0
cryptography>=42.0        # token encryption
tiktoken>=0.7             # chunk token counting
tenacity>=8.0             # retry logic
PyJWT>=2.0                # JWT issuance and validation
```

No `sentence-transformers`. No `torch`. No `chromadb`. Docker image target: ~90MB.

---

# Sprints 0–6 — COMPLETED

These sprints shipped the local-first stdio version of Anchor. Brief record kept for context.

## Sprint 0 ✓ — README + project skeleton
Delivered: `pyproject.toml`, project tree, MIT license, `.gitignore`, README describing intended v1 product.

## Sprint 1 ✓ — Config + CLI
Delivered: `config.py` (Pydantic `AnchorConfig`), `cli.py` (Click: `init`, `config show/set`, `doctor`, `auth login/status`, `sync`, `serve`), `errors.py`, `secrets.py`.

## Sprint 2 ✓ — Google Drive OAuth + file enumeration
Delivered: `auth.py` (InstalledAppFlow, token encrypted at rest with Fernet), `drive.py` (`DriveClient` with pagination and exponential-backoff retry).

## Sprint 3 ✓ — Text extraction + chunking
Delivered: `extract.py` (PDF via pypdf, plain text, Google Docs export), `chunk.py` (`Chunk` model with deterministic ID, recursive character splitter, tiktoken token counting).

## Sprint 4 ✓ — Vector backend + embedding
Delivered: `embed.py` (BGE-M3 via sentence-transformers, lazy-loaded), `backends/base.py` (Protocol), `backends/chroma_backend.py`, `backends/pinecone_backend.py`.

**Post-sprint architectural change (Sprint 7 prep):**
ChromaDB removed entirely. Pinecone is now the only backend. BGE-M3 removed from server path — Pinecone inference replaces it in Sprint 7-A.

## Sprint 5 ✓ — Sync engine
Delivered: `sync.py` (`SyncState`, `FileState`, `Syncer` with add/update/delete diff, `SyncReport`). Auto-migration: if backend is empty but sync state has entries, forces full re-sync (handles backend switch).

**Post-sprint addition:** `FileRegistry` / `RegistryEntry` sidecar written after every sync. `FileState` now stores `file_name`. `list_sources` reads the sidecar — no backend call.

## Sprint 6 ✓ — MCP server (read tools)
Delivered: `server.py` (FastMCP, `search`, `get_document`, `list_sources`, `sync_drive`). Stdout protected from library noise. `_ensure_config()` lightweight singleton for `list_sources`. `_ensure_initialized()` full singleton (config + backend) for search/get_document/sync. `start_background_init()` for warm startup.

**Post-sprint additions:** device config (cpu/cuda/mps), tqdm progress bars on sync, fix for stdout/stdio corruption.

---

# Sprint 7 — Backend migration: Pinecone inference + hybrid search + GCS  [SONNET] [~25k tokens]

**Goal**: replace local BGE-M3 with Pinecone's inference API, add hybrid (dense + sparse BM25) retrieval for better citation accuracy, and move all state files to Cloud Storage. This makes the server fully stateless — a prerequisite for Cloud Run.

## Pre-thinking checklist
- [ ] Read the header. BGE-M3 and sentence-transformers are **gone**. Do not add them back.
- [ ] Pinecone inference endpoint: `pc.inference.embed(model, inputs, parameters)`. Returns a list of embedding objects. Dense model: `"multilingual-e5-large"` (1024-dim). Sparse model: `"pinecone-sparse-english-v0"` (BM25-based).
- [ ] Hybrid upsert: each vector has both `values` (dense float list) and `sparse_values` (dict of `{indices: [...], values: [...]}`)
- [ ] Hybrid query: pass both `vector` and `sparse_vector`, plus `alpha` (0.0 = keyword-only, 1.0 = semantic-only, default 0.7).
- [ ] GCS access: use `google-cloud-storage`. Service account has `roles/storage.objectAdmin` on the bucket.
- [ ] Local dev fallback: if `GCS_BUCKET` env var is unset, fall back to local file paths. This keeps `anchor sync` working locally during development.
- [ ] The existing Pinecone index was created with dimension 1024 for BGE-M3. `multilingual-e5-large` is also 1024-dim — compatible. But vectors were embedded with a different model, so do a **clean wipe and full re-sync** after this sprint.

## Implementation steps

### 7-A: PineconeEmbedder (replaces Embedder)
1. Create `src/anchor_mcp/embed.py` (rewrite, not extend):
   - Remove `SentenceTransformer` entirely.
   - `PineconeEmbedder` class. `__init__(pc_client, dense_model, sparse_model)`.
   - `embed_chunks(chunks) -> list[HybridEmbedding]` where `HybridEmbedding = {dense: list[float], sparse: SparseValues}`.
   - `embed_query(text) -> HybridEmbedding` — same, single input.
   - `SparseValues` pydantic model: `indices: list[int]`, `values: list[float]`.
   - Batch chunks in groups of 96 (Pinecone inference API limit).
   - Raise `BackendError` with clear message if inference API call fails.
2. Update `pyproject.toml`: remove `sentence-transformers`. `pinecone>=3.0` stays (already there).

### 7-B: Hybrid upsert + query in PineconeBackend
1. Update `backends/pinecone_backend.py`:
   - `upsert(chunks, embeddings: list[HybridEmbedding])` — include `sparse_values` in each vector dict.
   - `query(embedding: HybridEmbedding, top_k, alpha=0.7, file_name_filter=None)` — pass `vector`, `sparse_vector`, `alpha` to Pinecone query.
   - Update `get_chunks_by_file` neutral vector: generate a dummy dense vector via `embed_query` (or use the embedder reference stored at init).
2. Update `backends/base.py` Protocol to use `HybridEmbedding` instead of `list[float]`.
3. Update `VectorBackend` Protocol — `query` now takes `HybridEmbedding`.

### 7-C: GCS state store
1. Create `src/anchor_mcp/state_store.py`:
   - `StateStore` abstract Protocol: `read(key: str) -> bytes | None`, `write(key: str, data: bytes) -> None`.
   - `GCSStateStore(bucket_name: str)` — wraps `google.cloud.storage.Client`. Reads/writes blobs.
   - `LocalStateStore(base_dir: Path)` — reads/writes local files. Used when `GCS_BUCKET` is unset.
   - `get_state_store(config: AnchorConfig) -> StateStore` factory — checks `GCS_BUCKET` env var.
2. Update `FileRegistry.load/save` and `SyncState.load/save` to accept a `StateStore` instead of a `Path`.
3. Update `Syncer.__init__` to take a `StateStore`.
4. Update `server.py` `_ensure_config()` to also initialise a state store singleton (`_state_store`).
5. Update `server.py` `list_sources` to call `_state_store.read("file_registry.json")`.
6. Update `sync_drive` tool to pass `_state_store` to the `Syncer`.
7. Update `cli.py` `sync` command similarly.

### 7-D: Config updates
1. Update `AnchorConfig`:
   - Remove `device` field (no local model).
   - Add `pinecone_dense_model: str = "multilingual-e5-large"`.
   - Add `pinecone_sparse_model: str = "pinecone-sparse-english-v0"`.
   - Add `search_alpha: float = 0.7` (default hybrid blend).
2. Update `cli.py` `init` command: drop device prompt, no mention of chroma.
3. Wipe `~/.anchor/cache/sync_state.json` and Pinecone index before running first post-migration sync.

### 7-E: Update search tool signature
- `search(query, top_k=5, alpha=None, file_name_filter=None)` — `alpha` overrides config default if provided. Add to tool description: "alpha controls keyword vs semantic balance (0=keyword, 1=semantic, default 0.7)."

### 7-F: Tests
- Mock `pc.inference.embed` returning synthetic dense + sparse outputs.
- Hybrid upsert: verify vector dict shape contains both `values` and `sparse_values`.
- Hybrid query: verify `alpha` is forwarded correctly.
- GCS state store: mock `google.cloud.storage.Client`, verify read/write calls.
- Local state store: real filesystem, temp dir, round-trip test.

## Anti-patterns (refuse)
- Do NOT import or reference `sentence_transformers`, `torch`, or `chromadb` anywhere.
- Do NOT add a re-ranking layer. Hybrid is the retrieval improvement; re-ranking is backlog.
- Do NOT compute embeddings in the request path for sync (it's already async enough); batch them.
- Do NOT store raw chunk text in GCS — it belongs in Pinecone metadata, as it is now.

## Acceptance criteria
- `pip install -e .` in a fresh venv has no torch/sentence-transformers download.
- `anchor sync` (with `PINECONE_API_KEY` set, local state store) completes cleanly, hybrid vectors visible in Pinecone console.
- `anchor serve` + MCP Inspector: `search` returns results with both dense and sparse contributing.
- ruff + pyright clean.

---

# Sprint 8 — Cloud deployment: transport + OAuth + Docker + CI/CD  [SONNET] [~28k tokens]

**Goal**: the full Cloud Run deployment. StreamableHTTP transport, Google-OAuth-based user auth (Claude Desktop does the auth dance — no manual token management), Dockerfile, and a single `cloudbuild.yaml` that builds + deploys on push to main.

## Pre-thinking checklist
- [ ] The MCP spec defines OAuth 2.0 for remote servers. Claude Desktop handles the dance automatically when it discovers `/.well-known/oauth-authorization-server`. The server must implement: discovery, `/oauth/authorize`, `/oauth/callback`, `/oauth/token`.
- [ ] Google is the upstream identity provider. Users sign in with their Google account. The server gets their email from Google's userinfo endpoint, checks it against the GCS allowlist, and issues its own short-lived JWT.
- [ ] The JWT is symmetric (HS256), signed with `JWT_SECRET` from Secret Manager. Claims: `sub` (email), `role` (reader/admin), `exp` (24h), `iat`.
- [ ] Drive auth uses a Service Account, not user OAuth. Admin shares their Drive folder with the service account email. Service account JSON is stored in Secret Manager as `GOOGLE_SERVICE_ACCOUNT_KEY`.
- [ ] `--allow-unauthenticated` on Cloud Run means Cloud Run itself does not check auth — the MCP server does. This is correct: MCP auth lives at the application layer.
- [ ] Local stdio mode must still work for dev. The `serve` command gains a `--local` flag that uses stdio transport and skips JWT validation.

## Implementation steps

### 8-A: JWT middleware
1. Create `src/anchor_mcp/auth_middleware.py`:
   - `decode_jwt(token: str) -> JWTClaims` — validates signature, expiry; raises `AuthError` on failure.
   - `JWTClaims` pydantic model: `sub: str` (email), `role: Literal["reader", "admin"]`, `exp: int`.
   - `require_role(role)` decorator for tool functions: reads JWT from FastMCP request context, raises `AuthError` with 403 if wrong role.
2. Apply `@require_role("admin")` to `sync_drive` and `add_note` tools in `server.py`.
3. All other tools: require valid JWT of any role (reader or admin).

### 8-B: OAuth endpoints
1. Add to `server.py` (or a separate `auth_routes.py` mounted on the same app):
   - `GET /.well-known/oauth-authorization-server` — static JSON per MCP OAuth spec:
     ```json
     {
       "issuer": "https://{CLOUD_RUN_URL}",
       "authorization_endpoint": "https://{CLOUD_RUN_URL}/oauth/authorize",
       "token_endpoint": "https://{CLOUD_RUN_URL}/oauth/token",
       "response_types_supported": ["code"],
       "grant_types_supported": ["authorization_code"]
     }
     ```
   - `GET /oauth/authorize?client_id=&redirect_uri=&state=&code_challenge=` — validates params, redirects to `accounts.google.com/o/oauth2/auth` with the server's Google OAuth client ID and `{CLOUD_RUN_URL}/oauth/callback` as redirect URI. Stores `state` in a short-lived in-memory or GCS-backed session.
   - `GET /oauth/callback?code=&state=` — exchanges code with Google for user's email via userinfo endpoint. Checks email against GCS allowlist. Generates a PKCE auth code (short-lived, single-use). Redirects to the original `redirect_uri` with the code.
   - `POST /oauth/token` — exchanges PKCE auth code for a signed JWT. Returns `access_token` (JWT), `token_type: "Bearer"`, `expires_in: 86400`.
   - `GET /health` — 200 OK, no auth required. Used by Cloud Build smoke test.
   - `POST /admin/users` — admin JWT required. Body: `{"email": "x@y.com", "role": "reader"}`. Reads + rewrites `allowlist.json` in GCS.
   - `DELETE /admin/users/{email}` — admin JWT required. Removes from allowlist.
2. Store Google OAuth client ID/secret as env vars from Secret Manager. No hardcoding.

### 8-C: Service Account for Drive
1. Update `auth.py`:
   - Add `load_service_account_credentials() -> google.oauth2.service_account.Credentials`.
   - Reads `GOOGLE_SERVICE_ACCOUNT_KEY` env var (JSON string from Secret Manager).
   - Scope: `https://www.googleapis.com/auth/drive.readonly`.
2. Update `server.py` `_ensure_initialized()`: use `load_service_account_credentials()` when `GOOGLE_SERVICE_ACCOUNT_KEY` is set; fall back to `load_credentials()` (OAuth token file) otherwise.
   This means: local dev still works with the OAuth token. Cloud Run uses the service account. No code path change needed for tools.
3. Keep `auth.py`'s existing OAuth flow for local dev. It is NOT removed.

### 8-D: Transport switch
1. Update `cli.py` `serve` command:
   - Add `--local` flag. Default is HTTP.
   - `--local`: uses `mcp.run(transport="stdio")` (existing behaviour, keeps Claude Code working).
   - Without `--local`: uses `mcp.run(transport="streamable-http", host="0.0.0.0", port=8080)`.
2. Update `claude_desktop_config.json` example in README:
   ```json
   {
     "mcpServers": {
       "anchor": { "url": "https://anchor-xyz-uc.a.run.app" }
     }
   }
   ```
   Claude Desktop handles the OAuth dance automatically when it sees the URL.
3. Keep local config block in README for dev:
   ```json
   {
     "mcpServers": {
       "anchor": {
         "command": "anchor", "args": ["serve", "--local"],
         "env": { "PINECONE_API_KEY": "..." }
       }
     }
   }
   ```

### 8-E: Dockerfile
```dockerfile
FROM python:3.12-slim
WORKDIR /app
COPY pyproject.toml .
COPY src/ src/
RUN pip install --no-cache-dir -e .
EXPOSE 8080
CMD ["anchor", "serve"]
```
No sentence-transformers. No torch. No CUDA. Build time < 2 minutes. Image < 100MB.

### 8-F: cloudbuild.yaml
```yaml
steps:
  - name: gcr.io/cloud-builders/docker
    args: [build, -t, gcr.io/$PROJECT_ID/anchor-mcp:$SHORT_SHA, .]

  - name: gcr.io/cloud-builders/docker
    args: [push, gcr.io/$PROJECT_ID/anchor-mcp:$SHORT_SHA]

  - name: gcr.io/google.com/cloudsdktool/cloud-sdk
    entrypoint: gcloud
    args:
      - run deploy anchor-mcp
      - --image=gcr.io/$PROJECT_ID/anchor-mcp:$SHORT_SHA
      - --region=us-central1
      - --service-account=anchor-sa@$PROJECT_ID.iam.gserviceaccount.com
      - --allow-unauthenticated
      - --set-secrets=PINECONE_API_KEY=PINECONE_API_KEY:latest,OPENROUTER_API_KEY=OPENROUTER_API_KEY:latest,GOOGLE_SERVICE_ACCOUNT_KEY=GOOGLE_SERVICE_ACCOUNT_KEY:latest,GOOGLE_OAUTH_CLIENT_ID=GOOGLE_OAUTH_CLIENT_ID:latest,GOOGLE_OAUTH_CLIENT_SECRET=GOOGLE_OAUTH_CLIENT_SECRET:latest,JWT_SECRET=JWT_SECRET:latest
      - --set-env-vars=GCS_BUCKET=anchor-$PROJECT_ID-state
      - --port=8080

  - name: gcr.io/google.com/cloudsdktool/cloud-sdk
    entrypoint: bash
    args:
      - -c
      - curl -sf https://$(gcloud run services describe anchor-mcp --region=us-central1 --format='value(status.url)')/health

images:
  - gcr.io/$PROJECT_ID/anchor-mcp:$SHORT_SHA
```

### 8-G: GCP setup (one-time, documented as runbook in SPRINT_NOTES.md)
Run these once before the first deploy:
```bash
gcloud services enable run.googleapis.com cloudbuild.googleapis.com \
  secretmanager.googleapis.com storage.googleapis.com artifactregistry.googleapis.com

# Service account
gcloud iam service-accounts create anchor-sa \
  --display-name="Anchor MCP service account"

# Permissions
gcloud projects add-iam-policy-binding $PROJECT_ID \
  --member="serviceAccount:anchor-sa@$PROJECT_ID.iam.gserviceaccount.com" \
  --role="roles/secretmanager.secretAccessor"
gcloud projects add-iam-policy-binding $PROJECT_ID \
  --member="serviceAccount:anchor-sa@$PROJECT_ID.iam.gserviceaccount.com" \
  --role="roles/storage.objectAdmin"

# GCS bucket
gsutil mb -l us-central1 gs://anchor-$PROJECT_ID-state

# Secrets (populated manually with actual values)
for secret in PINECONE_API_KEY OPENROUTER_API_KEY GOOGLE_SERVICE_ACCOUNT_KEY \
              GOOGLE_OAUTH_CLIENT_ID GOOGLE_OAUTH_CLIENT_SECRET JWT_SECRET; do
  gcloud secrets create $secret --replication-policy=automatic
done

# Cloud Build trigger (on push to main)
gcloud builds triggers create github \
  --repo-name=anchor --repo-owner=<github-user> \
  --branch-pattern=^main$ --build-config=cloudbuild.yaml
```

### 8-H: Tests
- `decode_jwt` round-trip: issue → decode → assert claims match.
- `decode_jwt` rejects expired token.
- `decode_jwt` rejects wrong signature.
- Mock the allowlist GCS read; verify `require_role("admin")` blocks readers.
- OAuth callback: mock Google userinfo response; verify JWT is issued for allowlisted email; verify 403 for non-allowlisted email.
- Health endpoint: 200, no auth.

## Anti-patterns (refuse)
- Do NOT put any secret value in `cloudbuild.yaml` or the Dockerfile. Secrets come from Secret Manager at deploy time.
- Do NOT validate JWTs in the OAuth endpoints themselves (those are pre-auth paths). Only tool handlers check the JWT.
- Do NOT use Cloud Run's built-in auth (`--no-allow-unauthenticated`). The MCP spec requires the server to handle auth at the application layer.
- Do NOT remove the `--local` flag or break the stdio path. Local dev must stay intact.

## Acceptance criteria
- `docker build` produces an image under 120MB.
- `cloudbuild.yaml` triggered by push to main builds, pushes, and deploys without manual steps.
- Health endpoint at Cloud Run URL returns 200.
- User opens Claude Desktop, adds the Cloud Run URL as an MCP server, is redirected to Google login, signs in, and can call `list_sources` successfully.
- Admin user can call `sync_drive`; reader gets 403 on the same call.
- `anchor serve --local` still works for local dev via stdio.

---

# Sprint 9 — Write tools: `add_note` + `verify_claim`  [OPUS] [~20k tokens]

**Goal**: append-only knowledge capture and server-side faithfulness verification. **Marked Opus** because the judge prompt is the highest-leverage prose in the codebase — a bad prompt produces confident-sounding misclassifications.

Note: These tools now run server-side on Cloud Run. `add_note` is admin-only (shared KB, not per-user). `verify_claim` is available to all roles.

## Pre-thinking checklist
- [ ] `add_note` writes to GCS (`notes/{timestamp}_{slug}.md`) AND upserts to Pinecone with `source_type: "user_note"` metadata. The note is immediately searchable.
- [ ] `verify_claim` calls OpenRouter via httpx. No SDK. Single POST. Parse strict JSON.
- [ ] If `OPENROUTER_API_KEY` is not set, `verify_claim` is NOT registered as a tool. Server starts; tool simply doesn't exist. Document this.
- [ ] `add_note` requires admin role (role check via JWT middleware from Sprint 8). Rationale: it permanently modifies the shared knowledge base.
- [ ] The judge prompt must output strict JSON. If OpenRouter returns malformed JSON, retry once with "Your last response was not valid JSON. Return only the JSON object." If second attempt also fails, raise `VerificationError`.

## Implementation steps
1. Add `add_note` tool to `server.py`:
   - Signature: `add_note(content: str, title: str, tags: list[str] = []) -> NoteReceipt`.
   - `@require_role("admin")` decorator.
   - Writes `notes/{timestamp}_{slug(title)}.md` to GCS via `_state_store`.
   - Generates hybrid embeddings via `_embedder`, upserts to Pinecone with `source_type: "user_note"` in metadata.
   - Updates `file_registry.json` with the note as a new "file."
   - `NoteReceipt`: `note_id: str`, `title: str`, `chunk_count: int`, `stored_at: str`.
   - Tool description: "**This permanently appends a note to the shared knowledge base. Only call this when the user explicitly says 'remember this,' 'save this,' 'add a note about,' or similar. Admin access required.** The note is immediately searchable by all users."
2. Create `src/anchor_mcp/judge.py`:
   - `OpenRouterJudge(api_key: str, model: str)`.
   - Judge prompt (stored as module-level constant — single source of truth):
     ```
     You are a strict faithfulness judge evaluating whether a CLAIM is supported
     by retrieved EVIDENCE from a knowledge base.

     Verdicts:
     - supported: every factual assertion in the claim is directly stated in the evidence.
     - partially_supported: some assertions are in the evidence; others are absent or contradicted.
     - not_supported: the claim's key assertions are absent from or contradicted by the evidence.

     Output ONLY valid JSON, no explanation outside the JSON:
     {"verdict": "supported|partially_supported|not_supported",
      "rationale": "<1-2 sentences>",
      "evaluated_chunk_ids": ["id1", "id2"]}

     CLAIM: {claim}

     EVIDENCE:
     {evidence_block}
     ```
   - `verify(claim: str, chunk_ids: list[str], backend: VectorBackend) -> VerifyResult`.
   - `VerifyResult`: `verdict: Literal["supported", "partially_supported", "not_supported"]`, `rationale: str`, `evaluated_chunk_ids: list[str]`.
3. Register `verify_claim` tool conditionally (only if `OPENROUTER_API_KEY` is set):
   - Signature: `verify_claim(claim: str, chunk_ids: list[str]) -> VerifyResult`.
   - Tool description: "Verify whether a factual claim is supported by specific evidence chunks from the knowledge base. Call this after making a substantive claim based on `search` results. Pass the claim and the `chunk_id` values from the search results you used. If verdict is not 'supported', you MUST tell the user your claim could not be fully verified against their documents."
4. Tests:
   - `add_note` round-trip: mock GCS + Pinecone. Verify GCS write called, Pinecone upsert called with correct metadata, registry updated.
   - `judge.verify` with mocked OpenRouter for each verdict type.
   - Judge handles malformed JSON (retries once).
   - `verify_claim` not registered when env var absent.

## Anti-patterns (refuse)
- Do NOT let `verify_claim` do its own retrieval. It only operates on chunks the LLM already retrieved. Contract stays small.
- Do NOT add a numeric confidence score. Three-state verdict + rationale only. LLM confidence scores are noise.
- Do NOT call OpenRouter from any module other than `judge.py`.
- Do NOT make `add_note` available to readers. Role check is mandatory.

## Acceptance criteria
- Admin calls `add_note` → note appears in `list_sources` and surfaces in `search` within the same server session.
- `verify_claim` with a claim directly from a retrieved chunk returns `supported`.
- `verify_claim` with a fabricated claim returns `not_supported`.
- Atharva manually runs both through Claude Desktop and documents in `SPRINT_NOTES.md`.

---

# Sprint 10 — Hardening, docs, demo  [SONNET] [~18k tokens]

**Goal**: production-quality polish, reconciled documentation, and a recorded demo. Converts a working system into a CV-worthy, recruiter-shippable artifact.

## Implementation steps
1. Reconcile README against shipped reality. Update architecture diagram, quickstart (now URL-based), add `OPENROUTER_API_KEY` to env docs, remove ChromaDB references.
2. Write `docs/ARCHITECTURE.md` — module map, data flow (ingest path, query path, verification path), GCS schema, JWT claims schema.
3. Write `docs/DESIGN.md` — decision log with rationale for: Pinecone-only (no local DB), Pinecone inference (no local embedder), hybrid search (why dense alone isn't enough for citations), Google OAuth via MCP spec (no manual tokens), service account for Drive (no user-OAuth server-side), OpenRouter for judge (model-agnostic), three-state verdict (no confidence scores), admin-only `add_note` (shared KB ownership).
4. Update `BACKLOG.md` with: OCR, re-ranking, webhook-based incremental sync, per-user note namespaces, usage analytics, PyPI publish.
5. Add `Makefile` targets: `install`, `lint`, `typecheck`, `test`, `serve-local`, `deploy`.
6. Record Loom (5–7 min): add URL to Claude Desktop → Google login → `list_sources` → `search` with citation → `verify_claim` supported → `verify_claim` not_supported → `add_note` → confirm note searchable → `sync_drive` admin demo.
7. Tag `v0.2.0`. Update CHANGELOG.

## Acceptance criteria
- Fresh clone + README → working Cloud Run MCP server in < 15 minutes.
- All docs render cleanly on GitHub.
- Loom is shippable to a recruiter without embarrassment.

---

# Token budget summary

| Sprint | Status | Model | Est. tokens |
|--------|--------|-------|-------------|
| 0 — README + skeleton | ✅ done | Sonnet | ~12k |
| 1 — Config + CLI | ✅ done | Sonnet | ~15k |
| 2 — Drive OAuth | ✅ done | Sonnet | ~20k |
| 3 — Extract + chunk | ✅ done | Sonnet | ~15k |
| 4 — Vector backend + embed | ✅ done | Sonnet | ~20k |
| 5 — Sync engine | ✅ done | Sonnet | ~20k |
| 6 — MCP read tools | ✅ done | Opus | ~20k |
| 7 — Pinecone inference + hybrid + GCS | 🔜 next | Sonnet | ~25k |
| 8 — Cloud deploy + OAuth + Docker + CI | 🔜 | Sonnet | ~28k |
| 9 — `add_note` + `verify_claim` | 🔜 | **Opus** | ~20k |
| 10 — Hardening + docs + demo | 🔜 | Sonnet | ~18k |

Remaining: ~91k tokens across 4 sprints.

---

# Standing instructions for Atharva between sprints

- After each sprint, **run the project yourself for 5 minutes** before starting the next. Catch drift early.
- Maintain `BACKLOG.md` with anything Claude Code surfaces as "future work."
- If a sprint exceeds 1.3× budget, stop and reassess. Either the sprint was mis-scoped or the locked architecture is wrong.
- Keep `SPRINT_NOTES.md` as a running diary. Useful for the DESIGN.md rationale section and for interview storytelling.
- **Before Sprint 8**: set up the GCP project, enable billing, and confirm `gcloud auth login` works in this terminal session. Sprint 8 needs live GCP to verify the deploy.
- **Before Sprint 7**: confirm `PINECONE_API_KEY` is set and accessible. The existing Pinecone index can stay — we'll wipe and re-sync during sprint 7-D.
