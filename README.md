# Anchor

**An MCP server that grounds LLMs in your Google Drive — source-cited hybrid retrieval, server-side faithfulness verification, and a cloud-native multi-user deployment.**

---

## The problem

Your meeting transcripts, project notes, and reference docs live in Google Drive, but every LLM conversation starts from zero — the model hallucinates details or you waste time copy-pasting context. Anchor turns a Drive folder into a citable knowledge base that any MCP client can query, with a built-in judge that checks whether the model's claims are actually supported by your documents.

---

## What Anchor does

Anchor is a [Model Context Protocol](https://modelcontextprotocol.io) server. You point it at a Google Drive folder; it extracts, chunks, embeds, and indexes the documents into Pinecone. Any MCP-compatible client (Claude Desktop, Claude Code, Cursor) can then search that corpus with citations and verify claims against the retrieved evidence.

### MCP tools

| Tool | Role | Description |
|---|---|---|
| `search` | reader | Hybrid semantic + keyword search over the corpus. Returns ranked chunks with `chunk_id`, file name, source URL, and relevance score. `alpha` blends keyword (0.0) ↔ semantic (1.0); default 0.7. |
| `get_document` | reader | Reconstruct the full text of a single Drive file by `file_id`. |
| `list_sources` | reader | List every indexed document with chunk counts and modified times (served from a state sidecar — no vector call). |
| `verify_claim` | reader | Faithfulness judge: given a claim and the `chunk_id`s you used, an LLM (via OpenRouter) returns `supported` / `partially_supported` / `not_supported` with a rationale. Only registered when `OPENROUTER_API_KEY` is set. |
| `sync_drive` | **admin** | Re-index the Drive folder — picks up new, modified, and deleted files. |
| `add_note` | **admin** | Append a permanent, immediately-searchable note to the shared knowledge base. |

### How retrieval works

- **Hybrid search.** Each chunk is stored as a dense vector (semantic) *and* a sparse vector (BM25-style keyword). Queries blend both via `alpha`, which materially improves citation accuracy over dense-only search.
- **Server-side embeddings.** Embeddings come from the **Pinecone inference API** (`multilingual-e5-large` dense + `pinecone-sparse-english-v0` sparse) — there is no local model to download, which keeps the container small and startup fast.
- **Deterministic chunking.** Recursive character splitter with tiktoken token counting; chunk IDs are content-hashed so re-syncs are idempotent.

---

## Architecture

```
  Claude Desktop / Claude Code            Google Cloud Run: anchor-mcp
  (any machine)                           ┌───────────────────────────────────┐
        │  HTTPS + Bearer JWT             │ FastMCP (StreamableHTTP) at /mcp   │
        └────────────────────────────────▶ OAuth: /.well-known, /oauth/*      │
                                          │ MCP tools (role-checked per call)  │
                                          │ Drive via Service Account          │
                                          └───┬───────────────┬────────────┬───┘
                                              │               │            │
                                     [Pinecone Serverless] [Cloud Storage] [OpenRouter]
                                     hybrid dense+sparse   state + notes   judge LLM
```

**Identity & roles.** Users sign in with Google (OAuth). Anchor verifies the email against an allowlist in Cloud Storage and issues its own short-lived **HS256 JWT** (`sub`, `role`, `exp`). The role is checked per tool:

- `reader` → `search`, `get_document`, `list_sources`, `verify_claim`
- `admin` → all reader tools **+** `sync_drive`, `add_note`, and the `/admin/users` endpoints

**Drive access** uses a **Google Service Account** (the admin shares the folder with the service-account email), so no per-user Drive OAuth is needed server-side.

**State** lives in a Cloud Storage bucket (`file_registry.json`, `sync_state.json`, `allowlist.json`, `notes/`). With `GCS_BUCKET` unset, Anchor falls back to local files for development.

---

## Use it: connect to a running instance

If someone has already deployed Anchor and added your email to the allowlist, just point your MCP client at the URL. In Claude Desktop's `claude_desktop_config.json` (note the **`/mcp`** suffix):

```json
{
  "mcpServers": {
    "anchor": { "url": "https://YOUR-SERVICE-URL.run.app/mcp" }
  }
}
```

Restart Claude Desktop. It discovers the OAuth endpoints automatically, walks you through Google sign-in, and Anchor's tools appear. No tokens to manage by hand.

---

## Deploy your own (Google Cloud Run)

**Prerequisites:** a GCP project with billing, the `gcloud` CLI, a [Pinecone](https://www.pinecone.io) API key, and (optional, for `verify_claim`) an [OpenRouter](https://openrouter.ai) key.

1. **Enable APIs**

   ```bash
   gcloud services enable run.googleapis.com cloudbuild.googleapis.com \
     secretmanager.googleapis.com storage.googleapis.com \
     artifactregistry.googleapis.com drive.googleapis.com
   ```

2. **Service account** (runtime identity + Drive reader)

   ```bash
   gcloud iam service-accounts create anchor-sa
   SA="anchor-sa@$PROJECT_ID.iam.gserviceaccount.com"
   gcloud projects add-iam-policy-binding $PROJECT_ID \
     --member="serviceAccount:$SA" --role="roles/secretmanager.secretAccessor"
   gcloud projects add-iam-policy-binding $PROJECT_ID \
     --member="serviceAccount:$SA" --role="roles/storage.objectAdmin"
   gcloud iam service-accounts keys create sa-key.json --iam-account=$SA
   ```

3. **State bucket**

   ```bash
   gcloud storage buckets create gs://anchor-$PROJECT_ID-state --location=us-central1
   ```

4. **OAuth client** (console) — *APIs & Services → Credentials → OAuth client ID → Web application*. Configure the consent screen (External) and add yourself as a test user. Save the **Client ID** and **Client secret**; you'll add `https://<service-url>/oauth/callback` to its **Authorized redirect URIs** after the first deploy.

5. **Secrets** — create and populate six secrets:

   ```
   PINECONE_API_KEY  OPENROUTER_API_KEY  GOOGLE_SERVICE_ACCOUNT_KEY
   GOOGLE_OAUTH_CLIENT_ID  GOOGLE_OAUTH_CLIENT_SECRET  JWT_SECRET
   ```

   (`GOOGLE_SERVICE_ACCOUNT_KEY` is the `sa-key.json` contents; `JWT_SECRET` is any 32-byte random hex.)

6. **Deploy** — two passes, because the OAuth issuer/redirect needs the assigned URL:

   ```bash
   # Pass 1 — build + deploy, then read the URL
   gcloud run deploy anchor-mcp --source . --region=us-central1 \
     --service-account=$SA --allow-unauthenticated --port=8080 --timeout=900 \
     --set-env-vars=GCS_BUCKET=anchor-$PROJECT_ID-state,ANCHOR_DRIVE_FOLDER_ID=<FOLDER_ID> \
     --set-secrets=PINECONE_API_KEY=PINECONE_API_KEY:latest,OPENROUTER_API_KEY=OPENROUTER_API_KEY:latest,GOOGLE_SERVICE_ACCOUNT_KEY=GOOGLE_SERVICE_ACCOUNT_KEY:latest,GOOGLE_OAUTH_CLIENT_ID=GOOGLE_OAUTH_CLIENT_ID:latest,GOOGLE_OAUTH_CLIENT_SECRET=GOOGLE_OAUTH_CLIENT_SECRET:latest,JWT_SECRET=JWT_SECRET:latest

   URL=$(gcloud run services describe anchor-mcp --region=us-central1 --format='value(status.url)')

   # Pass 2 — tell the server its own URL (fixes OAuth discovery + JWT issuer)
   gcloud run services update anchor-mcp --region=us-central1 --update-env-vars=SERVER_URL=$URL
   ```

7. **Seed the allowlist** (or no one can sign in), and **share the Drive folder** with the service-account email:

   ```bash
   echo '{"admins":["you@gmail.com"],"readers":[]}' > allowlist.json
   gcloud storage cp allowlist.json gs://anchor-$PROJECT_ID-state/allowlist.json
   ```

8. **Add the redirect URI** `https://<service-url>/oauth/callback` to the OAuth client, then connect your MCP client to `https://<service-url>/mcp` and run `sync_drive` once to index the folder.

**Continuous deployment** is wired via [`cloudbuild.yaml`](cloudbuild.yaml): connect the repo in Cloud Build → Triggers and set the `_DRIVE_FOLDER_ID` and `_SERVER_URL` substitutions. Every push to `master` rebuilds and redeploys.

---

## Local development (stdio)

For development you can run Anchor over stdio with no JWT layer. This path uses a local `config.json` and a Drive OAuth token instead of a service account, and falls back to local files when `GCS_BUCKET` is unset.

```bash
pip install -e .                              # from a clone
anchor init                                   # prompts for Drive folder ID, writes ~/.anchor/config.json
anchor auth login --credentials creds.json    # one-time Drive OAuth (Desktop client)
export PINECONE_API_KEY=...                   # PowerShell: $env:PINECONE_API_KEY="..."
anchor sync                                   # index the folder into Pinecone
anchor serve --local                          # stdio MCP server
```

Claude Desktop config for the local server:

```json
{
  "mcpServers": {
    "anchor": {
      "command": "anchor",
      "args": ["serve", "--local"],
      "env": { "PINECONE_API_KEY": "...", "OPENROUTER_API_KEY": "..." }
    }
  }
}
```

CLI commands: `init`, `config show|set`, `doctor`, `auth login|status`, `sync`, `serve [--local]`.

---

## Configuration

Config is read from `~/.anchor/config.json` (created by `anchor init`). On Cloud Run, where there is no config file, Anchor builds its config from environment variables instead. Secrets are always read from the environment — never stored in config.

### Service config

| Variable | Notes |
|---|---|
| `ANCHOR_DRIVE_FOLDER_ID` | Drive folder to index (config-from-env on Cloud Run). |
| `ANCHOR_PINECONE_INDEX` | Pinecone index name (default `anchor`; auto-created on first sync). |
| `ANCHOR_SEARCH_ALPHA` | Default hybrid blend (default `0.7`). |
| `ANCHOR_JUDGE_MODEL` | OpenRouter model for `verify_claim` (default `anthropic/claude-haiku-4-5`). |
| `GCS_BUCKET` | State bucket. Unset → local file fallback. |
| `SERVER_URL` | Public URL; used for OAuth discovery + JWT issuer. |

### Secrets

| Secret | Required | Purpose |
|---|---|---|
| `PINECONE_API_KEY` | Yes | Vector storage + inference embeddings. |
| `OPENROUTER_API_KEY` | No | Enables `verify_claim`. |
| `GOOGLE_SERVICE_ACCOUNT_KEY` | Cloud | Service-account JSON for Drive read access. |
| `GOOGLE_OAUTH_CLIENT_ID` / `GOOGLE_OAUTH_CLIENT_SECRET` | Cloud | User sign-in (Google as identity provider). |
| `JWT_SECRET` | Cloud | Signs the server-issued HS256 JWTs. |

---

## Supported file types

- PDF (text-extractable — **no OCR**, scanned PDFs error out)
- Plain text and Markdown (`.txt`, `.md`)
- Google Docs (exported to text)

Not supported: Google Sheets/Slides/Forms, binary Office formats (`.docx`, `.xlsx`, `.pptx`), images, audio, video.

---

## Limitations

- **Manual sync.** Anchor doesn't poll Drive; call `sync_drive` (admin) or `anchor sync` to pick up changes.
- **Append-only notes.** `add_note` is permanent; there is no `delete_note`.
- **No OCR** and no re-ranking layer (both are candidate future work).
- **Shared knowledge base.** Notes and the index are shared across all allowlisted users, not per-user.

---

## License

MIT — see [`LICENSE`](LICENSE).
