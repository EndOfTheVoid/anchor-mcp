# Anchor

**A local-first MCP server that grounds LLMs in your Google Drive corpus — source-cited retrieval, BYOK faithfulness verification, multilingual embeddings.**

---

## The problem

Your meeting transcripts, project notes, and reference docs live in Google Drive, but every LLM conversation starts from zero — the model either hallucinates details or you waste time copy-pasting context. Dedicated knowledge tools like Glean are enterprise-only and require IT approval. Anchor fixes this for a single user with zero cloud infrastructure.

---

## What Anchor does

Anchor is an [MCP](https://modelcontextprotocol.io) server you run locally. Once configured, any MCP-compatible client (Claude Desktop, Cursor, Claude Code) can query your Drive corpus as if the documents were in its context window — with citations you can verify.

### MCP tools

| Tool | Description |
|---|---|
| `search` | Semantic search across all indexed documents. Returns ranked text chunks with source file, page, and URL for citation. |
| `get_document` | Retrieve the full reconstructed text of a single Drive file by ID. |
| `list_sources` | List all indexed documents with chunk counts and last-sync timestamps. |
| `sync_drive` | Re-index the configured Drive folder — picks up new, modified, and deleted files. |
| `add_note` | Append a permanent, searchable note to your knowledge base. The note is written to `~/.anchor/notes/` and ingested into the vector store immediately. |
| `verify_claim` | Given a factual claim and the chunk IDs retrieved during `search`, ask a judge LLM (via OpenRouter) whether the claim is supported, partially supported, or not supported by those chunks. Returns a verdict with rationale. |

---

## Quickstart

### Prerequisites

- Python 3.11+
- A Google Cloud project with the Drive API enabled and an OAuth 2.0 Desktop client credentials file (`credentials.json`). See [Google's guide](https://developers.google.com/drive/api/quickstart/python) for setup.
- An [OpenRouter](https://openrouter.ai) API key (optional — only required for `verify_claim`).

### Install

```bash
pip install anchor-mcp
```

Or with uv:

```bash
uv tool install anchor-mcp
```

### Initialize

```bash
anchor init
```

This creates `~/.anchor/`, prompts for your Drive folder ID, and writes an initial config.

### Authenticate with Google Drive

```bash
anchor auth login --credentials /path/to/credentials.json
```

A browser window opens for the OAuth consent screen. Anchor requests `drive.readonly` scope — it never writes to Drive.

### Sync your Drive folder

```bash
anchor sync
```

On first run this downloads, extracts text, chunks, and embeds every file in the configured folder. Subsequent runs only process changes.

### Configure Claude Desktop

Add to your Claude Desktop `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "anchor": {
      "command": "anchor",
      "args": ["serve"],
      "env": {
        "OPENROUTER_API_KEY": "sk-or-..."
      }
    }
  }
}
```

Restart Claude Desktop. Anchor's tools appear automatically.

---

## Architecture

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

**Local state** (`~/.anchor/`):

```
~/.anchor/
  config.json          # folder ID, backend choice, embedding model
  oauth_token.json     # Drive OAuth refresh token (encrypted at rest)
  chroma/              # ChromaDB persistent storage
  cache/sync_state.json
  notes/               # sidecar markdown for add_note
  logs/anchor.log      # rotating log
```

**Transport**: stdio — Anchor runs as a subprocess of your MCP client. No network port, no daemon.

**Embeddings**: [bge-m3](https://huggingface.co/BAAI/bge-m3) via `sentence-transformers` — free, multilingual, 8192-token context, 1024-dim dense vectors. First run downloads ~2 GB to the HuggingFace cache.

---

## Configuration

All configuration is in `~/.anchor/config.json` (managed by `anchor config`). Secrets are read from environment variables — never stored in config.

| Environment variable | Required | Description |
|---|---|---|
| `OPENROUTER_API_KEY` | No | Enables `verify_claim`. Without it the tool is not registered. |
| `ANCHOR_DRIVE_FOLDER_ID` | Yes (for init) | Drive folder to index. |
| `ANCHOR_VECTOR_BACKEND` | No | `chroma` (default) or `pinecone`. |
| `PINECONE_API_KEY` | If using Pinecone | Pinecone API key. |
| `PINECONE_INDEX_NAME` | If using Pinecone | Index name (default: `anchor`). |

Copy `.env.example` and fill in your values.

---

## Supported file types

**Supported in v1:**
- PDF (text-extractable — not scanned)
- `.txt`, `.md`
- Google Docs (exported as plain text)
- Google Meet transcripts (stored as Google Docs)

**Not supported in v1:**
- Scanned PDFs (no OCR)
- Google Sheets, Slides, Forms
- Binary formats (`.docx`, `.xlsx`, `.pptx`)
- Images, audio, video
- Files larger than 50 MB

---

## Demo

[Loom link: TBA]

---

## Limitations

- **Single-user.** Anchor uses installed-app OAuth for one Google account. Multi-tenant ACL mirroring is future work (see `docs/DESIGN.md`).
- **Append-only notes.** `add_note` writes permanently. There is no `delete_note` in v1.
- **Manual sync.** Anchor does not poll Drive in the background. Call `sync_drive` or run `anchor sync` to pick up changes.
- **No OCR.** Scanned PDFs raise an extraction error.
- **Local only.** The MCP transport is stdio. Anchor is not a web service.

---

## Future work

See [`docs/DESIGN.md`](docs/DESIGN.md) for the decision log and planned extensions, including Streamable HTTP transport, multi-tenant ACL mirroring, OCR, re-ranking, and observability hooks.

---

## License

MIT — see [`LICENSE`](LICENSE).
