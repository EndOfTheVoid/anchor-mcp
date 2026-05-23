# Anchor Sprint Notes

## Sprint 8 — Cloud deployment (2026-05-23)

### GCP one-time setup runbook

Run these once per project before the first `cloudbuild.yaml` deploy.

```bash
export PROJECT_ID=your-gcp-project-id

# Enable APIs
gcloud services enable \
  run.googleapis.com \
  cloudbuild.googleapis.com \
  secretmanager.googleapis.com \
  storage.googleapis.com \
  artifactregistry.googleapis.com

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

gcloud projects add-iam-policy-binding $PROJECT_ID \
  --member="serviceAccount:anchor-sa@$PROJECT_ID.iam.gserviceaccount.com" \
  --role="roles/run.invoker"

# GCS state bucket
gsutil mb -l us-central1 gs://anchor-$PROJECT_ID-state

# Create secrets (then populate each with actual values)
for secret in PINECONE_API_KEY OPENROUTER_API_KEY GOOGLE_SERVICE_ACCOUNT_KEY \
              GOOGLE_OAUTH_CLIENT_ID GOOGLE_OAUTH_CLIENT_SECRET JWT_SECRET; do
  gcloud secrets create $secret --replication-policy=automatic
done

# Populate secrets (run once per secret with real values)
echo -n "your-pinecone-key" | gcloud secrets versions add PINECONE_API_KEY --data-file=-
echo -n "your-jwt-secret-min-32-chars" | gcloud secrets versions add JWT_SECRET --data-file=-
# GOOGLE_SERVICE_ACCOUNT_KEY: paste the full JSON key from GCP Console
cat service-account-key.json | gcloud secrets versions add GOOGLE_SERVICE_ACCOUNT_KEY --data-file=-

# Cloud Build trigger (push to main → deploy)
gcloud builds triggers create github \
  --repo-name=anchor \
  --repo-owner=<your-github-username> \
  --branch-pattern='^main$' \
  --build-config=cloudbuild.yaml

# Grant Cloud Build access to Secret Manager
CLOUDBUILD_SA=$(gcloud projects describe $PROJECT_ID \
  --format='value(projectNumber)')@cloudbuild.gserviceaccount.com

gcloud projects add-iam-policy-binding $PROJECT_ID \
  --member="serviceAccount:$CLOUDBUILD_SA" \
  --role="roles/secretmanager.secretAccessor"

gcloud projects add-iam-policy-binding $PROJECT_ID \
  --member="serviceAccount:$CLOUDBUILD_SA" \
  --role="roles/run.admin"

gcloud iam service-accounts add-iam-policy-binding \
  anchor-sa@$PROJECT_ID.iam.gserviceaccount.com \
  --member="serviceAccount:$CLOUDBUILD_SA" \
  --role="roles/iam.serviceAccountUser"
```

### Google OAuth client setup

1. GCP Console → APIs & Services → Credentials → Create OAuth 2.0 Client ID
2. Application type: Web application
3. Authorized redirect URIs: `https://<CLOUD_RUN_URL>/oauth/callback`
4. Copy Client ID and Client Secret → populate `GOOGLE_OAUTH_CLIENT_ID` / `GOOGLE_OAUTH_CLIENT_SECRET` secrets

### Drive service account setup

1. Share the Google Drive folder with the service account email:
   `anchor-sa@<PROJECT_ID>.iam.gserviceaccount.com`
2. Give "Viewer" access
3. Create a JSON key for the service account → populate `GOOGLE_SERVICE_ACCOUNT_KEY` secret

### First deploy

```bash
# Set SERVER_URL after first manual deploy to get the Cloud Run URL
gcloud run deploy anchor-mcp \
  --image=gcr.io/$PROJECT_ID/anchor-mcp:latest \
  --region=us-central1 \
  --service-account=anchor-sa@$PROJECT_ID.iam.gserviceaccount.com \
  --allow-unauthenticated \
  --set-secrets=... \
  --set-env-vars=GCS_BUCKET=anchor-$PROJECT_ID-state \
  --port=8080

# Get the URL
CLOUD_RUN_URL=$(gcloud run services describe anchor-mcp \
  --region=us-central1 --format='value(status.url)')

# Update SERVER_URL env var (must match issuer in JWTs)
gcloud run services update anchor-mcp \
  --region=us-central1 \
  --set-env-vars=GCS_BUCKET=anchor-$PROJECT_ID-state,SERVER_URL=$CLOUD_RUN_URL
```

### Seeding the allowlist

After first deploy, add yourself as admin:

```bash
# Write allowlist.json directly to GCS
echo '{"readers": [], "admins": ["you@gmail.com"]}' | \
  gsutil cp - gs://anchor-$PROJECT_ID-state/allowlist.json
```

### Claude Desktop config

For Cloud Run (production):
```json
{
  "mcpServers": {
    "anchor": {
      "url": "https://anchor-xyz-uc.a.run.app"
    }
  }
}
```

For local dev (stdio):
```json
{
  "mcpServers": {
    "anchor": {
      "command": "anchor",
      "args": ["serve", "--local"],
      "env": {
        "PINECONE_API_KEY": "your-key"
      }
    }
  }
}
```

### Architecture decisions

- **`--allow-unauthenticated` on Cloud Run**: Cloud Run itself does not check auth. The MCP server handles auth at the application layer per the MCP OAuth spec. This is intentional.
- **HS256 JWT, not RS256**: Simpler key management (one symmetric secret in Secret Manager vs. key pair rotation). 24h TTL limits exposure window.
- **In-memory OAuth state**: Single-instance Cloud Run is fine. If multi-instance is needed, migrate `_oauth_pending` / `_auth_codes` to GCS or Redis.
- **Service account for Drive**: Admin shares the Drive folder once with the SA email. No per-user OAuth tokens to manage server-side.
- **`JWT_SECRET` from Secret Manager**: Rotatable without redeployment by updating the secret version and restarting the service.
