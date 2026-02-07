# Deploy HY-Motion on Google Cloud Run (GPU)

Scale to zero when idle — you pay only while instances are running.

Two services:
- **API** (`Dockerfile`) — production, JSON-only, used by the SaaS app. Does not load the wooden body model (`DISABLE_WOODEN_MESH=1`): returns `rot6d`, `transl`, `root_rotations_mat` and `keypoints3d` as zeros; the app does retargeting and ground alignment.
- **Gradio** (`Dockerfile.gradio`) — testing/demo UI for the model

## Prerequisites

- Google Cloud project with billing enabled ($300 free credits for new accounts)
- `gcloud` CLI installed and logged in (`gcloud auth login`)
- Optional: Docker (for local builds)

## 1. One-time setup

```bash
export PROJECT_ID=your-gcp-project-id
export REGION=us-central1
export REPO=hymotion

gcloud config set project $PROJECT_ID

gcloud services enable \
  run.googleapis.com \
  artifactregistry.googleapis.com \
  cloudbuild.googleapis.com

gcloud artifacts repositories create $REPO \
  --repository-format=docker \
  --location=$REGION \
  --description="HY-Motion container images"
```

**GPU-supported regions:** `us-central1`, `europe-west1`, `europe-west4`, `asia-southeast1`.

## 2. Build

### Cloud Build (recommended — no local Docker/GPU needed)

```bash
# API (production)
gcloud builds submit \
  --config cloudbuild.yaml \
  --substitutions=_REGION=$REGION

# Gradio (testing)
gcloud builds submit \
  --config cloudbuild.yaml \
  --substitutions=_SERVICE=gradio,_DOCKERFILE=Dockerfile.gradio,_REGION=$REGION
```

Cloud Build uses a 3-step pipeline: pulls cached images, builds the shared base (CUDA + deps + all models), then builds the thin service layer. First build takes ~20-30 min (model downloads); subsequent builds use cached layers.

### Local build

```bash
# Base (shared — only rebuild when deps or models change)
docker build -f Dockerfile.base -t hymotion-base .

# API
docker build -t hymotion-api .

# Gradio
docker build -f Dockerfile.gradio -t hymotion-gradio .

# Tag and push
gcloud auth configure-docker $REGION-docker.pkg.dev --quiet
docker tag hymotion-api $REGION-docker.pkg.dev/$PROJECT_ID/$REPO/hymotion-api:latest
docker push $REGION-docker.pkg.dev/$PROJECT_ID/$REPO/hymotion-api:latest
```

## 3. Deploy

### API (production) — T4 GPU

With INT4 quantization + MMGP Profile 1, peak VRAM is ~5-6 GB. A T4 (16 GB) has plenty of headroom and costs half as much as an L4.

```bash
gcloud run deploy hymotion-api \
  --image=$REGION-docker.pkg.dev/$PROJECT_ID/$REPO/hymotion-api:latest \
  --region=$REGION \
  --gpu=1 \
  --gpu-type=nvidia-t4 \
  --memory=16Gi \
  --cpu=4 \
  --timeout=600 \
  --max-instances=5 \
  --min-instances=0 \
  --cpu-boost \
  --allow-unauthenticated
```

### Gradio (testing) — T4 GPU

```bash
gcloud run deploy hymotion-gradio \
  --image=$REGION-docker.pkg.dev/$PROJECT_ID/$REPO/hymotion-gradio:latest \
  --region=$REGION \
  --gpu=1 \
  --gpu-type=nvidia-t4 \
  --memory=16Gi \
  --cpu=4 \
  --timeout=600 \
  --max-instances=2 \
  --min-instances=0 \
  --cpu-boost \
  --allow-unauthenticated
```

## 4. Test

```bash
export API_URL=$(gcloud run services describe hymotion-api --region=$REGION --format='value(status.url)')

curl $API_URL/health

curl -X POST $API_URL/v1/motion \
  -H "Content-Type: application/json" \
  -d '{"text":"A person walks forward and waves","duration":3.0,"seed":42}'
```

## 5. Connect to your SaaS app

In `nirvana-animate-saas/.env.local`:

```bash
HY_MOTION_API_URL=https://hymotion-api-xxxxxxxxxx-uc.a.run.app
```

## GPU cost

| GPU | VRAM | Cost/hour | VRAM used | Recommendation |
|-----|------|-----------|-----------|----------------|
| T4 | 16 GB | ~$0.35 | ~5-6 GB | **Use this** (INT4 + MMGP 1) |
| L4 | 24 GB | ~$0.73 | ~5-6 GB | Overkill with current optimizations |

Billed per second. Scale-to-zero = $0 when idle. ~$0.01 per request on T4.

With $300 free credits: ~850 hours of T4 time.

## Update deployment

```bash
# Rebuild API
gcloud builds submit --config cloudbuild.yaml --substitutions=_REGION=$REGION

# Redeploy
gcloud run deploy hymotion-api \
  --image=$REGION-docker.pkg.dev/$PROJECT_ID/$REPO/hymotion-api:latest \
  --region=$REGION
```

## Default optimizations baked into the container

All models (HY-Motion-Lite, Qwen3-8B, CLIP) are pre-downloaded at build time — no network needed at startup.

| Setting | Value | Effect |
|---------|-------|--------|
| `QWEN_QUANTIZATION` | `int4` | Qwen3-8B at ~4-5 GB instead of ~16 GB |
| `DISABLE_PROMPT_ENGINEERING` | `True` | Saves ~4 GB (no prompt rewriter) |
| `MMGP_PROFILE` | `1` | Max offloading — peak ~5-6 GB VRAM |
| `DISABLE_WOODEN_MESH` | `1` | Skips body model (API only) |
| `USE_HF_MODELS` | `0` | Uses pre-downloaded local models |

See `VRAM_OPTIMIZATION_GUIDE.md` for details and tuning options.

## Troubleshooting

**Cold start slow:** GPU containers take 30-60s on first request. `--cpu-boost` helps. For always-warm, set `--min-instances=1` (but you pay for idle time).

**Out of memory:** Verify `MMGP_PROFILE=1` and `QWEN_QUANTIZATION=int4` in container logs. Increase `--memory` if needed (max 32Gi).

**Build timeout:** All three models download during build (~20-30 min). Cloud Build timeout is 40 min in `cloudbuild.yaml`.

**Want faster inference?** Set `MMGP_PROFILE=3` (balanced) or `MMGP_PROFILE=0` (no offloading) — uses more VRAM but runs faster. Still fits on T4 at Profile 3 (~8-10 GB).
