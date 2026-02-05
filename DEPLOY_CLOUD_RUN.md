# Deploy HY-Motion on Google Cloud Run (GPU)

Scale to zero when idle — you pay only while instances are running.

Two services:
- **API** (`Dockerfile`) — production, JSON-only, used by the SaaS app. Does not load the wooden body model (`DISABLE_WOODEN_MESH=1`): returns `rot6d`, `transl`, `root_rotations_mat` and `keypoints3d` as zeros; the app does retargeting and ground alignment.
- **Gradio** (`Dockerfile.gradio`) — testing/demo UI for the model

## Prerequisites

- Google Cloud project with billing enabled
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

### Cloud Build (no Docker needed)

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

### Local build

```bash
# API
docker build -t $REGION-docker.pkg.dev/$PROJECT_ID/$REPO/hymotion-api:latest .

# Gradio
docker build -f Dockerfile.gradio \
  -t $REGION-docker.pkg.dev/$PROJECT_ID/$REPO/hymotion-gradio:latest .

# Push
gcloud auth configure-docker $REGION-docker.pkg.dev --quiet
docker push $REGION-docker.pkg.dev/$PROJECT_ID/$REPO/hymotion-api:latest
docker push $REGION-docker.pkg.dev/$PROJECT_ID/$REPO/hymotion-gradio:latest
```

## 3. Deploy

### API (production)

```bash
gcloud run deploy hymotion-api \
  --image=$REGION-docker.pkg.dev/$PROJECT_ID/$REPO/hymotion-api:latest \
  --region=$REGION \
  --gpu=1 \
  --gpu-type=nvidia-l4 \
  --memory=16Gi \
  --cpu=4 \
  --timeout=600 \
  --max-instances=5 \
  --min-instances=0 \
  --cpu-boost \
  --allow-unauthenticated
```

### Gradio (testing)

```bash
gcloud run deploy hymotion-gradio \
  --image=$REGION-docker.pkg.dev/$PROJECT_ID/$REPO/hymotion-gradio:latest \
  --region=$REGION \
  --gpu=1 \
  --gpu-type=nvidia-l4 \
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
export API_URL=https://hymotion-api-xxxxxxxxxx-uc.a.run.app

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

| GPU | VRAM | Cost/hour | Best for |
|-----|------|-----------|----------|
| T4 | 16GB | ~$0.35 | Dev, low-cost |
| L4 | 24GB | ~$0.73 | Production |

Billed per second. Scale-to-zero = $0 when idle.

## Update deployment

```bash
# Rebuild API
gcloud builds submit --config cloudbuild.yaml --substitutions=_REGION=$REGION

# Redeploy
gcloud run deploy hymotion-api \
  --image=$REGION-docker.pkg.dev/$PROJECT_ID/$REPO/hymotion-api:latest \
  --region=$REGION
```

## Troubleshooting

**Cold start slow:** GPU containers take 30-60s on first request. `--cpu-boost` helps. For always-warm, set `--min-instances=1`.

**Out of memory:** Increase `--memory` (max 32Gi with GPU). Verify `QWEN_QUANTIZATION=int4` is set.

**Build timeout:** Model download takes ~15 min. Cloud Build timeout is 40 min in `cloudbuild.yaml`.
