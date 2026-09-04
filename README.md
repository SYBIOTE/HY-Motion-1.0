# HY-Motion 1.0 — RunPod API branch

[Tencent HY-Motion upstream](https://github.com/Tencent-Hunyuan/HY-Motion-1.0) · [Paper](https://arxiv.org/pdf/2512.23464) · [Models (Hugging Face)](https://huggingface.co/tencent/HY-Motion-1.0)

This branch is trimmed for **GPU inference on [RunPod](https://www.runpod.io/)** (Serverless + network volume). It ships a **RunPod queue worker** (`handler.py` over the `api.py` runtime) and Docker images without Gradio, Cloud Build, or SSAE/eval tooling.

| Doc | Purpose |
|-----|---------|
| [RUNPOD.md](RUNPOD.md) | Build, push, env vars, volume layout |
| [API_README.md](API_README.md) | Job contract (`/run`, `/status`) |
| [ckpts/README.md](ckpts/README.md) | Weight layout for offline / volume sync |

**License:** see `License.txt` (same as upstream where applicable).
