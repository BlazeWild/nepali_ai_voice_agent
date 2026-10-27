# Nepali Realtime Voice Agent

A local-first Nepali voice-conversation pipeline designed for real-time
STT → LLM → TTS evaluation and deployment.

```text
Browser microphone → Silero VAD → IndicConformer STT → Gemma or Gemini → Nepali TTS → browser playback
```

The core voice pipeline is deliberately modular: retrieval and other
application capabilities can be added behind the orchestration layer without
changing STT, LLM, or TTS contracts.

## Model decision: Gemma 4 E2B Q4

The selected local model is **Gemma 4 E2B Q4**:

```text
gemma4:e2b-it-q4_K_M
```

It is the practical choice from the recorded E2B/E4B/12B experiments for a
12–16 GB GPU that must also run STT and optional TTS. E4B and 12B are not
inferior models—12B produced the strongest captured long-document quality—but
they are not the local concurrent-voice deployment profile.

| Model | Experiment result | Role in this project |
| --- | --- | --- |
| Gemma 4 E2B Q4 | Completed on one 16 GB T4 | Default local LLM |
| Gemma 4 E4B 4-bit | One 16 GB T4 exhausted memory during generation | Kaggle/high-VRAM comparison only |
| Gemma 4 12B 4-bit | Completed only on T4 x2 | Quality ceiling / human-reviewed benchmark only |

E4B's 16 GB failure was caused by the total inference footprint, not simply
the parameter count: quantized weights, framework-reserved memory, the prompt,
thinking tokens, and KV cache together consumed the card. The recorded failure
was at about 14.56 GiB already in use before another allocation. It may fit
under shorter contexts or a different runtime, but that does not leave reliable
headroom for this agent's STT process on a 12–16 GB GPU.

## Components

| Component | Default | Reason |
| --- | --- | --- |
| VAD | Silero | Speech/end-of-turn detection with an energy fallback. |
| STT | AI4Bharat IndicConformer 600M CTC | Best measured local result: 31.53% WER and 0.024 RTF on the controlled noisy benchmark. |
| Local LLM | Gemma 4 E2B Q4 via Ollama | Chosen shared-memory local profile. |
| Cloud LLM | Gemini | Optional when external processing is allowed. |
| TTS | Edge Nepali neural voice | Optional; enable to measure full speak-to-hear latency. |

## Setup

### Prerequisites

- Python 3.10+
- [Ollama](https://ollama.com/) for the local LLM profile
- NVIDIA drivers/CUDA-compatible environment for GPU acceleration
- The approved IndicConformer ONNX files available locally

### Install the local model

```bash
cp .env.example .env
./scripts/download_models.sh
ollama list
```

The `.env.example` is the default full-local profile:

```env
PIPELINE_PROFILE=full_local
USE_GEMINI=false
OLLAMA_MODEL=gemma4:e2b-it-q4_K_M
LOCAL_LLM_PRESET=gemma_e2b
USE_TTS=true
STT_BACKEND=indic_conformer
```

Keep `OLLAMA_NUM_CTX=1536` and leave nonessential models off the GPU. Check
actual peak memory after loading STT and Ollama with `nvidia-smi`; runtime and
driver overhead differ between systems.

### Python dependencies

```bash
python -m venv .venv
source .venv/bin/activate                 # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
```

## Run status

This repository currently provides the validated pipeline contract,
configuration, dependencies, model-download helper, and package-level
preflight checks. The streaming application server and browser client have not
yet been added, so there is no production voice-server launch command yet.

The intended runtime must implement the architecture in
[docs/architecture.md](docs/architecture.md): final STT text only, a common
Gemini/Ollama interface, optional TTS, and latency metrics for each stage.

Run the available checks from the repository root:

```bash
PYTHONPATH=src python3 -m nepali_voice_agent doctor
STT_ONLY=true PYTHONPATH=src python3 -m nepali_voice_agent stt-check
```

`doctor` validates the LLM profile. `stt-check` validates the required
IndicConformer assets without loading an LLM—the right first test for STT-only
measurement.

## Profiles

See [docs/pipeline_profiles.md](docs/pipeline_profiles.md) for full-local,
Gemini, and instruction-evaluation configuration. To use the hosted LLM,
change only:

```env
USE_GEMINI=true
GEMINI_API_KEY=your_key_here
```

Never commit `.env` or API keys.

## Kaggle model tests

Use Kaggle to reproduce the E2B/E4B/12B quality comparison before changing the
local default. The hardware setup and fair-test procedure are documented in
[docs/kaggle-benchmark.md](docs/kaggle-benchmark.md). Kaggle is for benchmark
runs, not for serving this real-time browser agent.

## Layout

```text
src/nepali_voice_agent/  application package (runtime to be added)
docs/                    architecture, profiles, and benchmark procedure
scripts/                 model-download-only helpers
```

## Safety

Use source-bound workflows and human review for legal, medical, financial, or
other high-impact content. Fluent model output is not an authoritative source.
