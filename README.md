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

### Install Gemma 4 E2B with Ollama

Install [Ollama](https://ollama.com/download), start its service, and download
the default model:

```bash
ollama serve
# In another terminal:
ollama pull gemma4:e2b-it-q4_K_M
ollama list
```

The repository helper runs the same pull command:

```bash
./scripts/download_models.sh
```

Create the application configuration with `cp .env.example .env`. The default
local settings select `LLM_MODE=gemma_e2b`,
`OLLAMA_MODEL=gemma4:e2b-it-q4_K_M`, `OLLAMA_THINK=false`, a 1536-token context, and local
IndicConformer STT. Ollama selects CPU/GPU placement based on available
hardware; `ollama ps` reports the actual placement after the first request.

Edge TTS requires Internet access, so speech synthesis is not fully offline.

### Python dependencies

```bash
python -m venv .venv
source .venv/bin/activate                 # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
```

## Run status

The implemented path is browser microphone → streaming Silero VAD → local
IndicConformer STT → Gemini or Ollama E2B → Edge TTS → browser playback.

Run the available checks from the repository root:

```bash
PYTHONPATH=src python3 -m nepali_voice_agent doctor
STT_ONLY=true PYTHONPATH=src python3 -m nepali_voice_agent stt-check
```

`doctor` validates the LLM profile. `stt-check` validates the required
IndicConformer assets without loading an LLM—the right first test for STT-only
measurement.

Start the browser voice agent:

```bash
PYTHONPATH=src python3 -m nepali_voice_agent serve
```

Open `http://localhost:8000`, allow microphone access, and click the orb once to start the call. Speak naturally; Silero VAD detects speech and sends the turn automatically after silence. Click the orb again to disconnect. Transcripts are enabled by default (`SHOW_TRANSCRIPTS=true`).

## Gemini mode

To use the hosted LLM, change:

```env
LLM_MODE=gemini
GEMINI_API_KEY=your_key_here
```

Never commit `.env` or API keys.

## Kaggle model tests

Use Kaggle to reproduce the E2B/E4B/12B quality comparison before changing the
local default. The hardware setup and fair-test procedure are documented in
[docs/kaggle-benchmark.md](docs/kaggle-benchmark.md). Kaggle is for benchmark
runs, not for serving this real-time browser agent. Use
[kaggle/voice_pipeline_test.ipynb](kaggle/voice_pipeline_test.ipynb) for a
prerecorded local STT → local Gemma → TTS functional test.

## Layout

```text
src/nepali_voice_agent/  realtime application package
docs/                    Kaggle benchmark procedure
scripts/                 model-download-only helpers
```

## Safety

Use source-bound workflows and human review for legal, medical, financial, or
other high-impact content. Fluent model output is not an authoritative source.
