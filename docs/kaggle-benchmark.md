# Kaggle model benchmark

Two notebooks are provided:

- [gemma_benchmark.ipynb](../kaggle/gemma_benchmark.ipynb) compares local LLM text quality and speed.
- [voice_pipeline_test.ipynb](../kaggle/voice_pipeline_test.ipynb) runs prerecorded audio through local IndicConformer STT, a selected Gemma model, and Nepali TTS.

Both install pinned package versions and release GPU memory between major stages.

## Setup

1. Create a Kaggle notebook and select the available **T4 x2** accelerator.
2. Enable Internet.
3. Accept the Gemma access terms on Hugging Face for every model being tested.
4. Add a Hugging Face Read token as the Kaggle secret `HF_TOKEN` and enable notebook access.
5. For the voice notebook, attach the IndicConformer model directory and a mono 16 kHz PCM WAV sample as Kaggle datasets.
6. Open or copy the desired notebook and update its `/kaggle/input/...` paths.

Never put the token in a code cell or save model-cache directories as outputs.

## Run order

The notebook starts with `gemma4_e2b`. Confirm that run first, then change:

```python
MODELS_TO_RUN = ["gemma4_e2b", "gemma4_e4b", "gemma4_12b"]
```

Keep the same prompt, quantization, output-token limit, and decoding settings for every model. Record answer quality, completion time, output tokens, and peak GPU memory. E2B is the 12–16 GB voice-agent default; E4B and 12B are comparison candidates only until a concurrent STT + LLM memory test passes.

The local LLMs are included by Hugging Face model ID and download when the notebook runs:

| Selection | Hugging Face model |
| --- | --- |
| `gemma4_e2b` | `google/gemma-4-E2B-it` |
| `gemma4_e4b` | `google/gemma-4-E4B-it` |
| `gemma4_12b` | `google/gemma-4-12B-it` |

Kaggle validates prerecorded pipeline correctness and model performance. Test live microphone streaming, Silero endpoint timing, echo handling, and barge-in through the local browser UI because Kaggle's notebook frontend is not the deployment WebSocket environment.

## What to evaluate

- Nepali fluency and instruction following
- Unsupported or invented claims
- Time to first response and total generation time
- Peak VRAM after the model, prompt, and KV cache load
- Full voice-pipeline latency separately: endpointing, STT, LLM, TTS, playback
