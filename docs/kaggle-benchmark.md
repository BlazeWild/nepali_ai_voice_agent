# Kaggle model benchmark

Use [gemma_benchmark.ipynb](../kaggle/gemma_benchmark.ipynb) as the canonical Kaggle notebook. It installs pinned package versions and runs one model at a time, releasing GPU memory between models.

## Setup

1. Create a Kaggle notebook and select the available **T4 x2** accelerator.
2. Enable Internet.
3. Accept the Gemma access terms on Hugging Face for every model being tested.
4. Add a Hugging Face Read token as the Kaggle secret `HF_TOKEN` and enable notebook access.
5. Upload this repository as a Kaggle Dataset, then open or copy the notebook.

Never put the token in a code cell or save model-cache directories as outputs.

## Run order

The notebook starts with `gemma4_e2b`. Confirm that run first, then change:

```python
MODELS_TO_RUN = ["gemma4_e2b", "gemma4_e4b", "gemma4_12b"]
```

Keep the same prompt, quantization, output-token limit, and decoding settings for every model. Record answer quality, completion time, output tokens, and peak GPU memory. E2B is the 12–16 GB voice-agent default; E4B and 12B are comparison candidates only until a concurrent STT + LLM memory test passes.

## What to evaluate

- Nepali fluency and instruction following
- Unsupported or invented claims
- Time to first response and total generation time
- Peak VRAM after the model, prompt, and KV cache load
- Full voice-pipeline latency separately: endpointing, STT, LLM, TTS, playback
