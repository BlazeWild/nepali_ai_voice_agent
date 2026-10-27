# Kaggle Gemma benchmark procedure

Use this procedure to compare E2B, E4B, and 12B fairly before changing the
local voice-agent default. It evaluates LLM quality; it does not run the
browser/STT/TTS application.

## Hardware selection

| Test | Kaggle accelerator | Why |
| --- | --- | --- |
| E2B smoke test | T4 x1 (16 GB) | The recorded 4-bit run completed on one T4. |
| E4B or 12B | T4 x2 | A one-T4 E4B run exhausted VRAM during generation. |
| Full comparison | T4 x2 | Keeps the environment consistent for the larger models. |

The earlier E4B out-of-memory event is evidence for the deployment decision,
not a universal minimum-VRAM claim. Context length, thinking mode, batch size,
quantization, and runtime change the peak. The fair comparison should keep
those values identical across candidates.

## Prepare the notebook

1. Open **Kaggle → Code → New Notebook**.
2. Enable **Internet** in notebook settings.
3. Select **GPU T4 x1** for E2B or **GPU T4 x2** for E4B/12B.
4. On Hugging Face, accept the Gemma terms for every model you will test and
   create a Read access token.
5. Add that token as the Kaggle secret `HF_TOKEN`, then enable **Allow notebook
   access**. Never paste it into a notebook cell.

## Minimal reproducible test

Install the required inference libraries in the first notebook cell:

```python
!pip install -q -U "transformers>=5.10.1" accelerate bitsandbytes sentencepiece huggingface_hub
```

Use these fixed controls for every candidate:

```python
REPEATS = 1
MAX_NEW_TOKENS = 2400
TEMPERATURE = 0.0
ENABLE_THINKING = True
```

First run E2B alone with `google/gemma-4-E2B-it` and 4-bit NF4 loading. After
it succeeds, run E4B (`google/gemma-4-E4B-it`) and 12B
(`google/gemma-4-12B-it`) one at a time, with `device_map="auto"` across both
T4 GPUs. Limit each GPU to about `11GiB` to preserve headroom for prompt and KV
cache allocations. Delete the model and call `torch.cuda.empty_cache()` between
models.

For each prompt, save a CSV row containing at least:

```text
model, model_id, test, prompt, answer, thinking, seconds, output_tokens,
manual_score_0_to_5, review_notes
```

Save the CSV from `/kaggle/working/` as a notebook output before ending the
session. Do not publish the Hugging Face token or model-cache directories.

## Review criteria

Score final answers, not hidden/thinking text. Check Nepali grammar,
instruction/format compliance, response completeness, latency, and whether
the answer invents facts. For legal or government documents, also require
consistent placeholders for missing data and reject invented dates, names,
legal provisions, or remedies.

The expected deployment conclusion remains: E2B is the local shared-GPU voice
profile; E4B/12B are quality benchmarks unless a separate VRAM measurement
shows safe concurrent STT + LLM operation on the target hardware.
