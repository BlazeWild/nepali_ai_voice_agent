# Pipeline profiles

Use one profile at a time. They have the same VAD and STT settings unless stated otherwise.

## 1. Full local voice pipeline

`Silero VAD → IndicConformer STT → local Ollama LLM → TTS`

```env
PIPELINE_PROFILE=full_local
USE_GEMINI=false
LOCAL_LLM_PRESET=gemma_e2b
USE_TTS=true
```

Gemma 4 E2B Q4 (`gemma4:e2b-it-q4_K_M`) is the default. It is the selected
candidate for a 12–16 GB voice pipeline because it leaves headroom for STT,
runtime allocations, and optional TTS. Do not use E4B or 12B in this profile.

## 2. STT plus Gemini plus TTS

`Silero VAD → IndicConformer STT → Gemini → TTS`

```env
PIPELINE_PROFILE=stt_gemini_tts
USE_GEMINI=true
USE_TTS=true
```

This is the quality/reference voice profile.

## 3. Local-LLM instruction evaluation

`typed transliterated input → Gemini normalization to Nepali → local Ollama LLM`

```env
PIPELINE_PROFILE=llm_eval
USE_GEMINI=false
LOCAL_LLM_PRESET=gemma_e2b
USE_TTS=false
```

Gemini is used only as an input normalizer in this profile; the local model produces the answer. The evaluation system prompt permits only Nepal law and justice questions so instruction-following can be measured.

## Model downloads

Gemma is installed locally as `gemma4:e2b-it-q4_K_M`. E4B and 12B are evaluated
in Kaggle/high-VRAM environments only; they are not local voice-profile
downloads.
