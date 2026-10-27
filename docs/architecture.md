# Architecture decisions

- Keep retrieval, tools, and business workflows behind explicit interfaces so they do not alter the STT/LLM/TTS contracts.
- Use final endpointed STT text for LLM input. Partial STT may be shown in the UI later, but should not drive the LLM because it can revise.
- Preserve a common request/response interface for Gemini and Ollama so model comparisons are fair.
- Keep TTS optional while measuring STT and LLM latency independently.
- Record endpoint delay, final-STT time, LLM time-to-first-token, TTS synthesis time, and first-playback time.
