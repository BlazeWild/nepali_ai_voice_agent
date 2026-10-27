#!/usr/bin/env bash
set -euo pipefail

# Downloads models only. It never starts the application or modifies .env.
ollama pull gemma4:e2b-it-q4_K_M
echo "Gemma 4 E2B download complete. Configure LLM_MODE=gemma_e2b to use it."
echo "E4B and 12B are benchmark/high-VRAM models; do not use them in the 12–16 GB voice profile."
