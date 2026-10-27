"""Explicit deployment constraints for supported LLM modes."""
from dataclasses import dataclass

@dataclass(frozen=True)
class ModelProfile:
    mode: str; model: str; minimum_vram_gb: int | None; purpose: str

PROFILES = {
    "gemini": ModelProfile("gemini", "gemini-3.1-flash-lite", None, "STT-only and cloud reference"),
    "gemma_e2b": ModelProfile("gemma_e2b", "gemma4:e2b-it-q4_K_M", 12, "default shared-GPU voice profile"),
    "gemma_12b": ModelProfile("gemma_12b", "configure a 12B Ollama tag", 24, "guarded high-VRAM quality profile"),
}
