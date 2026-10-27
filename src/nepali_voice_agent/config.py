"""Typed runtime configuration; environment parsing lives only here."""
from dataclasses import dataclass
import os


def _bool(name: str, default: bool) -> bool:
    return os.getenv(name, str(default)).lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    llm_mode: str = os.getenv("LLM_MODE", "gemma_e2b")
    gemini_model: str = os.getenv("GEMINI_MODEL", "gemini-3.1-flash-lite")
    gemini_api_key: str = os.getenv("GEMINI_API_KEY", "")
    ollama_url: str = os.getenv("OLLAMA_URL", "http://localhost:11434/api/chat")
    ollama_model: str = os.getenv("OLLAMA_MODEL", "gemma4:e2b-it-q4_K_M")
    ollama_num_ctx: int = int(os.getenv("OLLAMA_NUM_CTX", "1536"))
    ollama_num_predict: int = int(os.getenv("OLLAMA_NUM_PREDICT", "180"))
    stt_backend: str = os.getenv("STT_BACKEND", "indic_conformer")
    stt_model_dir: str = os.getenv("INDIC_CONFORMER_MODEL_DIR", "../stt_research/models/indic-conformer-600m-multilingual-onnx")
    stt_only: bool = _bool("STT_ONLY", False)
    use_tts: bool = _bool("USE_TTS", True)

    def validate(self) -> None:
        if self.llm_mode not in {"gemini", "gemma_e2b", "gemma_12b"}:
            raise ValueError("LLM_MODE must be gemini, gemma_e2b, or gemma_12b")
        if self.llm_mode == "gemini" and not self.gemini_api_key:
            raise ValueError("GEMINI_API_KEY is required when LLM_MODE=gemini")
        if self.llm_mode == "gemma_12b" and self.ollama_num_ctx > 2048:
            raise ValueError("Use OLLAMA_NUM_CTX<=2048 for the guarded 12B profile; measure VRAM first.")
