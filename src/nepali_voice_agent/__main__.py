"""Operational checks; the streaming server is intentionally a separate future adapter."""
import argparse
from pathlib import Path
from .config import Settings
from .profiles import PROFILES

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["doctor", "stt-check", "serve"])
    args = parser.parse_args()
    settings = Settings(); settings.validate()
    if args.command == "serve":
        import uvicorn
        uvicorn.run("nepali_voice_agent.server:app", host="0.0.0.0", port=8000)
    elif args.command == "doctor":
        profile = PROFILES[settings.llm_mode]
        print(f"LLM mode: {profile.mode} ({profile.purpose})")
        print(f"STT: {settings.stt_backend}; TTS: {'on' if settings.use_tts else 'off'}")
        print("STT-only mode" if settings.stt_only else "Full pipeline mode")
    else:
        assets = Path(settings.stt_model_dir) / "assets"
        required = ["preprocessor.ts", "encoder.onnx", "ctc_decoder.onnx", "vocab.json", "language_masks.json"]
        missing = [name for name in required if not (assets / name).is_file()]
        if missing: raise SystemExit(f"Missing STT assets: {', '.join(missing)}")
        print("IndicConformer assets are present. Run the dedicated STT benchmark with a WAV fixture next.")
if __name__ == "__main__": main()
