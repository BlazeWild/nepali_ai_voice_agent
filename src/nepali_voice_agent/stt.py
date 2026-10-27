"""Lean local IndicConformer CTC transcription."""
import json, wave
from pathlib import Path
import numpy as np
import torch

class IndicConformer:
    def __init__(self, model_dir: str):
        import onnxruntime as ort
        assets = Path(model_dir) / "assets"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if self.device.type == "cuda" else ["CPUExecutionProvider"]
        self.pre = torch.jit.load(str(assets / "preprocessor.ts"), map_location=self.device)
        self.encoder = ort.InferenceSession(str(assets / "encoder.onnx"), providers=providers)
        self.ctc = ort.InferenceSession(str(assets / "ctc_decoder.onnx"), providers=providers)
        self.vocab = json.loads((assets / "vocab.json").read_text())["ne"]
        self.mask = json.loads((assets / "language_masks.json").read_text())["ne"]
    def transcribe(self, filename: str) -> str:
        with wave.open(filename, "rb") as f:
            if (f.getnchannels(), f.getsampwidth(), f.getframerate()) != (1,2,16000): raise ValueError("Audio must be mono 16 kHz PCM WAV")
            samples=np.frombuffer(f.readframes(f.getnframes()), dtype="<i2").astype(np.float32)/32768
        wav=torch.from_numpy(samples).unsqueeze(0).to(self.device); length=torch.tensor([wav.shape[-1]],device=self.device)
        signal,length=self.pre(input_signal=wav,length=length)
        encoded,_=self.encoder.run(["outputs","encoded_lengths"],{"audio_signal":signal.cpu().numpy(),"length":length.cpu().numpy()})
        logits=self.ctc.run(["logprobs"],{"encoder_output":encoded})[0]
        ids=torch.from_numpy(logits[:,:,self.mask]).log_softmax(-1).argmax(-1)[0]
        return "".join(self.vocab[i] for i in torch.unique_consecutive(ids) if i != 256).replace("▁"," ").strip()
