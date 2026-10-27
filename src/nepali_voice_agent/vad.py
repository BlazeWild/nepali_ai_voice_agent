"""Streaming endpoint detection with Silero and an energy fallback."""
from collections import deque
import numpy as np
from .config import Settings

class EnergyVAD:
    def __init__(self, silence_ms=600):
        self.silence_ms=silence_ms; self.reset()
    def process(self, chunk):
        rms=float(np.sqrt(np.mean(chunk.astype(np.float32)**2))+1e-10)
        ratio=rms/max(self.noise,1e-8); confidence=float(1/(1+np.exp(-2*(ratio-4))))
        speech=rms>.035 and ratio>4.5; started=ended=False
        if speech:
            if not self.active: started=True; self.active=True
            self.silence=0
        elif self.active:
            self.silence+=len(chunk)*1000/16000
            if self.silence>=self.silence_ms: ended=True; self.active=False; self.silence=0
        else: self.noise=.95*self.noise+.05*rms
        return {"is_speech":speech,"confidence":round(confidence,3),"speech_started":started,"speech_ended":ended}
    def reset(self): self.active=False; self.silence=0; self.noise=.001

class SileroVAD(EnergyVAD):
    def __init__(self, threshold=.5, silence_ms=600):
        super().__init__(silence_ms); from silero_vad import load_silero_vad
        self.model=load_silero_vad(); self.threshold=threshold; self.pending=np.empty(0,dtype=np.float32)
    def process(self, chunk):
        import torch
        self.pending=np.concatenate((self.pending,chunk.astype(np.float32,copy=False))); probs=[]
        while self.pending.size>=512:
            frame,self.pending=self.pending[:512],self.pending[512:]; probs.append(float(self.model(torch.from_numpy(frame),16000).item()))
        confidence=max(probs,default=0); speech=confidence>=self.threshold; started=ended=False
        if speech:
            if not self.active: started=True; self.active=True
            self.silence=0
        elif self.active:
            self.silence+=len(chunk)*1000/16000
            if self.silence>=self.silence_ms: ended=True; self.active=False; self.silence=0
        return {"is_speech":speech,"confidence":round(confidence,3),"speech_started":started,"speech_ended":ended}
    def reset(self): super().reset(); self.pending=np.empty(0,dtype=np.float32)

def create_vad(settings: Settings):
    if settings.vad_backend=="silero":
        try: return SileroVAD(settings.vad_threshold,settings.vad_silence_ms)
        except Exception as exc: print(f"[VAD] Silero unavailable ({exc}); using energy fallback")
    return EnergyVAD(settings.vad_silence_ms)
