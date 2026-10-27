import asyncio, tempfile, os
import edge_tts
from google import genai
from .config import Settings
from .prompts import system_prompt
from .stt import IndicConformer

class VoiceRuntime:
    def __init__(self):
        self.s=Settings(); self.s.validate(); self.stt=IndicConformer(self.s.stt_model_dir)
        self.gemini=genai.Client(api_key=self.s.gemini_api_key) if self.s.llm_mode=="gemini" else None
    def reply(self, text: str) -> str:
        if self.gemini:
            response=self.gemini.models.generate_content(model=self.s.gemini_model,contents=f"{system_prompt(self.s.llm_mode)}\n\nUser: {text}")
            return response.text or "माफ गर्नुहोस्, उत्तर प्राप्त भएन।"
        import ollama
        response=ollama.chat(model=self.s.ollama_model,messages=[{"role":"system","content":system_prompt(self.s.llm_mode)},{"role":"user","content":text}],options={"num_ctx":self.s.ollama_num_ctx,"num_predict":self.s.ollama_num_predict})
        return response["message"]["content"]
    async def speak(self,text:str)->bytes:
        with tempfile.NamedTemporaryFile(suffix=".mp3",delete=False) as f: path=f.name
        try:
            await edge_tts.Communicate(text,"ne-NP-SagarNeural").save(path)
            return open(path,"rb").read()
        finally: os.unlink(path)
