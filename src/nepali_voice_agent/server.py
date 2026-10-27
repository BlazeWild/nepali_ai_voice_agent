import base64, tempfile, os, asyncio
from fastapi import FastAPI, WebSocket
from fastapi.responses import HTMLResponse
from .runtime import VoiceRuntime
app=FastAPI(title="Nepali Realtime Voice Agent"); runtime=None
@app.on_event("startup")
async def start():
 global runtime; runtime=VoiceRuntime()
@app.get("/health")
def health(): return {"status":"ok","llm_mode":runtime.s.llm_mode,"stt":"indic_conformer","tts":runtime.s.use_tts}
@app.get("/")
def home(): return HTMLResponse(open(os.path.join(os.path.dirname(__file__),"static","index.html")).read())
@app.websocket("/ws")
async def ws(ws:WebSocket):
 await ws.accept()
 while True:
  data=await ws.receive_bytes()
  with tempfile.NamedTemporaryFile(suffix=".wav",delete=False) as f: f.write(data); path=f.name
  try:
   text=await asyncio.to_thread(runtime.stt.transcribe,path); await ws.send_json({"type":"transcript","text":text})
   answer=await asyncio.to_thread(runtime.reply,text); await ws.send_json({"type":"answer","text":answer})
   if runtime.s.use_tts: await ws.send_json({"type":"audio","data":base64.b64encode(await runtime.speak(answer)).decode()})
  except Exception as e: await ws.send_json({"type":"error","text":str(e)})
  finally: os.unlink(path)
