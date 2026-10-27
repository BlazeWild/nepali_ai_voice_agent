import asyncio, base64, io, os, tempfile, wave
from collections import deque
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from .runtime import VoiceRuntime
from .vad import create_vad

app=FastAPI(title="Nepali Realtime Voice Agent"); runtime=None
STATIC=os.path.join(os.path.dirname(__file__),"static")

@app.on_event("startup")
async def startup():
    global runtime; runtime=await asyncio.to_thread(VoiceRuntime)
    print(f"[Server] Ready: {runtime.s.llm_mode}, {runtime.s.stt_backend}, VAD={runtime.s.vad_backend}")

@app.get("/")
async def home(): return FileResponse(os.path.join(STATIC,"index.html"))
@app.get("/favicon.ico",include_in_schema=False)
async def favicon(): return {"status":"none"}
@app.get("/health")
async def health(): return {"status":"ok","llm_mode":runtime.s.llm_mode,"stt":runtime.s.stt_backend,"tts":runtime.s.use_tts,"vad":runtime.s.vad_backend,"show_transcripts":runtime.s.show_transcripts}

def pcm_to_wav(raw):
    samples=np.frombuffer(raw,dtype=np.float32); pcm=(samples*32767).clip(-32768,32767).astype(np.int16); out=io.BytesIO()
    with wave.open(out,"wb") as f: f.setnchannels(1); f.setsampwidth(2); f.setframerate(16000); f.writeframes(pcm.tobytes())
    return out.getvalue()

async def process_turn(websocket,wav_bytes):
    path=None
    try:
        with tempfile.NamedTemporaryFile(suffix=".wav",delete=False) as f: f.write(wav_bytes); path=f.name
        await websocket.send_json({"type":"status","data":"Transcribing…"})
        text=await asyncio.to_thread(runtime.stt.transcribe,path)
        if not text: return await websocket.send_json({"type":"done"})
        if runtime.s.show_transcripts: await websocket.send_json({"type":"transcript","data":text})
        await websocket.send_json({"type":"status","data":"Generating response…"})
        answer=await asyncio.to_thread(runtime.reply,text)
        await websocket.send_json({"type":"token","data":answer})
        if runtime.s.use_tts:
            audio=await runtime.speak(answer); await websocket.send_json({"type":"audio","sentence":answer,"data":base64.b64encode(audio).decode()})
        await websocket.send_json({"type":"done","data":answer})
    except asyncio.CancelledError: raise
    except Exception as exc:
        try: await websocket.send_json({"type":"error","data":str(exc)})
        except Exception: pass
    finally:
        if path and os.path.exists(path): os.unlink(path)

@app.websocket("/ws/realtime")
async def realtime(websocket:WebSocket):
    await websocket.accept(); vad=create_vad(runtime.s); pre=deque(maxlen=10); audio=[]; task=None
    try:
        while True:
            message=await websocket.receive()
            if message["type"]=="websocket.disconnect": break
            raw=message.get("bytes")
            if not raw or len(raw)<640: continue
            pcm=np.frombuffer(raw,dtype=np.float32); result=vad.process(pcm)
            await websocket.send_json({"type":"vad","is_speech":result["is_speech"],"confidence":result["confidence"]})
            if task and not task.done():
                if result["speech_started"]:
                    task.cancel(); await websocket.send_json({"type":"interrupted"}); audio=list(pre)+[raw]; pre.clear()
                continue
            if not result["is_speech"] and not audio: pre.append(raw)
            if result["speech_started"]: audio.extend(pre); pre.clear(); audio.append(raw)
            elif result["is_speech"] or audio: audio.append(raw)
            if result["speech_ended"] and len(audio)>=8:
                wav=pcm_to_wav(b"".join(audio)); audio.clear(); pre.clear(); vad.reset(); task=asyncio.create_task(process_turn(websocket,wav))
    except WebSocketDisconnect: pass
    finally:
        if task and not task.done(): task.cancel()

# Compatibility alias for older clients.
app.websocket("/ws")(realtime)
