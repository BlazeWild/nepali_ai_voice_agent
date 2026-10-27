import asyncio
import base64
import io
import json
import os
import tempfile
import wave
from collections import deque

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, Response

from .runtime import VoiceRuntime
from .vad import create_vad

app = FastAPI(title="Nepali Realtime Voice Agent")
runtime = None
STATIC = os.path.join(os.path.dirname(__file__), "static")


@app.on_event("startup")
async def startup():
    global runtime
    runtime = await asyncio.to_thread(VoiceRuntime)
    print(f"[Server] Ready: {runtime.s.llm_mode}, {runtime.s.stt_backend}, VAD={runtime.s.vad_backend}")


@app.get("/")
async def home():
    return FileResponse(os.path.join(STATIC, "index.html"))


@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    return Response(status_code=204)


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "llm_mode": runtime.s.llm_mode,
        "stt": runtime.s.stt_backend,
        "tts": runtime.s.use_tts,
        "vad": runtime.s.vad_backend,
        "show_transcripts": True,
    }


def pcm_to_wav(raw: bytes) -> bytes:
    samples = np.frombuffer(raw, dtype=np.float32)
    pcm = (samples * 32767).clip(-32768, 32767).astype(np.int16)
    output = io.BytesIO()
    with wave.open(output, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(16000)
        wav.writeframes(pcm.tobytes())
    return output.getvalue()


async def process_turn(websocket: WebSocket, wav_bytes: bytes):
    path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as audio_file:
            audio_file.write(wav_bytes)
            path = audio_file.name
        await websocket.send_json({"type": "status", "data": "Transcribing…"})
        transcript = (await asyncio.to_thread(runtime.stt.transcribe, path)).strip()
        if not transcript:
            await websocket.send_json({"type": "discarded", "reason": "empty_transcript"})
            return
        # Always expose STT during evaluation so false endpoints and recognition
        # mistakes cannot masquerade as LLM/prompt failures.
        await websocket.send_json({"type": "transcript", "data": transcript})
        await websocket.send_json({"type": "status", "data": "Generating response…"})
        answer = (await asyncio.to_thread(runtime.reply, transcript)).strip()
        await websocket.send_json({"type": "token", "data": answer})
        if runtime.s.use_tts and answer:
            audio = await runtime.speak(answer)
            await websocket.send_json({"type": "audio", "sentence": answer, "data": base64.b64encode(audio).decode()})
        await websocket.send_json({"type": "done", "data": answer})
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        try:
            await websocket.send_json({"type": "error", "data": str(exc)})
        except Exception:
            pass
    finally:
        if path and os.path.exists(path):
            os.unlink(path)


async def realtime(websocket: WebSocket):
    await websocket.accept()
    vad = create_vad(runtime.s)
    pre_speech = deque(maxlen=8)
    turn_audio = []
    turn_voiced_ms = 0.0
    barge_audio = []
    barge_voiced_ms = 0.0
    processing_task = None
    playback_active = False

    async def interrupt_agent():
        nonlocal processing_task, playback_active
        if processing_task and not processing_task.done():
            processing_task.cancel()
        processing_task = None
        playback_active = False
        await websocket.send_json({"type": "interrupted"})

    try:
        while True:
            message = await websocket.receive()
            if message["type"] == "websocket.disconnect":
                break

            if message.get("text"):
                try:
                    control = json.loads(message["text"])
                except json.JSONDecodeError:
                    continue
                if control.get("type") == "playback_start":
                    playback_active = True
                elif control.get("type") == "playback_end":
                    playback_active = False
                    barge_audio.clear()
                    barge_voiced_ms = 0.0
                    vad.reset()
                elif control.get("type") == "interrupt":
                    await interrupt_agent()
                continue

            raw = message.get("bytes")
            if not raw or len(raw) < 640 or len(raw) % 4:
                continue
            pcm = np.frombuffer(raw, dtype=np.float32)
            chunk_ms = len(pcm) * 1000.0 / 16000
            result = vad.process(pcm)
            await websocket.send_json({"type": "vad", "is_speech": result["is_speech"], "confidence": result["confidence"]})

            agent_busy = playback_active or (processing_task and not processing_task.done())
            if agent_busy:
                if result["speech_started"]:
                    barge_audio = list(pre_speech) + [raw]
                    pre_speech.clear()
                    barge_voiced_ms = chunk_ms
                elif result["is_speech"] and barge_audio:
                    barge_audio.append(raw)
                    barge_voiced_ms += chunk_ms
                elif barge_audio and not result["speech_ended"]:
                    barge_audio.append(raw)

                if barge_voiced_ms >= runtime.s.vad_barge_in_ms:
                    await interrupt_agent()
                    turn_audio = barge_audio
                    turn_voiced_ms = barge_voiced_ms
                    barge_audio = []
                    barge_voiced_ms = 0.0
                elif result["speech_ended"]:
                    barge_audio = []
                    barge_voiced_ms = 0.0
                continue

            if not result["is_speech"] and not turn_audio:
                pre_speech.append(raw)
            if result["speech_started"]:
                turn_audio = list(pre_speech) + [raw]
                pre_speech.clear()
                turn_voiced_ms = chunk_ms
            elif result["is_speech"] and turn_audio:
                turn_audio.append(raw)
                turn_voiced_ms += chunk_ms
            elif turn_audio:
                turn_audio.append(raw)

            if result["speech_ended"]:
                if turn_voiced_ms >= runtime.s.vad_min_speech_ms:
                    wav_bytes = pcm_to_wav(b"".join(turn_audio))
                    processing_task = asyncio.create_task(process_turn(websocket, wav_bytes))
                else:
                    await websocket.send_json({"type": "discarded", "reason": "speech_too_short"})
                turn_audio = []
                turn_voiced_ms = 0.0
                pre_speech.clear()
                vad.reset()
    except WebSocketDisconnect:
        pass
    finally:
        if processing_task and not processing_task.done():
            processing_task.cancel()


app.websocket("/ws/realtime")(realtime)
app.websocket("/ws")(realtime)
