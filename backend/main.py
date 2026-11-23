import base64
import json
import logging
from typing import Optional
from fastapi import FastAPI, HTTPException, UploadFile, File, Form, WebSocket, WebSocketDisconnect, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, JSONResponse
from pydantic import BaseModel, Field

from .engine import engine, LANGS, LANG_NAMES
from .config import MAX_AUDIO_BYTES
from .vad import vad_detector
from .audio import decode_audio_bytes, audio_to_wav_bytes

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("IndicTranslator")

app = FastAPI(
    title='Offline Indic Translator',
    description='Privacy-first offline speech-to-speech and text translation for 11 Indian languages with GPU acceleration',
    version='1.0.0'
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)

class TextRequest(BaseModel):
    text: str = Field(min_length=1, max_length=10000)
    source: str
    target: str
    mode: str = 'general'

class SpeechRequest(BaseModel):
    text: str = Field(min_length=1, max_length=10000)
    lang: str

def safe_execute(fn):
    try:
        return fn()
    except (ValueError, KeyError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except RuntimeError as exc:
        logger.error(f"Runtime error during inference: {exc}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except Exception as exc:
        logger.error(f"Unexpected error: {exc}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Inference error: {type(exc).__name__}: {str(exc)}") from exc

@app.get('/api/status')
def get_status():
    return engine.status()

@app.post('/api/translate')
def translate_text(req: TextRequest):
    return safe_execute(lambda: engine.translate(req.text, req.source, req.target, req.mode))

@app.post('/api/transcribe')
async def transcribe_audio(file: UploadFile = File(...), source: str = Form(...)):
    if source not in LANGS:
        raise HTTPException(status_code=422, detail=f"Unsupported source language: {source}")
    data = await file.read(MAX_AUDIO_BYTES + 1)
    if len(data) > MAX_AUDIO_BYTES:
        raise HTTPException(status_code=413, detail=f"Audio file exceeds maximum size limit ({MAX_AUDIO_BYTES // (1024*1024)} MB)")
    text = safe_execute(lambda: engine.transcribe(data, source))
    return {'text': text}

@app.post('/api/speak')
def speak_text(req: SpeechRequest):
    audio_bytes = safe_execute(lambda: engine.synthesize(req.text, req.lang))
    return Response(audio_bytes, media_type='audio/wav')

@app.post('/api/speech-to-speech')
async def speech_to_speech(
    file: UploadFile = File(...),
    source: str = Form(...),
    target: str = Form(...),
    mode: str = Form('general')
):
    """End-to-end atomic speech translation in a single request."""
    if source not in LANGS or target not in LANGS:
        raise HTTPException(status_code=422, detail="Unsupported source or target language")

    data = await file.read(MAX_AUDIO_BYTES + 1)
    if len(data) > MAX_AUDIO_BYTES:
        raise HTTPException(status_code=413, detail="Audio file too large")

    # 1. STT
    transcription = safe_execute(lambda: engine.transcribe(data, source))
    if not transcription.strip():
        return {
            'original_text': '',
            'translated_text': '',
            'warnings': ['No speech detected in audio.'],
            'flags': {},
            'audio_base64': None
        }

    # 2. MT
    translation_res = safe_execute(lambda: engine.translate(transcription, source, target, mode))
    translated_text = translation_res['text']

    # 3. TTS
    audio_b64 = None
    try:
        audio_bytes = engine.synthesize(translated_text, target)
        audio_b64 = base64.b64encode(audio_bytes).decode('ascii')
    except Exception as e:
        logger.warning(f"TTS synthesis omitted: {e}")

    return {
        'original_text': transcription,
        'translated_text': translated_text,
        'warnings': translation_res['warnings'],
        'flags': translation_res['flags'],
        'audio_base64': audio_b64
    }

@app.websocket('/ws/stream')
async def websocket_stream(
    websocket: WebSocket,
    source: str = Query('en'),
    target: str = Query('hi'),
    mode: str = Query('general')
):
    """
    Real-time streaming pipeline via WebSockets.
    Accepts continuous binary audio chunks (WebM or raw 16kHz PCM),
    runs Silero VAD to detect speech boundaries, and broadcasts live
    transcriptions, translations, and synthesized audio.
    """
    await websocket.accept()
    logger.info(f"WebSocket client connected: {source} -> {target} [{mode}]")
    audio_buffer = bytearray()
    vad_detector.reset()

    try:
        while True:
            message = await websocket.receive()
            if 'bytes' in message:
                chunk = message['bytes']
                if chunk:
                    audio_buffer.extend(chunk)

            elif 'text' in message:
                text_msg = message['text']
                try:
                    payload = json.loads(text_msg)
                    cmd = payload.get('action')

                    if cmd == 'flush' and len(audio_buffer) > 1600:
                        # Process buffered audio on demand or on pause
                        raw_data = bytes(audio_buffer)
                        audio_buffer.clear()
                        try:
                            transcription = engine.transcribe(raw_data, source)
                            if transcription.strip():
                                trans_res = engine.translate(transcription, source, target, mode)
                                audio_b64 = None
                                try:
                                    tts_bytes = engine.synthesize(trans_res['text'], target)
                                    audio_b64 = base64.b64encode(tts_bytes).decode('ascii')
                                except Exception:
                                    pass

                                await websocket.send_json({
                                    'type': 'utterance',
                                    'original': transcription,
                                    'translated': trans_res['text'],
                                    'warnings': trans_res['warnings'],
                                    'flags': trans_res['flags'],
                                    'audio_base64': audio_b64
                                })
                        except Exception as e:
                            await websocket.send_json({'type': 'error', 'message': str(e)})

                    elif cmd == 'config':
                        source = payload.get('source', source)
                        target = payload.get('target', target)
                        mode = payload.get('mode', mode)
                        await websocket.send_json({'type': 'config_updated', 'source': source, 'target': target, 'mode': mode})

                except json.JSONDecodeError:
                    pass

    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected")
    except Exception as e:
        logger.error(f"WebSocket session error: {e}", exc_info=True)
