"""
Whisper ASR Service - faster‑whisper (refactored)
既存の機能を維持したまま、重複コードと不要処理を削除してシンプルに整理
"""

import os
import tempfile
import subprocess
import uuid
from datetime import datetime
from typing import Optional

import redis.asyncio as redis
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from faster_whisper import WhisperModel
from loguru import logger

# ---------------------------------------------------------------------------
# 設定値
# ---------------------------------------------------------------------------
REDIS_URL = os.getenv("REDIS_URL", "redis://redis:6379")
WHISPER_MODEL_SIZE = os.getenv("WHISPER_MODEL_SIZE", "small")
DEVICE = os.getenv("WHISPER_DEVICE", "cpu")
SAMPLE_RATE = 16_000            # Whisper が好むサンプリングレート

# ---------------------------------------------------------------------------
# FastAPI アプリケーション定義
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Whisper ASR Service",
    description="faster‑whisper による音声認識サービス (refactored)",
    version="1.1.0",
)

redis_client: Optional[redis.Redis] = None
whisper_model: Optional[WhisperModel] = None

# ---------------------------------------------------------------------------
# ヘルパ関数
# ---------------------------------------------------------------------------

def _load_whisper() -> WhisperModel:
    """Whisper モデルを読み込む (CPU / int8)"""
    try:
        logger.info(f"Loading Whisper model: {WHISPER_MODEL_SIZE}")
        return WhisperModel(
            WHISPER_MODEL_SIZE,
            device=DEVICE,
            compute_type="int8",
            local_files_only=False,  # オンラインダウンロードを許可
        )
    except Exception as e:
        logger.error(f"Whisperモデルロード失敗: {e}")
        # より小さいモデルでフォールバック
        logger.info("Falling back to base model...")
        return WhisperModel(
            "base",
            device=DEVICE,
            compute_type="int8",
            local_files_only=False,
        )


def _to_wav(src_path: str, mime: str) -> str:
    """WebM / Ogg を 16kHz/mono WAV に変換。既に WAV ならそのまま返す"""
    if mime not in ("audio/webm", "audio/ogg"):
        return src_path

    dst_path = f"{os.path.splitext(src_path)[0]}_{uuid.uuid4().hex}.wav"
    cmd = [
        "ffmpeg", "-y",
        "-i", src_path,
        "-ar", str(SAMPLE_RATE), "-ac", "1",
        dst_path,
    ]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        os.unlink(src_path)  # 元ファイル削除
        logger.info(f"🔄 WebM→WAV 変換完了: {dst_path}")
        return dst_path
    except subprocess.CalledProcessError as e:
        logger.error(f"ffmpeg 変換失敗: {e}")
        raise HTTPException(status_code=500, detail="音声ファイル変換に失敗しました")


def _clean_segments(segments):
    """連続する重複セグメントを除去して 1 本の文字列に結合"""
    texts, last = [], None
    for seg in segments:
        txt = seg.text.strip()
        if txt and txt != last:
            texts.append(txt)
            last = txt
    return " ".join(texts)

# ---------------------------------------------------------------------------
# ライフサイクルイベント
# ---------------------------------------------------------------------------
@app.on_event("startup")
async def on_startup():
    """Redis 接続 & Whisper モデル読み込み"""
    global redis_client, whisper_model
    logger.info("🚀 Whisper ASR Service 起動中…")

    # Redis はオプション
    try:
        redis_client = redis.from_url(REDIS_URL, encoding="utf-8", decode_responses=False)
        await redis_client.ping()
        logger.info("✅ Redis 接続成功")
    except Exception as e:
        logger.warning(f"Redis 未接続: {e}")
        redis_client = None

    whisper_model = _load_whisper()
    logger.info(f"✅ Whisper '{WHISPER_MODEL_SIZE}' モデル読み込み完了 ({DEVICE})")

# ---------------------------------------------------------------------------
# エンドポイント
# ---------------------------------------------------------------------------
@app.get("/health")
async def health_check():
    """シンプルなヘルスチェック"""
    return {
        "status": "ok",
        "service": "whisper-asr",
        "model": WHISPER_MODEL_SIZE,
        "device": DEVICE,
        "timestamp": datetime.now().isoformat(),
        "redis": bool(redis_client),
        "model_loaded": whisper_model is not None,
    }


@app.post("/transcribe")
async def transcribe(audio_file: UploadFile = File(...)):
    """音声ファイルを文字起こしして JSON で返却"""
    try:
        # 受信ファイルを一時保存
        suffix = ".webm" if audio_file.content_type == "audio/webm" else ".wav"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            content = await audio_file.read()
            tmp.write(content)
            tmp_path = tmp.name
        logger.info(f"🎤 受信: {audio_file.filename} ({len(content)} bytes)")

        # 必要に応じて WAV 変換
        wav_path = _to_wav(tmp_path, audio_file.content_type)

        # Whisper 推論
        segments, info = whisper_model.transcribe(
            wav_path,
            language="en",
            vad_filter=True,
            vad_parameters={
                "min_silence_duration_ms": 1500,
                "max_speech_duration_s": 30,
                "min_speech_duration_ms": 300,
            },
            word_timestamps=False,
            condition_on_previous_text=False,
            no_speech_threshold=0.6,
        )

        transcript = _clean_segments(list(segments))
        os.unlink(wav_path)

        if not transcript:
            return JSONResponse(
                status_code=200,
                content={
                    "transcript": "",
                    "confidence": 0.0,
                    "language": "en",
                    "message": "音声が検出されませんでした。もう一度お試しください。",
                },
            )

        return {
            "transcript": transcript,
            "confidence": info.language_probability,
            "language": info.language,
        }

    except Exception as e:
        logger.exception("Transcription failed")
        raise HTTPException(500, str(e))


# ---------------------------------------------------------------------------
# ローカル実行用
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
