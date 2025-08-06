"""
Piper TTS サービス - 高品質ニューラルTTS
要件定義書準拠: 高品質音声合成
"""

import asyncio
import io
import os
import subprocess
import tempfile
import wave
from typing import Optional, Dict
import logging
from datetime import datetime

from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
import numpy as np
from loguru import logger

# FastAPI アプリケーション
app = FastAPI(
    title="Piper TTS Service",
    description="Piper による高品質ニューラル音声合成サービス",
    version="1.0.0"
)

# グローバル変数
tts_voice: Optional = None

# 設定
SAMPLE_RATE = 22050
DEVICE = "cpu"

@app.on_event("startup")
async def startup_event():
    """アプリケーション起動時の初期化"""
    global tts_voice
    
    logger.info("🔊 Piper TTS Service 起動中...")
    
    # Piper TTS モデル読み込み
    try:
        from piper import PiperVoice
        import subprocess
        
        # モデルディレクトリ作成
        os.makedirs("/models", exist_ok=True)
        
        # Piperモデルファイルをダウンロード
        import urllib.request
        
        model_url = "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_US/lessac/medium/en_US-lessac-medium.onnx"
        model_path = "/models/en_US-lessac-medium.onnx"
        config_url = "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_US/lessac/medium/en_US-lessac-medium.onnx.json"
        config_path = "/models/en_US-lessac-medium.onnx.json"
        
        # モデルファイルをダウンロード
        if not os.path.exists(model_path):
            logger.info("📥 Piperモデルをダウンロード中...")
            urllib.request.urlretrieve(model_url, model_path)
            urllib.request.urlretrieve(config_url, config_path)
            logger.info("✅ Piperモデルダウンロード完了")
        
        # Piperモデル読み込み (正しいAPI)
        tts_voice = PiperVoice.load(model_path)
        logger.info("✅ Piper TTS モデル読み込み完了")
        
    except Exception as e:
        logger.error(f"❌ Piper モデル読み込み失敗: {e}")
        raise RuntimeError(f"Piper TTS初期化失敗: {e}")
    
    logger.info("🎉 Piper TTS Service 初期化完了")

async def synthesize_speech(text: str) -> Optional[bytes]:
    """Piper TTSで音声合成"""
    try:
        # Piper TTSで音声データを生成
        buffer = io.BytesIO()
        
        # Piperで音声合成（正しいAPI）
        with wave.open(buffer, 'wb') as wav_file:
            wav_file.setnchannels(1)  # モノラル
            wav_file.setsampwidth(2)  # 16bit
            wav_file.setframerate(SAMPLE_RATE)
            
            # Piperで音声を合成（正しいAPI使用）
            audio_chunks = tts_voice.synthesize(text)
            
            # AudioChunkの集合をPCMデータに変換
            pcm_bytes = []
            sample_rate = None
            
            for audio_chunk in audio_chunks:
                # AudioChunkから16-bit PCMバイトデータを取得
                chunk_bytes = audio_chunk.audio_int16_bytes
                pcm_bytes.append(chunk_bytes)
                
                # サンプルレートを記録（最初のチャンクから）
                if sample_rate is None:
                    sample_rate = audio_chunk.sample_rate
            
            # 全PCMデータを結合
            if pcm_bytes:
                combined_pcm = b''.join(pcm_bytes)
                
                # WAVファイルのサンプルレートを更新
                if sample_rate:
                    wav_file.setframerate(sample_rate)
                
                wav_file.writeframes(combined_pcm)
                logger.info(f"🎵 音声データ結合: {len(pcm_bytes)} chunks, {len(combined_pcm)} bytes, {sample_rate}Hz")
            else:
                logger.warning("⚠️ 音声チャンクが生成されませんでした")
        
        audio_bytes = buffer.getvalue()
        logger.info(f"✅ Piper TTS音声合成完了: {len(audio_bytes)} bytes")
        return audio_bytes
            
    except Exception as e:
        logger.error(f"❌ Piper TTS音声合成エラー: {e}")
        raise HTTPException(status_code=500, detail=f"音声合成に失敗しました: {e}")


@app.get("/health") 
async def health_check():
    """ヘルスチェック"""
    return {
        "status": "ok",
        "service": "piper-tts",
        "model": "en_US-lessac" if tts_voice else "not_loaded",
        "device": DEVICE,
        "sample_rate": SAMPLE_RATE,
        "timestamp": datetime.now().isoformat()
    }

@app.post("/synthesize")
async def synthesize_text(request: dict):
    """テキスト音声合成API"""
    try:
        text = request.get("text", "")
        if not text:
            raise HTTPException(status_code=400, detail="テキストが必要です")
        
        logger.info(f"🔊 音声合成開始: '{text[:50]}...'")
        
        # Piper TTSで音声合成
        audio_data = await synthesize_speech(text)
        
        return Response(
            content=audio_data,
            media_type="audio/wav",
            headers={"Content-Disposition": "attachment; filename=speech.wav"}
        )
        
    except Exception as e:
        logger.error(f"❌ 音声合成API エラー: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8003)