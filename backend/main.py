"""
統合APIゲートウェイ - マイクロサービス連携
要件定義書準拠: FastAPI + Redis Streams
"""

import asyncio
import time
from typing import Dict, List, Optional
import logging
from datetime import datetime

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
import redis.asyncio as redis
import httpx
from loguru import logger
from prometheus_client import Counter, Histogram, generate_latest

# FastAPI アプリケーション
app = FastAPI(
    title="AI English Conversation Gateway",
    description="マイクロサービス統合API - 要件定義書準拠",
    version="1.0.0"
)

# CORS設定
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# グローバル変数
redis_client: Optional[redis.Redis] = None

# 設定
REDIS_URL = "redis://redis:6379"
WHISPER_SERVICE_URL = "http://whisper:8001"
LLM_SERVICE_URL = "http://llm:8002"
TTS_SERVICE_URL = "http://tts:8003"

# Prometheus メトリクス
request_count = Counter('requests_total', 'Total requests', ['method', 'endpoint'])
request_duration = Histogram('request_duration_seconds', 'Request duration')
pipeline_duration = Histogram('pipeline_duration_seconds', 'Pipeline processing time', ['stage'])

@app.on_event("startup")
async def startup_event():
    """アプリケーション起動時の初期化"""
    global redis_client
    
    logger.info("🚀 AI Conversation Gateway 起動中...")
    
    # Redis接続
    try:
        redis_client = redis.from_url(REDIS_URL, encoding="utf-8", decode_responses=True)
        await redis_client.ping()
        logger.info("✅ Redis 接続成功")
    except Exception as e:
        logger.error(f"❌ Redis 接続失敗: {e}")
        raise
    
    logger.info("🎉 AI Conversation Gateway 初期化完了")

@app.on_event("shutdown")
async def shutdown_event():
    """アプリケーション終了時のクリーンアップ"""
    global redis_client
    if redis_client:
        await redis_client.close()
    logger.info("👋 AI Conversation Gateway 終了")

# API エンドポイント

@app.get("/health")
async def health_check():
    """システム健全性チェック - 要件定義書準拠"""
    services_status = {}
    
    # 各マイクロサービスの状態確認
    services = {
        "whisper": WHISPER_SERVICE_URL,
        "llm": LLM_SERVICE_URL,
        "tts": TTS_SERVICE_URL
    }
    
    for service_name, service_url in services.items():
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(f"{service_url}/health", timeout=5.0)
                services_status[service_name] = response.status_code == 200
        except:
            services_status[service_name] = False
    
    # Redis状態確認
    try:
        await redis_client.ping()
        services_status["redis"] = True
    except:
        services_status["redis"] = False
    
    return {
        "status": "ok",
        "timestamp": datetime.now().isoformat(),
        "services": services_status
    }

@app.get("/metrics")
async def metrics():
    """Prometheus メトリクス - 要件定義書準拠"""
    return Response(generate_latest(), media_type="text/plain")

@app.post("/chat/text")
async def chat_text(request: dict):
    """テキストベースの会話 - LLMサービス直接呼び出し"""
    request_count.labels(method="POST", endpoint="/chat/text").inc()
    
    try:
        with request_duration.time():
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{LLM_SERVICE_URL}/generate",
                    json=request,
                    timeout=10.0  # 10秒に短縮
                )
                
                if response.status_code == 200:
                    return response.json()
                else:
                    raise HTTPException(status_code=response.status_code, detail="LLMサービスエラー")
                    
    except Exception as e:
        logger.error(f"❌ テキストチャットエラー: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/speech/chat")
async def speech_chat(audio_file: UploadFile = File(...)):
    """
    音声ベースの会話 - Redis Streams パイプライン
    要件定義書の音声処理フローに準拠
    """
    request_count.labels(method="POST", endpoint="/speech/chat").inc()
    start_time = time.time()
    
    try:
        with request_duration.time():
            # Step 1: Whisper ASRサービスで音声認識
            with pipeline_duration.labels(stage="asr").time():
                content = await audio_file.read()
                
                async with httpx.AsyncClient() as client:
                    files = {"audio_file": ("audio.wav", content, "audio/wav")}
                    asr_response = await client.post(
                        f"{WHISPER_SERVICE_URL}/transcribe",
                        files=files,
                        timeout=25.0  # 音声認識用に25秒に延長
                    )
                    
                    if asr_response.status_code != 200:
                        raise HTTPException(status_code=500, detail="音声認識に失敗しました")
                    
                    asr_result = asr_response.json()
                    user_text = asr_result.get("transcript", "")
            
            if not user_text.strip():
                logger.warning("⚠️ 音声認識結果が空 - デフォルトメッセージで応答")
                return {
                    "transcription": "",
                    "response": "Sorry, I couldn't hear you clearly. Could you please try again?",
                    "user_text": "",  # 下位互換
                    "ai_response": "Sorry, I couldn't hear you clearly. Could you please try again?",  # 下位互換
                    "conversation_history": [],
                    "asr_confidence": 0.0,
                    "processing_time": time.time() - start_time
                }
            
            # Step 2: LLMサービスで応答生成
            with pipeline_duration.labels(stage="llm").time():
                async with httpx.AsyncClient() as client:
                    llm_response = await client.post(
                        f"{LLM_SERVICE_URL}/generate",
                        json={"text": user_text},
                        timeout=15.0  # 15秒に短縮
                    )
                    
                    if llm_response.status_code != 200:
                        raise HTTPException(status_code=500, detail="LLM応答生成に失敗しました")
                    
                    llm_result = llm_response.json()
                    ai_response = llm_result.get("response", "")
            
            # Step 3: TTSサービスで音声合成
            with pipeline_duration.labels(stage="tts").time():
                async with httpx.AsyncClient() as client:
                    tts_response = await client.post(
                        f"{TTS_SERVICE_URL}/synthesize",
                        json={"text": ai_response},
                        timeout=15.0  # 15秒に短縮
                    )
                    
                    if tts_response.status_code != 200:
                        logger.warning("TTS失敗 - テキストのみ返却")
                        tts_audio = None
                    else:
                        tts_audio = tts_response.content
            
            # 結果返却（フロントエンド互換形式）
            result = {
                "transcription": user_text,
                "response": ai_response,
                "user_text": user_text,  # 下位互換
                "ai_response": ai_response,  # 下位互換
                "conversation_history": llm_result.get("conversation_history", []),
                "asr_confidence": asr_result.get("confidence", 0.0),
                "processing_time": time.time() - start_time
            }
            
            logger.info(f"音声会話完了: {user_text[:30]}... → {ai_response[:30]}...")
            return result
            
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        logger.error(f"❌ 音声会話エラー: {e}")
        logger.error(f"詳細なエラートレース: {error_details}")
        raise HTTPException(status_code=500, detail=f"音声処理エラー: {str(e)}")

@app.post("/api/speech/synthesize")
async def synthesize_speech(request: dict):
    """テキスト音声合成 - TTSサービス呼び出し"""
    request_count.labels(method="POST", endpoint="/speech/synthesize").inc()
    
    try:
        with request_duration.time():
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{TTS_SERVICE_URL}/synthesize",
                    json=request,
                    timeout=10.0  # 10秒に短縮
                )
                
                if response.status_code == 200:
                    return Response(
                        content=response.content,
                        media_type="audio/wav",
                        headers={"Content-Disposition": "attachment; filename=speech.wav"}
                    )
                else:
                    raise HTTPException(status_code=response.status_code, detail="TTS生成エラー")
                    
    except Exception as e:
        logger.error(f"❌ 音声合成エラー: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/conversation/reset")
async def reset_conversation():
    """会話履歴リセット"""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(f"{LLM_SERVICE_URL}/reset", timeout=10.0)
            
            if response.status_code == 200:
                return {"message": "会話履歴をリセットしました"}
            else:
                raise HTTPException(status_code=500, detail="リセットに失敗しました")
                
    except Exception as e:
        logger.error(f"❌ 会話リセットエラー: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Redis Streams 監視（将来のWebSocket実装用）
async def monitor_streams():
    """Redis Streamsの監視（WebSocket実装時に使用）"""
    logger.info("📡 Redis Streams 監視開始")
    # 将来のWebSocket実装時に使用

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)