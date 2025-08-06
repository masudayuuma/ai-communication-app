"""
Phi-3 LLMサービス - Ollama統合
要件定義書準拠: ストリーミング応答生成
"""

import asyncio
import json
from typing import Optional, List, Dict
import logging
from datetime import datetime

from fastapi import FastAPI, HTTPException
import httpx
from loguru import logger

# FastAPI アプリケーション
app = FastAPI(
    title="Phi-3 LLM Service",
    description="Ollama を使用した対話生成サービス（Phi-3 mini-4k-instruct）",
    version="1.0.0"
)

# グローバル変数
conversation_history: List[Dict] = []

# 設定
OLLAMA_HOST = "127.0.0.1:11434"
MODEL_NAME = "phi3:mini"

@app.on_event("startup")
async def startup_event():
    """アプリケーション起動時の初期化"""
    logger.info("🤖 Phi-3 LLM Service 起動中...")
    
    # Phi-3モデルの確認・自動プル
    try:
        async with httpx.AsyncClient() as client:
            # モデル一覧を確認
            response = await client.get(f"http://{OLLAMA_HOST}/api/tags", timeout=10.0)
            if response.status_code == 200:
                models = response.json().get("models", [])
                model_names = [model["name"] for model in models]
                
                if MODEL_NAME not in model_names:
                    logger.info(f"📥 Phi-3モデル '{MODEL_NAME}' をプル中...")
                    # モデルプル（非同期で開始）
                    await client.post(
                        f"http://{OLLAMA_HOST}/api/pull",
                        json={"name": MODEL_NAME},
                        timeout=300.0  # 5分のタイムアウト
                    )
                    logger.info("✅ Phi-3モデルプル完了")
                else:
                    logger.info(f"✅ Phi-3モデル '{MODEL_NAME}' 利用可能")
            
    except Exception as e:
        logger.warning(f"⚠️ モデル確認/プルエラー: {e}")
    
    logger.info("🎉 Phi-3 LLM Service 初期化完了")

@app.get("/health")
async def health_check():
    """ヘルスチェック"""
    # Ollama接続確認
    ollama_connected = False
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(f"http://{OLLAMA_HOST}/api/tags", timeout=5.0)
            ollama_connected = response.status_code == 200
    except:
        pass
    
    return {
        "status": "ok",
        "service": "phi3-llm",
        "model": MODEL_NAME,
        "ollama_connected": ollama_connected,
        "conversation_length": len(conversation_history),
        "timestamp": datetime.now().isoformat()
    }

@app.post("/generate")
async def generate_text(request: dict):
    """テキスト生成API"""
    try:
        user_text = request.get("text", "")
        if not user_text:
            raise HTTPException(status_code=400, detail="テキストが必要です")
        
        # 会話履歴から文脈を構築
        context = ""
        for entry in conversation_history[-3:]:  # 直近3回
            context += f"User: {entry['user']}\\nAI: {entry['ai']}\\n"
        
        # Phi-3用のシンプルなプロンプト
        if context.strip():
            prompt = f"""You are a helpful English conversation partner for language learning. Keep responses natural and engaging.

Recent conversation:
{context}

User: {user_text}
Assistant:"""
        else:
            prompt = f"""You are a helpful English conversation partner for language learning. Keep responses natural and engaging.

User: {user_text}
Assistant:"""
        
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"http://{OLLAMA_HOST}/api/generate",
                json={
                    "model": MODEL_NAME,
                    "prompt": prompt,
                    "stream": False,  # ストリーミング一旦無効化
                    "options": {
                        "temperature": 0.3,  # より決定的な応答で高速化
                        "num_predict": 30,   # トークン数削減で高速化
                        "top_p": 0.9,       # より集中した応答
                        "top_k": 20,        # 語彙選択を制限して高速化
                        "num_ctx": 1024     # コンテキスト長を制限
                    }
                },
                timeout=15.0  # タイムアウトを短縮
            )
            
            if response.status_code == 200:
                result = response.json()
                ai_response = result.get("response", "").strip()
                
                # 会話履歴に追加
                conversation_history.append({
                    "user": user_text,
                    "ai": ai_response,
                    "timestamp": datetime.now().isoformat()
                })
                
                # 最後の5回のみ保持
                if len(conversation_history) > 5:
                    conversation_history[:] = conversation_history[-5:]
                
                return {
                    "input": user_text,
                    "response": ai_response if ai_response else "I'm sorry, I didn't understand that. Could you try again?",
                    "conversation_history": conversation_history
                }
            else:
                return {
                    "input": user_text,
                    "response": "I'm having trouble connecting to my language model. Please try again.",
                    "conversation_history": conversation_history
                }
                
    except Exception as e:
        logger.error(f"❌ LLM応答生成エラー: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/reset")
async def reset_conversation():
    """会話履歴をリセット"""
    global conversation_history
    conversation_history = []
    return {"message": "会話履歴をリセットしました"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002)