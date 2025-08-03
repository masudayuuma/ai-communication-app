"""FastAPI backend for AI Communication App."""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
import speech_recognition as sr
import pyttsx3
import requests
import io
import tempfile
import os
import threading
import time
from typing import Optional, List
import base64
import asyncio
from datetime import datetime
from pydub import AudioSegment
from pydub.utils import which

app = FastAPI(title="AI Communication API", version="1.0.0")

# CORS設定
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000", "http://localhost:3002", "http://127.0.0.1:3002"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydanticモデル
class TextInput(BaseModel):
    text: str

class ChatResponse(BaseModel):
    success: bool
    message: str
    response: Optional[str] = None
    timestamp: str

class TTSRequest(BaseModel):
    text: str

class SystemStatus(BaseModel):
    ollama_connected: bool
    microphone_available: bool
    speaker_available: bool
    available_voices: int
    available_microphones: int

# グローバル変数
conversation_history = []
tts_engine = None

def init_tts_engine():
    """TTS engine を初期化"""
    global tts_engine
    try:
        tts_engine = pyttsx3.init()
        voices = tts_engine.getProperty('voices')
        if voices:
            for voice in voices:
                if 'english' in voice.name.lower() or 'en_' in voice.id.lower():
                    tts_engine.setProperty('voice', voice.id)
                    break
        tts_engine.setProperty('rate', 180)
        tts_engine.setProperty('volume', 0.9)
        return True
    except Exception as e:
        print(f"TTS初期化エラー: {e}")
        return False

def test_ollama_connection():
    """Ollama API接続テスト"""
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        return response.status_code == 200, response.json() if response.status_code == 200 else None
    except Exception as e:
        return False, str(e)

def generate_ollama_response(prompt: str):
    """Ollama APIで応答生成"""
    try:
        data = {
            "model": "llama3:8b",
            "prompt": prompt,
            "stream": False
        }
        response = requests.post("http://localhost:11434/api/generate", json=data, timeout=30)
        if response.status_code == 200:
            return True, response.json().get("response", "")
        else:
            return False, f"HTTP {response.status_code}"
    except Exception as e:
        return False, str(e)

def convert_audio_to_wav(input_path: str, output_path: str):
    """音声ファイルをWAV形式に変換"""
    try:
        # 様々な形式の音声ファイルを読み込み
        audio = AudioSegment.from_file(input_path)
        
        # WAV形式で出力 (16kHz, モノラル, 16bit)
        audio = audio.set_frame_rate(16000).set_channels(1)
        audio.export(output_path, format="wav")
        return True
    except Exception as e:
        print(f"音声変換エラー: {e}")
        # フォールバック: ファイルをそのままコピー
        try:
            import shutil
            shutil.copy2(input_path, output_path)
            print(f"フォールバック: ファイルを直接コピーしました")
            return True
        except Exception as copy_error:
            print(f"ファイルコピーエラー: {copy_error}")
            return False

@app.on_event("startup")
async def startup_event():
    """アプリ起動時の初期化"""
    init_tts_engine()
    print("🚀 FastAPI サーバーが起動しました")

@app.get("/")
async def root():
    """ルートエンドポイント"""
    return {"message": "AI Communication API", "status": "running"}

@app.get("/health")
async def health_check():
    """ヘルスチェック"""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

@app.get("/system/status", response_model=SystemStatus)
async def get_system_status():
    """システム状態確認"""
    # Ollama接続確認
    ollama_connected, _ = test_ollama_connection()
    
    # マイク確認
    microphone_available = False
    mic_count = 0
    try:
        mic_list = sr.Microphone.list_microphone_names()
        mic_count = len(mic_list)
        microphone_available = mic_count > 0
    except:
        pass
    
    # スピーカー・音声確認
    speaker_available = tts_engine is not None
    voice_count = 0
    try:
        if tts_engine:
            voices = tts_engine.getProperty('voices')
            voice_count = len(voices) if voices else 0
    except:
        pass
    
    return SystemStatus(
        ollama_connected=ollama_connected,
        microphone_available=microphone_available,
        speaker_available=speaker_available,
        available_voices=voice_count,
        available_microphones=mic_count
    )

@app.post("/chat/text", response_model=ChatResponse)
async def chat_with_text(input_data: TextInput):
    """テキストでAIと会話"""
    try:
        success, response = generate_ollama_response(input_data.text)
        
        if success:
            # 会話履歴に追加
            conversation_entry = {
                "user": input_data.text,
                "ai": response,
                "timestamp": datetime.now().isoformat()
            }
            conversation_history.append(conversation_entry)
            
            return ChatResponse(
                success=True,
                message="応答生成成功",
                response=response,
                timestamp=datetime.now().isoformat()
            )
        else:
            return ChatResponse(
                success=False,
                message=f"AI応答エラー: {response}",
                timestamp=datetime.now().isoformat()
            )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"内部エラー: {str(e)}")

@app.post("/speech/recognize")
async def recognize_speech(audio_file: UploadFile = File(...)):
    """音声認識"""
    try:
        # ファイル拡張子を判定
        file_extension = ".webm"  # デフォルト
        if audio_file.filename:
            if audio_file.filename.endswith('.mp4'):
                file_extension = ".mp4"
            elif audio_file.filename.endswith('.ogg'):
                file_extension = ".ogg"
            elif audio_file.filename.endswith('.wav'):
                file_extension = ".wav"
        
        # アップロードされた音声ファイルを一時保存
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as temp_file:
            content = await audio_file.read()
            temp_file.write(content)
            temp_path = temp_file.name
        
        # WAV形式に変換
        wav_path = temp_path.replace(file_extension, '.wav')
        
        try:
            # WAVファイルの場合は変換をスキップ
            if file_extension == '.wav':
                wav_path = temp_path
                print(f"WAVファイルを直接使用: {wav_path}")
            else:
                # 他の形式の場合はWAV形式に変換
                if not convert_audio_to_wav(temp_path, wav_path):
                    return {
                        "success": False,
                        "text": "",
                        "message": "音声ファイルの変換に失敗しました"
                    }
            
            # 音声認識実行
            r = sr.Recognizer()
            with sr.AudioFile(wav_path) as source:
                audio = r.record(source)
                text = r.recognize_google(audio, language="en-US")
                
                return {
                    "success": True,
                    "text": text,
                    "message": "音声認識成功"
                }
        
        except sr.UnknownValueError:
            return {
                "success": False,
                "text": "",
                "message": "音声を認識できませんでした"
            }
        except sr.RequestError as e:
            return {
                "success": False,
                "text": "",
                "message": f"音声認識サービスエラー: {e}"
            }
        finally:
            # WAVファイルを直接使用した場合は重複削除を避ける
            paths_to_delete = [temp_path] if file_extension == '.wav' else [temp_path, wav_path]
            for path in paths_to_delete:
                if os.path.exists(path):
                    os.unlink(path)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"音声認識エラー: {str(e)}")

@app.post("/speech/synthesize")
async def synthesize_speech(tts_request: TTSRequest):
    """音声合成"""
    try:
        if not tts_engine:
            raise HTTPException(status_code=500, detail="TTS engine が初期化されていません")
        
        # 一時ファイルに音声を保存
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
            temp_path = temp_file.name
        
        # 音声合成実行
        tts_engine.save_to_file(tts_request.text, temp_path)
        tts_engine.runAndWait()
        
        # ファイルが作成されるまで少し待つ
        await asyncio.sleep(0.5)
        
        if os.path.exists(temp_path):
            return FileResponse(
                temp_path,
                media_type="audio/wav",
                filename="speech.wav",
                background=lambda: os.unlink(temp_path) if os.path.exists(temp_path) else None
            )
        else:
            raise HTTPException(status_code=500, detail="音声ファイルの生成に失敗しました")
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"音声合成エラー: {str(e)}")

@app.post("/speech/chat")
async def speech_chat(audio_file: UploadFile = File(...)):
    """音声入力→AI応答→音声出力の一括処理"""
    try:
        print(f"音声ファイル受信: filename={audio_file.filename}, content_type={audio_file.content_type}")
        
        # ファイル拡張子を判定
        file_extension = ".webm"  # デフォルト
        if audio_file.filename:
            if audio_file.filename.endswith('.mp4'):
                file_extension = ".mp4"
            elif audio_file.filename.endswith('.ogg'):
                file_extension = ".ogg"
            elif audio_file.filename.endswith('.wav'):
                file_extension = ".wav"
        
        print(f"使用する拡張子: {file_extension}")
        
        # 1. 音声認識
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as temp_file:
            content = await audio_file.read()
            temp_file.write(content)
            temp_path = temp_file.name
        
        print(f"一時ファイル作成: {temp_path}, サイズ: {len(content)} bytes")
        
        # WAV形式に変換
        wav_path = temp_path.replace(file_extension, '.wav')
        
        try:
            # WAVファイルの場合は変換をスキップ
            if file_extension == '.wav':
                wav_path = temp_path
                print(f"WAVファイルを直接使用: {wav_path}")
            else:
                # 他の形式の場合はWAV形式に変換
                print(f"音声変換開始: {temp_path} -> {wav_path}")
                if not convert_audio_to_wav(temp_path, wav_path):
                    raise HTTPException(status_code=500, detail="音声ファイルの変換に失敗しました")
                print(f"音声変換完了")
            
            print(f"音声認識開始")
            r = sr.Recognizer()
            with sr.AudioFile(wav_path) as source:
                audio = r.record(source)
                user_text = r.recognize_google(audio, language="en-US")
                print(f"音声認識結果: {user_text}")
        finally:
            # WAVファイルを直接使用した場合は重複削除を避ける
            paths_to_delete = [temp_path] if file_extension == '.wav' else [temp_path, wav_path]
            for path in paths_to_delete:
                if os.path.exists(path):
                    print(f"一時ファイル削除: {path}")
                    os.unlink(path)
        
        # 2. AI応答生成
        success, ai_response = generate_ollama_response(user_text)
        
        if not success:
            raise HTTPException(status_code=500, detail=f"AI応答エラー: {ai_response}")
        
        # 3. 会話履歴に追加
        conversation_entry = {
            "user": user_text,
            "ai": ai_response,
            "timestamp": datetime.now().isoformat()
        }
        conversation_history.append(conversation_entry)
        
        # 4. 音声合成
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
            audio_temp_path = temp_file.name
        
        tts_engine.save_to_file(ai_response, audio_temp_path)
        tts_engine.runAndWait()
        
        await asyncio.sleep(0.5)
        
        return {
            "success": True,
            "user_text": user_text,
            "ai_response": ai_response,
            "audio_url": f"/download/audio/{os.path.basename(audio_temp_path)}",
            "timestamp": datetime.now().isoformat()
        }
    
    except sr.UnknownValueError:
        raise HTTPException(status_code=400, detail="音声を認識できませんでした")
    except sr.RequestError as e:
        raise HTTPException(status_code=500, detail=f"音声認識サービスエラー: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"音声チャットエラー: {str(e)}")

@app.get("/conversation/history")
async def get_conversation_history():
    """会話履歴取得"""
    return {
        "conversations": conversation_history,
        "total": len(conversation_history)
    }

@app.delete("/conversation/history")
async def clear_conversation_history():
    """会話履歴クリア"""
    global conversation_history
    conversation_history = []
    return {"message": "会話履歴をクリアしました"}

@app.get("/test/microphone")
async def test_microphone():
    """マイクテスト"""
    try:
        mic_list = sr.Microphone.list_microphone_names()
        return {
            "success": True,
            "microphones": mic_list,
            "count": len(mic_list)
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }

@app.get("/test/speaker")
async def test_speaker():
    """スピーカーテスト"""
    try:
        test_text = "This is a speaker test. Hello from the AI Communication API."
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
            temp_path = temp_file.name
        
        tts_engine.save_to_file(test_text, temp_path)
        tts_engine.runAndWait()
        
        await asyncio.sleep(0.5)
        
        if os.path.exists(temp_path):
            return FileResponse(
                temp_path,
                media_type="audio/wav",
                filename="speaker_test.wav",
                background=lambda: os.unlink(temp_path) if os.path.exists(temp_path) else None
            )
        else:
            raise HTTPException(status_code=500, detail="テスト音声の生成に失敗しました")
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"スピーカーテストエラー: {str(e)}")

@app.post("/test/upload")
async def test_upload(audio_file: UploadFile = File(...)):
    """音声ファイルアップロードテスト"""
    try:
        content = await audio_file.read()
        return {
            "filename": audio_file.filename,
            "content_type": audio_file.content_type,
            "size": len(content),
            "message": "ファイルアップロード成功"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"アップロードテストエラー: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)