"""Voice-enabled AI Communication App using SpeechRecognition and pyttsx3."""

import streamlit as st
import speech_recognition as sr
import pyttsx3
import requests
import time
import threading
import queue
from datetime import datetime
import io
import tempfile
import os

# Initialize TTS engine globally
@st.cache_resource
def get_tts_engine():
    """Initialize and configure TTS engine."""
    engine = pyttsx3.init()
    
    # Configure voice settings
    voices = engine.getProperty('voices')
    if voices:
        # Try to find English voice
        for voice in voices:
            if 'english' in voice.name.lower() or 'en_' in voice.id.lower():
                engine.setProperty('voice', voice.id)
                break
    
    # Set speech rate and volume
    engine.setProperty('rate', 180)  # Speed
    engine.setProperty('volume', 0.9)  # Volume
    return engine

def test_ollama_connection():
    """Test Ollama API connection."""
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code == 200:
            return True, response.json()
        else:
            return False, f"HTTP {response.status_code}"
    except Exception as e:
        return False, str(e)

def test_ollama_generate(prompt="Hello"):
    """Test Ollama text generation."""
    try:
        data = {
            "model": "llama3:8b",
            "prompt": prompt,
            "stream": False
        }
        response = requests.post("http://localhost:11434/api/generate", json=data, timeout=30)
        if response.status_code == 200:
            return True, response.json().get("response", "No response")
        else:
            return False, f"HTTP {response.status_code}"
    except Exception as e:
        return False, str(e)

def record_audio():
    """Record audio from microphone."""
    r = sr.Recognizer()
    
    # Use default microphone
    with sr.Microphone() as source:
        st.write("🎤 音声を調整中...")
        r.adjust_for_ambient_noise(source, duration=1)
        st.write("🎤 話してください...")
        
        try:
            # Record audio with timeout
            audio = r.listen(source, timeout=1, phrase_time_limit=5)
            return audio
        except sr.WaitTimeoutError:
            st.warning("音声が検出されませんでした。もう一度お試しください。")
            return None

def recognize_speech(audio):
    """Convert speech to text."""
    r = sr.Recognizer()
    
    try:
        # Use Google Speech Recognition (free tier)
        text = r.recognize_google(audio, language="en-US")
        return True, text
    except sr.UnknownValueError:
        return False, "音声を認識できませんでした"
    except sr.RequestError as e:
        return False, f"音声認識サービスエラー: {e}"

def speak_text(text):
    """Convert text to speech."""
    try:
        engine = get_tts_engine()
        engine.say(text)
        engine.runAndWait()
        return True
    except Exception as e:
        st.error(f"音声合成エラー: {e}")
        return False

def main():
    st.set_page_config(
        page_title="AI Voice Communication",
        page_icon="🎤",
        layout="wide"
    )
    
    st.title("🎤 AI音声英会話アプリ")
    st.subheader("音声で英語のAIと会話しよう！")
    
    # Initialize session state
    if 'conversation' not in st.session_state:
        st.session_state.conversation = []
    if 'is_listening' not in st.session_state:
        st.session_state.is_listening = False
    
    # Sidebar
    st.sidebar.header("🛠️ システム状態")
    
    # Test microphone
    st.sidebar.subheader("🎤 マイクテスト")
    if st.sidebar.button("マイクテスト"):
        r = sr.Recognizer()
        try:
            with sr.Microphone() as source:
                st.sidebar.write("マイクを調整中...")
                r.adjust_for_ambient_noise(source, duration=1)
                st.sidebar.success("✅ マイクが正常に動作しています")
        except Exception as e:
            st.sidebar.error(f"❌ マイクエラー: {e}")
    
    # Test speakers
    st.sidebar.subheader("🔊 スピーカーテスト")
    if st.sidebar.button("スピーカーテスト"):
        test_text = "Hello, this is a speaker test."
        if speak_text(test_text):
            st.sidebar.success("✅ スピーカーが正常に動作しています")
        else:
            st.sidebar.error("❌ スピーカーエラー")
    
    # Main interface
    col1, col2 = st.columns(2)
    
    with col1:
        st.header("🎙️ 音声会話")
        
        if st.button("🎤 音声で話す", use_container_width=True, type="primary"):
            with st.spinner("音声を録音中..."):
                audio = record_audio()
                
                if audio:
                    with st.spinner("音声を認識中..."):
                        success, user_text = recognize_speech(audio)
                        
                        if success:
                            st.success(f"✅ 認識結果: {user_text}")
                            
                            # Get AI response
                            with st.spinner("AIが回答を生成中..."):
                                ai_success, ai_response = test_ollama_generate(user_text)
                                
                                if ai_success:
                                    st.write(f"**AI:** {ai_response}")
                                    
                                    # Convert to speech
                                    with st.spinner("音声を合成中..."):
                                        if speak_text(ai_response):
                                            st.success("✅ 音声再生完了")
                                            
                                            # Add to conversation history
                                            st.session_state.conversation.append({
                                                'user': user_text,
                                                'ai': ai_response,
                                                'time': datetime.now().strftime("%H:%M:%S")
                                            })
                                        else:
                                            st.error("音声再生に失敗しました")
                                else:
                                    st.error(f"AI応答エラー: {ai_response}")
                        else:
                            st.error(f"音声認識エラー: {user_text}")
    
    with col2:
        st.header("💬 テキスト会話")
        
        user_input = st.text_input("英語で入力:", placeholder="Hello, how are you?")
        
        if st.button("テキストで送信", use_container_width=True) and user_input:
            with st.spinner("AIが回答中..."):
                success, response = test_ollama_generate(user_input)
                
                if success:
                    st.write(f"**AI:** {response}")
                    
                    # Option to play AI response
                    if st.button("🔊 AIの回答を音声で聞く"):
                        speak_text(response)
                    
                    # Add to conversation history
                    st.session_state.conversation.append({
                        'user': user_input,
                        'ai': response,
                        'time': datetime.now().strftime("%H:%M:%S")
                    })
                else:
                    st.error(f"AI応答エラー: {response}")
    
    # Conversation history
    if st.session_state.conversation:
        st.header("📝 会話履歴")
        
        for i, conv in enumerate(st.session_state.conversation):
            with st.expander(f"会話 {i+1} ({conv['time']})"):
                st.write(f"**👤 あなた:** {conv['user']}")
                st.write(f"**🤖 AI:** {conv['ai']}")
                
                # Play buttons for each response
                col1, col2, col3 = st.columns(3)
                with col1:
                    if st.button(f"🔊 ユーザー発言再生", key=f"user_{i}"):
                        speak_text(conv['user'])
                with col2:
                    if st.button(f"🔊 AI応答再生", key=f"ai_{i}"):
                        speak_text(conv['ai'])
                with col3:
                    if st.button(f"🗑️ 削除", key=f"del_{i}"):
                        st.session_state.conversation.pop(i)
                        st.rerun()
    
    # System information
    st.sidebar.subheader("📊 システム情報")
    st.sidebar.write(f"会話数: {len(st.session_state.conversation)}")
    st.sidebar.write(f"現在時刻: {datetime.now().strftime('%H:%M:%S')}")
    
    # Clear conversation
    if st.sidebar.button("🗑️ 会話履歴をクリア"):
        st.session_state.conversation = []
        st.sidebar.success("履歴をクリアしました")
        st.rerun()
    
    # Connection test
    st.sidebar.subheader("🔗 接続テスト")
    if st.sidebar.button("Ollama接続確認"):
        success, result = test_ollama_connection()
        if success:
            st.sidebar.success("✅ Ollama接続成功")
        else:
            st.sidebar.error(f"❌ 接続失敗: {result}")
    
    # Instructions
    st.sidebar.subheader("📖 使い方")
    st.sidebar.write("""
    1. **音声会話**: 「🎤 音声で話す」ボタンを押して英語で話す
    2. **テキスト会話**: テキストボックスに英語を入力
    3. **音声再生**: 各回答の「🔊」ボタンで音声再生
    4. **履歴管理**: 会話履歴の確認・削除
    """)

if __name__ == "__main__":
    main()