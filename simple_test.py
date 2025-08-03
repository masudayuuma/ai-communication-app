"""Simple Streamlit test without audio dependencies."""

import streamlit as st
import requests
import time
from datetime import datetime

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

def main():
    st.set_page_config(
        page_title="AI Communication Test",
        page_icon="🎤",
        layout="wide"
    )
    
    st.title("🎤 AI Communication App - テスト版")
    st.subheader("基本機能テスト")
    
    # Ollama接続テスト
    st.header("🔗 Ollama API接続テスト")
    if st.button("接続テスト実行"):
        with st.spinner("Ollama APIに接続中..."):
            success, result = test_ollama_connection()
            if success:
                st.success("✅ Ollama API接続成功！")
                st.json(result)
            else:
                st.error(f"❌ 接続失敗: {result}")
    
    # AI対話テスト
    st.header("💬 AI対話テスト")
    user_input = st.text_input("英語で質問してください:", placeholder="Hello, how are you?")
    
    if st.button("AI に質問する") and user_input:
        with st.spinner("AIが考え中..."):
            success, response = test_ollama_generate(user_input)
            if success:
                st.success("✅ AI応答:")
                st.write(response)
                
                # 会話ログに追加
                if 'conversation' not in st.session_state:
                    st.session_state.conversation = []
                
                st.session_state.conversation.append({
                    'user': user_input,
                    'ai': response,
                    'time': datetime.now().strftime("%H:%M:%S")
                })
            else:
                st.error(f"❌ AI応答失敗: {response}")
    
    # 会話履歴表示
    if 'conversation' in st.session_state and st.session_state.conversation:
        st.header("📝 会話履歴")
        for i, conv in enumerate(st.session_state.conversation):
            with st.expander(f"会話 {i+1} ({conv['time']})"):
                st.write(f"**あなた:** {conv['user']}")
                st.write(f"**AI:** {conv['ai']}")
    
    # システム情報
    st.sidebar.header("🛠️ システム情報")
    st.sidebar.write(f"現在時刻: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    if st.sidebar.button("会話履歴をクリア"):
        st.session_state.conversation = []
        st.sidebar.success("履歴をクリアしました")

if __name__ == "__main__":
    main()