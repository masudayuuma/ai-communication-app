"""Streamlit UI for AI Communication App."""

import asyncio
import time
from typing import Optional, Dict, Any
import threading
from datetime import datetime

import streamlit as st
from loguru import logger

from .audio_io import AudioManager
from .llm_client import LLMManager
from .s2s_engine import S2SManager


class ConversationUI:
    """Streamlit UI for conversation management."""
    
    def __init__(self):
        self.audio_manager = None
        self.llm_manager = None
        self.s2s_manager = None
        self.conversation_active = False
        self.processing_audio = False
        
    def initialize_components(self) -> bool:
        """Initialize all components."""
        try:
            # Initialize managers
            if 'audio_manager' not in st.session_state:
                st.session_state.audio_manager = AudioManager()
                
            if 'llm_manager' not in st.session_state:
                st.session_state.llm_manager = LLMManager()
                
            if 's2s_manager' not in st.session_state:
                st.session_state.s2s_manager = S2SManager()
            
            # Initialize session state variables
            if 'conversation_log' not in st.session_state:
                st.session_state.conversation_log = []
                
            if 'is_listening' not in st.session_state:
                st.session_state.is_listening = False
                
            if 'current_model' not in st.session_state:
                st.session_state.current_model = "llama3:8b-instruct"
                
            if 'system_status' not in st.session_state:
                st.session_state.system_status = "Not initialized"
            
            self.audio_manager = st.session_state.audio_manager
            self.llm_manager = st.session_state.llm_manager
            self.s2s_manager = st.session_state.s2s_manager
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize components: {e}")
            st.error(f"Initialization failed: {e}")
            return False
    
    def render_header(self) -> None:
        """Render application header."""
        st.title("🎤 AI English Conversation")
        st.subheader("Practice English with SeamlessM4T v2 Speech-to-Speech")
        
        # Status indicator
        status_color = "🟢" if st.session_state.system_status == "Ready" else "🔴"
        st.markdown(f"**Status:** {status_color} {st.session_state.system_status}")
    
    def render_controls(self) -> None:
        """Render control panel."""
        st.sidebar.header("🎛️ Controls")
        
        # Initialize system
        if st.sidebar.button("🚀 Initialize System", use_container_width=True):
            self.initialize_system()
        
        # Microphone control
        col1, col2 = st.sidebar.columns(2)
        
        with col1:
            if st.button("🎤 Start", use_container_width=True, disabled=st.session_state.is_listening):
                self.start_listening()
        
        with col2:
            if st.button("⏹️ Stop", use_container_width=True, disabled=not st.session_state.is_listening):
                self.stop_listening()
        
        # Model selection
        st.sidebar.subheader("🤖 Model Settings")
        available_models = self.llm_manager.get_available_models() if self.llm_manager else []
        
        if available_models:
            selected_model = st.sidebar.selectbox(
                "LLM Model",
                available_models,
                index=available_models.index(st.session_state.current_model) 
                if st.session_state.current_model in available_models else 0
            )
            
            if selected_model != st.session_state.current_model:
                st.session_state.current_model = selected_model
                if self.llm_manager:
                    self.llm_manager.switch_model(selected_model)
                    st.sidebar.success(f"Switched to {selected_model}")
        
        # Language settings
        st.sidebar.subheader("🌍 Language Settings")
        if self.s2s_manager:
            supported_langs = self.s2s_manager.get_supported_languages()
            lang_options = list(supported_langs.keys())
            lang_labels = [f"{code} - {name}" for code, name in supported_langs.items()]
            
            selected_lang_idx = st.sidebar.selectbox(
                "Speech Language",
                range(len(lang_options)),
                format_func=lambda x: lang_labels[x],
                index=0  # Default to English
            )
            
            selected_lang = lang_options[selected_lang_idx]
            if self.s2s_manager.default_language != selected_lang:
                self.s2s_manager.set_language(selected_lang)
                st.sidebar.success(f"Language set to {supported_langs[selected_lang]}")
        
        # Audio device info
        if st.sidebar.button("📱 Audio Devices"):
            self.show_audio_devices()
        
        # Clear conversation
        if st.sidebar.button("🗑️ Clear Conversation", use_container_width=True):
            self.clear_conversation()
    
    def render_conversation(self) -> None:
        """Render conversation log."""
        st.header("💬 Conversation")
        
        # Listening indicator
        if st.session_state.is_listening:
            st.info("🎤 Listening... Speak into your microphone")
        
        # Conversation display
        if st.session_state.conversation_log:
            for i, entry in enumerate(st.session_state.conversation_log):
                timestamp = entry.get('timestamp', '')
                role = entry.get('role', 'unknown')
                content = entry.get('content', '')
                
                if role == 'user':
                    st.markdown(f"**🗣️ You** ({timestamp})")
                    st.markdown(f"> {content}")
                elif role == 'assistant':
                    st.markdown(f"**🤖 AI Assistant** ({timestamp})")
                    st.markdown(f"> {content}")
                
                st.divider()
        else:
            st.info("Start a conversation by clicking 'Start' and speaking into your microphone!")
    
    def render_debug_info(self) -> None:
        """Render debug information."""
        with st.expander("🔧 Debug Information", expanded=False):
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("System Status")
                st.json({
                    "listening": st.session_state.is_listening,
                    "model": st.session_state.current_model,
                    "conversation_length": len(st.session_state.conversation_log),
                    "last_update": datetime.now().isoformat()
                })
            
            with col2:
                st.subheader("Performance")
                if hasattr(self, 'last_processing_time'):
                    st.metric("Last Processing Time", f"{self.last_processing_time:.2f}s")
                
                if hasattr(self, 'total_conversations'):
                    st.metric("Total Conversations", self.total_conversations)
    
    def initialize_system(self) -> None:
        """Initialize all system components."""
        with st.spinner("Initializing system components..."):
            try:
                # Initialize LLM
                st.session_state.system_status = "Initializing LLM..."
                if not self.llm_manager.initialize():
                    st.error("Failed to initialize LLM manager")
                    return
                
                # Initialize S2S
                st.session_state.system_status = "Initializing SeamlessM4T..."
                if not self.s2s_manager.initialize():
                    st.error("Failed to initialize S2S manager")
                    return
                
                # Set up audio callback
                self.audio_manager.set_audio_callback(self.process_audio_chunk)
                
                st.session_state.system_status = "Ready"
                st.success("System initialized successfully!")
                
            except Exception as e:
                logger.error(f"System initialization failed: {e}")
                st.error(f"System initialization failed: {e}")
                st.session_state.system_status = "Error"
    
    def start_listening(self) -> None:
        """Start audio listening."""
        try:
            if st.session_state.system_status != "Ready":
                st.warning("Please initialize the system first")
                return
            
            self.audio_manager.start_listening()
            st.session_state.is_listening = True
            st.success("Started listening...")
            
        except Exception as e:
            logger.error(f"Failed to start listening: {e}")
            st.error(f"Failed to start listening: {e}")
    
    def stop_listening(self) -> None:
        """Stop audio listening."""
        try:
            self.audio_manager.stop_listening()
            st.session_state.is_listening = False
            st.info("Stopped listening")
            
        except Exception as e:
            logger.error(f"Failed to stop listening: {e}")
            st.error(f"Failed to stop listening: {e}")
    
    def process_audio_chunk(self, audio_bytes: bytes) -> None:
        """Process incoming audio chunk."""
        if self.processing_audio:
            return  # Skip if already processing
        
        self.processing_audio = True
        
        # Run async processing in thread
        def run_async_processing():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                loop.run_until_complete(self._async_process_audio(audio_bytes))
            finally:
                loop.close()
                self.processing_audio = False
        
        thread = threading.Thread(target=run_async_processing, daemon=True)
        thread.start()
    
    async def _async_process_audio(self, audio_bytes: bytes) -> None:
        """Async audio processing pipeline."""
        try:
            start_time = time.time()
            
            # 1. Speech to text
            user_text = await self.s2s_manager.transcribe_audio(audio_bytes)
            
            if not user_text.strip():
                return
            
            # Add user message to conversation
            user_entry = {
                'role': 'user',
                'content': user_text,
                'timestamp': datetime.now().strftime("%H:%M:%S")
            }
            st.session_state.conversation_log.append(user_entry)
            
            # 2. Generate LLM response
            response_text = ""
            async for chunk in self.llm_manager.process_speech_input(user_text):
                response_text += chunk
            
            if response_text.strip():
                # Add AI response to conversation
                ai_entry = {
                    'role': 'assistant',
                    'content': response_text.strip(),
                    'timestamp': datetime.now().strftime("%H:%M:%S")
                }
                st.session_state.conversation_log.append(ai_entry)
                
                # 3. Text to speech and play
                async def audio_generator():
                    async for audio_chunk in self.s2s_manager.synthesize_speech(response_text):
                        yield audio_chunk
                
                await self.audio_manager.play_response(audio_generator())
            
            self.last_processing_time = time.time() - start_time
            
        except Exception as e:
            logger.error(f"Audio processing error: {e}")
    
    def show_audio_devices(self) -> None:
        """Show available audio devices."""
        devices = self.audio_manager.get_audio_devices()
        
        st.sidebar.subheader("Available Audio Devices")
        
        st.sidebar.markdown("**Input Devices:**")
        for device in devices.get('input_devices', []):
            st.sidebar.text(f"• {device['name']} (ID: {device['id']})")
        
        st.sidebar.markdown("**Output Devices:**")
        for device in devices.get('output_devices', []):
            st.sidebar.text(f"• {device['name']} (ID: {device['id']})")
    
    def clear_conversation(self) -> None:
        """Clear conversation history."""
        st.session_state.conversation_log.clear()
        if self.llm_manager:
            self.llm_manager.ollama_client.clear_conversation()
        st.success("Conversation cleared!")


def main():
    """Main Streamlit application."""
    st.set_page_config(
        page_title="AI Communication App",
        page_icon="🎤",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Apply custom CSS
    st.markdown("""
        <style>
        .main-header {
            text-align: center;
            padding: 1rem 0;
        }
        .conversation-container {
            max-height: 600px;
            overflow-y: auto;
            padding: 1rem;
            border: 1px solid #ddd;
            border-radius: 10px;
            background-color: #f9f9f9;
        }
        .status-indicator {
            display: inline-block;
            width: 10px;
            height: 10px;
            border-radius: 50%;
            margin-right: 5px;
        }
        .status-ready { background-color: #28a745; }
        .status-error { background-color: #dc3545; }
        .status-warning { background-color: #ffc107; }
        </style>
    """, unsafe_allow_html=True)
    
    # Initialize UI
    ui = ConversationUI()
    
    if not ui.initialize_components():
        st.error("Failed to initialize application components")
        return
    
    # Render UI components
    ui.render_header()
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        ui.render_conversation()
    
    with col2:
        ui.render_controls()
    
    # Debug info at bottom
    ui.render_debug_info()
    
    # Auto-refresh for real-time updates
    time.sleep(0.1)
    st.rerun()


if __name__ == "__main__":
    main()


# TODO: UI improvements for production
# 1. Add voice waveform visualization
# 2. Implement conversation export/import
# 3. Add settings for audio quality and processing
# 4. Create custom themes and dark mode
# 5. Add conversation analytics and statistics
# 6. Implement user profiles and preferences
# 7. Add keyboard shortcuts for common actions
# 8. Create mobile-responsive design
# 9. Add conversation search and filtering
# 10. Implement real-time collaboration features