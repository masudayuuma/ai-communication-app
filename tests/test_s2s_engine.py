"""Tests for SeamlessM4T speech-to-speech engine."""

import pytest
import numpy as np
import tempfile
import os
from unittest.mock import Mock, patch, MagicMock
import io
import wave

from ai_communication_app.s2s_engine import MockTranslator, SeamlessM4TEngine, S2SManager


class TestMockTranslator:
    """Test MockTranslator functionality."""
    
    def test_init(self):
        """Test MockTranslator initialization."""
        translator = MockTranslator(model_name="seamlessM4T_v2_large")
        
        assert translator.model_name == "seamlessM4T_v2_large"
        assert translator.device is not None
    
    def test_predict(self):
        """Test mock prediction."""
        translator = MockTranslator()
        
        result = translator.predict(
            input_data="test input",
            task_str="ASR",
            src_lang="eng",
            tgt_lang="eng"
        )
        
        assert "wav" in result
        assert "sample_rate" in result
        assert "text" in result
        assert isinstance(result["wav"], np.ndarray)
        assert result["sample_rate"] == 16000
        assert isinstance(result["text"], str)


class TestSeamlessM4TEngine:
    """Test SeamlessM4TEngine functionality."""
    
    def test_init(self):
        """Test SeamlessM4TEngine initialization."""
        engine = SeamlessM4TEngine(
            model_name="seamlessM4T_v2_large",
            device="cpu"
        )
        
        assert engine.model_name == "seamlessM4T_v2_large"
        assert engine.device.type == "cpu"
        assert not engine.initialized
        assert engine.translator is None
    
    @patch('ai_communication_app.s2s_engine.SEAMLESS_AVAILABLE', False)
    def test_initialize_mock(self):
        """Test initialization with mock translator."""
        engine = SeamlessM4TEngine()
        
        result = engine.initialize()
        
        assert result is True
        assert engine.initialized is True
        assert isinstance(engine.translator, MockTranslator)
    
    def test_device_auto_detection(self):
        """Test automatic device detection."""
        with patch('torch.cuda.is_available', return_value=True):
            engine = SeamlessM4TEngine()
            assert engine.device.type == "cuda"
        
        with patch('torch.cuda.is_available', return_value=False):
            with patch('torch.backends.mps.is_available', return_value=True):
                engine = SeamlessM4TEngine()
                assert engine.device.type == "mps"
        
        with patch('torch.cuda.is_available', return_value=False):
            with patch('torch.backends.mps.is_available', return_value=False):
                engine = SeamlessM4TEngine()
                assert engine.device.type == "cpu"
    
    @pytest.mark.asyncio
    async def test_speech_to_text_not_initialized(self):
        """Test speech-to-text with uninitialized engine."""
        engine = SeamlessM4TEngine()
        
        with pytest.raises(RuntimeError, match="not initialized"):
            await engine.speech_to_text(b"fake audio data")
    
    @pytest.mark.asyncio
    async def test_speech_to_text(self):
        """Test speech-to-text conversion."""
        engine = SeamlessM4TEngine()
        engine.initialize()
        
        # Create fake WAV file
        wav_data = create_mock_wav_bytes()
        
        result = await engine.speech_to_text(wav_data, "eng")
        
        assert isinstance(result, str)
        # Mock translator returns specific text
        assert "mock speech recognition text" in result.lower()
    
    @pytest.mark.asyncio
    async def test_text_to_speech_not_initialized(self):
        """Test text-to-speech with uninitialized engine."""
        engine = SeamlessM4TEngine()
        
        with pytest.raises(RuntimeError, match="not initialized"):
            async for _ in engine.text_to_speech("Hello"):
                pass
    
    @pytest.mark.asyncio
    async def test_text_to_speech(self):
        """Test text-to-speech conversion."""
        engine = SeamlessM4TEngine()
        engine.initialize()
        
        audio_chunks = []
        async for chunk in engine.text_to_speech("Hello world", "eng", 0):
            audio_chunks.append(chunk)
        
        assert len(audio_chunks) > 0
        for chunk in audio_chunks:
            assert isinstance(chunk, bytes)
            # Verify it's valid WAV data
            with wave.open(io.BytesIO(chunk), 'rb') as wav:
                assert wav.getnchannels() == 1
                assert wav.getsampwidth() == 2
    
    @pytest.mark.asyncio
    async def test_text_to_speech_empty(self):
        """Test text-to-speech with empty text."""
        engine = SeamlessM4TEngine()
        engine.initialize()
        
        audio_chunks = []
        async for chunk in engine.text_to_speech("", "eng", 0):
            audio_chunks.append(chunk)
        
        assert len(audio_chunks) == 0
    
    @pytest.mark.asyncio
    async def test_speech_to_speech(self):
        """Test direct speech-to-speech conversion."""
        engine = SeamlessM4TEngine()
        engine.initialize()
        
        wav_data = create_mock_wav_bytes()
        
        audio_chunks = []
        async for chunk in engine.speech_to_speech(wav_data, "eng", "eng", 0):
            audio_chunks.append(chunk)
        
        assert len(audio_chunks) > 0
        for chunk in audio_chunks:
            assert isinstance(chunk, bytes)
    
    def test_array_to_wav_bytes(self):
        """Test converting numpy array to WAV bytes."""
        engine = SeamlessM4TEngine()
        
        # Create test audio data
        audio_data = np.array([100, 200, 300, 400], dtype=np.int16)
        sample_rate = 16000
        
        wav_bytes = engine._array_to_wav_bytes(audio_data, sample_rate)
        
        assert isinstance(wav_bytes, bytes)
        
        # Verify WAV format
        with wave.open(io.BytesIO(wav_bytes), 'rb') as wav_file:
            assert wav_file.getnchannels() == 1
            assert wav_file.getsampwidth() == 2
            assert wav_file.getframerate() == sample_rate
            
            # Read back and verify data
            frames = wav_file.readframes(wav_file.getnframes())
            read_data = np.frombuffer(frames, dtype=np.int16)
            np.testing.assert_array_equal(audio_data, read_data)
    
    def test_get_supported_languages(self):
        """Test getting supported languages."""
        engine = SeamlessM4TEngine()
        
        languages = engine.get_supported_languages()
        
        assert isinstance(languages, dict)
        assert "eng" in languages
        assert "spa" in languages
        assert "fra" in languages
        assert languages["eng"] == "English"
    
    def test_cleanup(self):
        """Test engine cleanup."""
        engine = SeamlessM4TEngine()
        engine.initialize()
        
        assert engine.translator is not None
        assert engine.initialized is True
        
        engine.cleanup()
        
        assert engine.translator is None
        assert engine.initialized is False


class TestS2SManager:
    """Test S2SManager functionality."""
    
    def test_init(self):
        """Test S2SManager initialization."""
        manager = S2SManager(
            model_name="seamlessM4T_v2_large",
            device="cpu"
        )
        
        assert manager.engine is not None
        assert manager.default_language == "eng"
        assert manager.default_speaker == 0
    
    def test_initialize(self):
        """Test manager initialization."""
        manager = S2SManager()
        
        result = manager.initialize()
        
        assert result is True
    
    @pytest.mark.asyncio
    async def test_process_speech_for_conversation(self):
        """Test processing speech for conversation."""
        manager = S2SManager()
        manager.initialize()
        
        wav_data = create_mock_wav_bytes()
        response_text = "Hello, how are you?"
        
        audio_chunks = []
        async for chunk in manager.process_speech_for_conversation(wav_data, response_text):
            audio_chunks.append(chunk)
        
        assert len(audio_chunks) > 0
        for chunk in audio_chunks:
            assert isinstance(chunk, bytes)
    
    @pytest.mark.asyncio
    async def test_transcribe_audio(self):
        """Test audio transcription."""
        manager = S2SManager()
        manager.initialize()
        
        wav_data = create_mock_wav_bytes()
        
        result = await manager.transcribe_audio(wav_data)
        
        assert isinstance(result, str)
        assert len(result) > 0
    
    @pytest.mark.asyncio
    async def test_synthesize_speech(self):
        """Test speech synthesis."""
        manager = S2SManager()
        manager.initialize()
        
        audio_chunks = []
        async for chunk in manager.synthesize_speech("Hello world"):
            audio_chunks.append(chunk)
        
        assert len(audio_chunks) > 0
        for chunk in audio_chunks:
            assert isinstance(chunk, bytes)
    
    def test_set_language(self):
        """Test setting language."""
        manager = S2SManager()
        
        # Test valid language
        result = manager.set_language("spa")
        assert result is True
        assert manager.default_language == "spa"
        
        # Test invalid language
        result = manager.set_language("invalid")
        assert result is False
        assert manager.default_language == "spa"  # Should remain unchanged
    
    def test_set_speaker(self):
        """Test setting speaker."""
        manager = S2SManager()
        
        manager.set_speaker(5)
        assert manager.default_speaker == 5
    
    def test_get_supported_languages(self):
        """Test getting supported languages."""
        manager = S2SManager()
        
        languages = manager.get_supported_languages()
        
        assert isinstance(languages, dict)
        assert "eng" in languages
        assert "spa" in languages
    
    def test_cleanup(self):
        """Test manager cleanup."""
        manager = S2SManager()
        manager.initialize()
        
        manager.cleanup()
        
        # Engine should be cleaned up
        assert not manager.engine.initialized


def create_mock_wav_bytes(duration: float = 0.1, sample_rate: int = 16000) -> bytes:
    """Create mock WAV audio bytes for testing."""
    # Generate sine wave
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    audio_data = np.sin(2 * np.pi * 440 * t)  # 440 Hz tone
    
    # Convert to int16
    audio_data = (audio_data * 32767).astype(np.int16)
    
    # Create WAV bytes
    buffer = io.BytesIO()
    with wave.open(buffer, 'wb') as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(audio_data.tobytes())
    
    return buffer.getvalue()


# Integration tests
@pytest.mark.integration
class TestS2SIntegration:
    """Integration tests for S2S components."""
    
    @pytest.mark.asyncio
    async def test_full_pipeline(self):
        """Test complete S2S pipeline."""
        manager = S2SManager()
        assert manager.initialize() is True
        
        # Test transcription
        wav_data = create_mock_wav_bytes()
        transcription = await manager.transcribe_audio(wav_data)
        assert isinstance(transcription, str)
        
        # Test synthesis
        audio_chunks = []
        async for chunk in manager.synthesize_speech("Test response"):
            audio_chunks.append(chunk)
        
        assert len(audio_chunks) > 0
        
        # Cleanup
        manager.cleanup()


if __name__ == "__main__":
    pytest.main([__file__])