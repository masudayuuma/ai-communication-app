"""SeamlessM4T v2 Speech-to-Speech engine integration."""

import asyncio
import io
import os
import tempfile
import wave
from typing import Optional, AsyncGenerator, Dict, Any
from pathlib import Path

import torch
import numpy as np
from loguru import logger

try:
    from seamless_communication.models.inference import Translator
    from seamless_communication.models.inference.translator import SequenceGeneratorOutput
    SEAMLESS_AVAILABLE = True
except ImportError:
    logger.warning("seamless_communication not available, using mock implementation")
    SEAMLESS_AVAILABLE = False


class MockTranslator:
    """Mock translator for development when SeamlessM4T is not available."""
    
    def __init__(self, model_name: str = "seamlessM4T_v2_large", device: torch.device = None):
        self.model_name = model_name
        self.device = device or torch.device("cpu")
        logger.info(f"Mock translator initialized with {model_name} on {device}")
    
    def predict(
        self,
        input_data: str,
        task_str: str,
        src_lang: str = "eng",
        tgt_lang: str = "eng"
    ) -> Dict[str, Any]:
        """Mock prediction that returns dummy audio."""
        logger.info(f"Mock prediction: {task_str} from {src_lang} to {tgt_lang}")
        
        # Generate dummy audio (1 second of silence)
        sample_rate = 16000
        duration = 1.0
        dummy_audio = np.zeros(int(sample_rate * duration), dtype=np.float32)
        
        return {
            "wav": dummy_audio,
            "sample_rate": sample_rate,
            "text": "This is a mock response for testing purposes."
        }


class SeamlessM4TEngine:
    """SeamlessM4T v2 Speech-to-Speech processing engine."""
    
    def __init__(
        self,
        model_name: str = "seamlessM4T_v2_large",
        device: Optional[str] = None,
        cache_dir: Optional[str] = None
    ):
        self.model_name = model_name
        self.cache_dir = cache_dir or os.environ.get("MODEL_CACHE_DIR", str(Path.home() / ".cache" / "seamless"))
        
        # Auto-detect device
        if device is None:
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
                logger.info("Using CUDA GPU for inference")
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                self.device = torch.device("mps")
                logger.info("Using Apple Silicon MPS for inference")
            else:
                self.device = torch.device("cpu")
                logger.info("Using CPU for inference")
        else:
            self.device = torch.device(device)
        
        self.translator = None
        self.initialized = False
    
    def initialize(self) -> bool:
        """Initialize the SeamlessM4T model."""
        try:
            logger.info(f"Initializing SeamlessM4T {self.model_name}...")
            
            # Create cache directory
            os.makedirs(self.cache_dir, exist_ok=True)
            
            if SEAMLESS_AVAILABLE:
                # Initialize real SeamlessM4T model
                self.translator = Translator(
                    model_name_or_card=self.model_name,
                    device=self.device,
                    dtype=torch.float16 if self.device.type == "cuda" else torch.float32
                )
                logger.info("SeamlessM4T model loaded successfully")
            else:
                # Use mock translator for development
                self.translator = MockTranslator(self.model_name, self.device)
                logger.warning("Using mock translator - install seamless_communication for real functionality")
            
            self.initialized = True
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize SeamlessM4T: {e}")
            self.initialized = False
            return False
    
    async def speech_to_text(self, audio_bytes: bytes, src_lang: str = "eng") -> str:
        """Convert speech to text using SeamlessM4T ASR."""
        if not self.initialized:
            raise RuntimeError("SeamlessM4T engine not initialized")
        
        try:
            # Save audio to temporary file
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                temp_file.write(audio_bytes)
                temp_path = temp_file.name
            
            try:
                # Run ASR in thread pool to avoid blocking
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(
                    None,
                    self._run_asr,
                    temp_path,
                    src_lang
                )
                
                return result.get("text", "").strip()
                
            finally:
                # Clean up temporary file
                try:
                    os.unlink(temp_path)
                except OSError:
                    pass
                    
        except Exception as e:
            logger.error(f"Speech-to-text error: {e}")
            return ""
    
    def _run_asr(self, audio_path: str, src_lang: str) -> Dict[str, Any]:
        """Run ASR inference."""
        if SEAMLESS_AVAILABLE:
            return self.translator.predict(
                input=audio_path,
                task_str="ASR",
                src_lang=src_lang
            )
        else:
            # Mock ASR response
            return {
                "text": "This is mock speech recognition text for testing."
            }
    
    async def text_to_speech(
        self,
        text: str,
        tgt_lang: str = "eng",
        speaker_id: int = 0
    ) -> AsyncGenerator[bytes, None]:
        """Convert text to speech using SeamlessM4T TTS."""
        if not self.initialized:
            raise RuntimeError("SeamlessM4T engine not initialized")
        
        if not text.strip():
            return
        
        try:
            # Run TTS in thread pool
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,
                self._run_tts,
                text,
                tgt_lang,
                speaker_id
            )
            
            # Convert audio array to WAV bytes
            if "wav" in result:
                audio_data = result["wav"]
                sample_rate = result.get("sample_rate", 16000)
                
                # Convert to 16-bit PCM
                if audio_data.dtype != np.int16:
                    # Normalize and convert to int16
                    audio_data = (audio_data * 32767).astype(np.int16)
                
                # Create WAV bytes
                wav_bytes = self._array_to_wav_bytes(audio_data, sample_rate)
                yield wav_bytes
                
        except Exception as e:
            logger.error(f"Text-to-speech error: {e}")
            return
    
    def _run_tts(self, text: str, tgt_lang: str, speaker_id: int) -> Dict[str, Any]:
        """Run TTS inference."""
        if SEAMLESS_AVAILABLE:
            return self.translator.predict(
                input=text,
                task_str="T2ST",
                tgt_lang=tgt_lang,
                speaker_id=speaker_id
            )
        else:
            # Mock TTS response
            return self.translator.predict(text, "T2ST", tgt_lang=tgt_lang)
    
    async def speech_to_speech(
        self,
        audio_bytes: bytes,
        src_lang: str = "eng",
        tgt_lang: str = "eng",
        speaker_id: int = 0
    ) -> AsyncGenerator[bytes, None]:
        """Direct speech-to-speech translation."""
        if not self.initialized:
            raise RuntimeError("SeamlessM4T engine not initialized")
        
        try:
            # Save audio to temporary file
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                temp_file.write(audio_bytes)
                temp_path = temp_file.name
            
            try:
                # Run S2ST in thread pool
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(
                    None,
                    self._run_s2st,
                    temp_path,
                    src_lang,
                    tgt_lang,
                    speaker_id
                )
                
                # Convert audio array to WAV bytes
                if "wav" in result:
                    audio_data = result["wav"]
                    sample_rate = result.get("sample_rate", 16000)
                    
                    # Convert to 16-bit PCM
                    if audio_data.dtype != np.int16:
                        audio_data = (audio_data * 32767).astype(np.int16)
                    
                    # Create WAV bytes
                    wav_bytes = self._array_to_wav_bytes(audio_data, sample_rate)
                    yield wav_bytes
                    
            finally:
                # Clean up temporary file
                try:
                    os.unlink(temp_path)
                except OSError:
                    pass
                    
        except Exception as e:
            logger.error(f"Speech-to-speech error: {e}")
            return
    
    def _run_s2st(
        self,
        audio_path: str,
        src_lang: str,
        tgt_lang: str,
        speaker_id: int
    ) -> Dict[str, Any]:
        """Run S2ST inference."""
        if SEAMLESS_AVAILABLE:
            return self.translator.predict(
                input=audio_path,
                task_str="S2ST",
                src_lang=src_lang,
                tgt_lang=tgt_lang,
                speaker_id=speaker_id
            )
        else:
            # Mock S2ST response
            return self.translator.predict(audio_path, "S2ST", src_lang, tgt_lang)
    
    def _array_to_wav_bytes(self, audio_data: np.ndarray, sample_rate: int) -> bytes:
        """Convert numpy array to WAV bytes."""
        buffer = io.BytesIO()
        with wave.open(buffer, 'wb') as wav_file:
            wav_file.setnchannels(1)  # Mono
            wav_file.setsampwidth(2)  # 2 bytes for int16
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(audio_data.tobytes())
        return buffer.getvalue()
    
    def get_supported_languages(self) -> Dict[str, str]:
        """Get supported languages."""
        # Common SeamlessM4T v2 supported languages
        return {
            "eng": "English",
            "spa": "Spanish",
            "fra": "French",
            "deu": "German",
            "ita": "Italian",
            "por": "Portuguese",
            "rus": "Russian",
            "jpn": "Japanese",
            "kor": "Korean",
            "cmn": "Chinese (Mandarin)",
            "arb": "Arabic",
            "hin": "Hindi"
        }
    
    def cleanup(self) -> None:
        """Clean up resources."""
        if self.translator:
            del self.translator
            self.translator = None
        self.initialized = False
        logger.info("SeamlessM4T engine cleaned up")


class S2SManager:
    """High-level Speech-to-Speech manager."""
    
    def __init__(
        self,
        model_name: str = "seamlessM4T_v2_large",
        device: Optional[str] = None,
        cache_dir: Optional[str] = None
    ):
        self.engine = SeamlessM4TEngine(model_name, device, cache_dir)
        self.default_language = "eng"
        self.default_speaker = 0
    
    def initialize(self) -> bool:
        """Initialize the S2S engine."""
        return self.engine.initialize()
    
    async def process_speech_for_conversation(
        self,
        audio_bytes: bytes,
        response_text: str
    ) -> AsyncGenerator[bytes, None]:
        """Process speech input and generate TTS response."""
        try:
            # First, convert speech to text for logging/debugging
            if audio_bytes:
                user_text = await self.engine.speech_to_text(
                    audio_bytes,
                    self.default_language
                )
                logger.info(f"User speech: {user_text}")
            
            # Convert response text to speech
            if response_text.strip():
                async for audio_chunk in self.engine.text_to_speech(
                    response_text,
                    self.default_language,
                    self.default_speaker
                ):
                    yield audio_chunk
                    
        except Exception as e:
            logger.error(f"S2S conversation processing error: {e}")
    
    async def transcribe_audio(self, audio_bytes: bytes) -> str:
        """Transcribe audio to text."""
        return await self.engine.speech_to_text(audio_bytes, self.default_language)
    
    async def synthesize_speech(self, text: str) -> AsyncGenerator[bytes, None]:
        """Synthesize speech from text."""
        async for audio_chunk in self.engine.text_to_speech(
            text,
            self.default_language,
            self.default_speaker
        ):
            yield audio_chunk
    
    def set_language(self, language_code: str) -> bool:
        """Set the default language."""
        supported = self.engine.get_supported_languages()
        if language_code in supported:
            self.default_language = language_code
            logger.info(f"Language set to: {supported[language_code]}")
            return True
        return False
    
    def set_speaker(self, speaker_id: int) -> None:
        """Set the default speaker ID."""
        self.default_speaker = speaker_id
        logger.info(f"Speaker set to: {speaker_id}")
    
    def get_supported_languages(self) -> Dict[str, str]:
        """Get supported languages."""
        return self.engine.get_supported_languages()
    
    def cleanup(self) -> None:
        """Clean up resources."""
        self.engine.cleanup()


# TODO: Production improvements
# 1. Add voice cloning and custom speaker training
# 2. Implement streaming TTS for reduced latency
# 3. Add voice activity detection for better segmentation
# 4. Support for different audio formats and quality levels
# 5. Implement batch processing for multiple requests
# 6. Add pronunciation and accent control
# 7. Support for emotion and style transfer in TTS
# 8. Implement model quantization for faster inference
# 9. Add support for real-time voice conversion
# 10. Implement adaptive sampling rate based on content