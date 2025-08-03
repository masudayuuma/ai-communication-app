"""Audio input/output module using sounddevice and pydub."""

import asyncio
import io
import threading
import time
from typing import Optional, Callable, Generator
import wave

import numpy as np
import sounddevice as sd
from loguru import logger
from pydub import AudioSegment
from pydub.playback import play


class AudioRecorder:
    """Real-time audio recorder with 0.5s chunking."""
    
    def __init__(
        self,
        sample_rate: int = 16000,
        channels: int = 1,
        chunk_duration: float = 0.5,
        callback: Optional[Callable[[bytes], None]] = None
    ):
        self.sample_rate = sample_rate
        self.channels = channels
        self.chunk_duration = chunk_duration
        self.chunk_size = int(sample_rate * chunk_duration)
        self.callback = callback
        self.recording = False
        self.audio_buffer = []
        
    def start_recording(self) -> None:
        """Start real-time audio recording."""
        if self.recording:
            logger.warning("Recording already started")
            return
            
        self.recording = True
        self.audio_buffer.clear()
        
        def audio_callback(indata, frames, time, status):
            if status:
                logger.warning(f"Audio input status: {status}")
            
            if self.recording:
                # Convert float32 to int16
                audio_data = (indata[:, 0] * 32767).astype(np.int16)
                self.audio_buffer.extend(audio_data)
                
                # Process chunk when buffer is full
                if len(self.audio_buffer) >= self.chunk_size:
                    chunk = np.array(self.audio_buffer[:self.chunk_size], dtype=np.int16)
                    self.audio_buffer = self.audio_buffer[self.chunk_size:]
                    
                    # Convert to bytes and send to callback
                    if self.callback:
                        audio_bytes = self._numpy_to_wav_bytes(chunk)
                        self.callback(audio_bytes)
        
        try:
            self.stream = sd.InputStream(
                samplerate=self.sample_rate,
                channels=self.channels,
                callback=audio_callback,
                blocksize=1024,
                dtype='float32'
            )
            self.stream.start()
            logger.info(f"Started recording at {self.sample_rate}Hz")
            
        except Exception as e:
            logger.error(f"Failed to start recording: {e}")
            self.recording = False
            raise
    
    def stop_recording(self) -> None:
        """Stop audio recording."""
        if not self.recording:
            return
            
        self.recording = False
        if hasattr(self, 'stream'):
            self.stream.stop()
            self.stream.close()
        logger.info("Stopped recording")
    
    def _numpy_to_wav_bytes(self, audio_data: np.ndarray) -> bytes:
        """Convert numpy array to WAV bytes."""
        buffer = io.BytesIO()
        with wave.open(buffer, 'wb') as wav_file:
            wav_file.setnchannels(self.channels)
            wav_file.setsampwidth(2)  # 2 bytes for int16
            wav_file.setframerate(self.sample_rate)
            wav_file.writeframes(audio_data.tobytes())
        return buffer.getvalue()


class AudioPlayer:
    """Audio player with streaming support."""
    
    def __init__(self):
        self.playing = False
        self.play_queue = asyncio.Queue()
        self.play_thread = None
        
    async def play_audio_stream(self, audio_generator: Generator[bytes, None, None]) -> None:
        """Play audio stream with minimal latency."""
        try:
            self.playing = True
            logger.info("Starting audio stream playback")
            
            # Start playback thread
            self.play_thread = threading.Thread(
                target=self._playback_worker,
                daemon=True
            )
            self.play_thread.start()
            
            # Feed audio chunks to queue
            async for audio_chunk in audio_generator:
                if not self.playing:
                    break
                await self.play_queue.put(audio_chunk)
                
            # Signal end of stream
            await self.play_queue.put(None)
            
        except Exception as e:
            logger.error(f"Audio playback error: {e}")
        finally:
            self.playing = False
    
    def _playback_worker(self) -> None:
        """Worker thread for audio playback."""
        try:
            while self.playing:
                try:
                    # Get audio chunk from queue (blocking)
                    audio_chunk = asyncio.run(
                        asyncio.wait_for(self.play_queue.get(), timeout=0.1)
                    )
                    
                    if audio_chunk is None:  # End of stream
                        break
                        
                    # Play audio chunk
                    audio_segment = AudioSegment.from_wav(io.BytesIO(audio_chunk))
                    play(audio_segment)
                    
                except asyncio.TimeoutError:
                    continue
                except Exception as e:
                    logger.error(f"Playback worker error: {e}")
                    break
                    
        except Exception as e:
            logger.error(f"Playback worker failed: {e}")
    
    def stop_playback(self) -> None:
        """Stop audio playback."""
        self.playing = False
        if self.play_thread and self.play_thread.is_alive():
            self.play_thread.join(timeout=1.0)
        logger.info("Stopped audio playback")


class AudioManager:
    """Combined audio recording and playback manager."""
    
    def __init__(self):
        self.recorder = AudioRecorder()
        self.player = AudioPlayer()
        self.audio_callback = None
        
    def set_audio_callback(self, callback: Callable[[bytes], None]) -> None:
        """Set callback for processed audio chunks."""
        self.audio_callback = callback
        self.recorder.callback = callback
    
    def start_listening(self) -> None:
        """Start listening for audio input."""
        self.recorder.start_recording()
    
    def stop_listening(self) -> None:
        """Stop listening for audio input."""
        self.recorder.stop_recording()
    
    async def play_response(self, audio_generator: Generator[bytes, None, None]) -> None:
        """Play AI response audio."""
        await self.player.play_audio_stream(audio_generator)
    
    def stop_playback(self) -> None:
        """Stop current audio playback."""
        self.player.stop_playback()
    
    def get_audio_devices(self) -> dict:
        """Get available audio input/output devices."""
        try:
            devices = sd.query_devices()
            return {
                'input_devices': [
                    {'id': i, 'name': dev['name'], 'channels': dev['max_input_channels']}
                    for i, dev in enumerate(devices) 
                    if dev['max_input_channels'] > 0
                ],
                'output_devices': [
                    {'id': i, 'name': dev['name'], 'channels': dev['max_output_channels']}
                    for i, dev in enumerate(devices) 
                    if dev['max_output_channels'] > 0
                ]
            }
        except Exception as e:
            logger.error(f"Failed to query audio devices: {e}")
            return {'input_devices': [], 'output_devices': []}


# TODO: Improvements for production
# 1. Add VAD (Voice Activity Detection) to filter out silence
# 2. Implement noise reduction/cancellation
# 3. Add audio format conversion utilities
# 4. Support for different audio codecs (MP3, FLAC, etc.)
# 5. Add audio visualization (waveform, spectrum)
# 6. Implement echo cancellation for full-duplex
# 7. Add audio compression for network transmission
# 8. Support for multiple microphone arrays