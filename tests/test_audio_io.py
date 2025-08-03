"""Tests for audio I/O module."""

import pytest
import numpy as np
import io
import wave
from unittest.mock import Mock, patch, MagicMock

from ai_communication_app.audio_io import AudioRecorder, AudioPlayer, AudioManager


class TestAudioRecorder:
    """Test AudioRecorder functionality."""
    
    def test_init(self):
        """Test AudioRecorder initialization."""
        recorder = AudioRecorder(sample_rate=16000, channels=1, chunk_duration=0.5)
        
        assert recorder.sample_rate == 16000
        assert recorder.channels == 1
        assert recorder.chunk_duration == 0.5
        assert recorder.chunk_size == 8000  # 16000 * 0.5
        assert not recorder.recording
        assert recorder.audio_buffer == []
    
    def test_init_with_callback(self):
        """Test AudioRecorder initialization with callback."""
        callback = Mock()
        recorder = AudioRecorder(callback=callback)
        
        assert recorder.callback == callback
    
    @patch('ai_communication_app.audio_io.sd.InputStream')
    def test_start_recording(self, mock_stream_class):
        """Test starting audio recording."""
        mock_stream = Mock()
        mock_stream_class.return_value = mock_stream
        
        recorder = AudioRecorder()
        recorder.start_recording()
        
        assert recorder.recording is True
        mock_stream_class.assert_called_once()
        mock_stream.start.assert_called_once()
    
    def test_stop_recording(self):
        """Test stopping audio recording."""
        recorder = AudioRecorder()
        recorder.recording = True
        
        # Mock stream
        mock_stream = Mock()
        recorder.stream = mock_stream
        
        recorder.stop_recording()
        
        assert recorder.recording is False
        mock_stream.stop.assert_called_once()
        mock_stream.close.assert_called_once()
    
    def test_numpy_to_wav_bytes(self):
        """Test converting numpy array to WAV bytes."""
        recorder = AudioRecorder()
        audio_data = np.array([100, 200, 300, 400], dtype=np.int16)
        
        wav_bytes = recorder._numpy_to_wav_bytes(audio_data)
        
        assert isinstance(wav_bytes, bytes)
        assert len(wav_bytes) > 0
        
        # Verify WAV format
        with wave.open(io.BytesIO(wav_bytes), 'rb') as wav_file:
            assert wav_file.getnchannels() == 1
            assert wav_file.getsampwidth() == 2
            assert wav_file.getframerate() == 16000


class TestAudioPlayer:
    """Test AudioPlayer functionality."""
    
    def test_init(self):
        """Test AudioPlayer initialization."""
        player = AudioPlayer()
        
        assert not player.playing
        assert player.play_queue is not None
        assert player.play_thread is None
    
    def test_stop_playback(self):
        """Test stopping audio playback."""
        player = AudioPlayer()
        player.playing = True
        
        # Mock thread
        mock_thread = Mock()
        mock_thread.is_alive.return_value = True
        player.play_thread = mock_thread
        
        player.stop_playback()
        
        assert not player.playing
        mock_thread.join.assert_called_once_with(timeout=1.0)


class TestAudioManager:
    """Test AudioManager functionality."""
    
    def test_init(self):
        """Test AudioManager initialization."""
        manager = AudioManager()
        
        assert manager.recorder is not None
        assert manager.player is not None
        assert manager.audio_callback is None
    
    def test_set_audio_callback(self):
        """Test setting audio callback."""
        manager = AudioManager()
        callback = Mock()
        
        manager.set_audio_callback(callback)
        
        assert manager.audio_callback == callback
        assert manager.recorder.callback == callback
    
    @patch('ai_communication_app.audio_io.AudioRecorder.start_recording')
    def test_start_listening(self, mock_start):
        """Test starting audio listening."""
        manager = AudioManager()
        manager.start_listening()
        
        mock_start.assert_called_once()
    
    @patch('ai_communication_app.audio_io.AudioRecorder.stop_recording')
    def test_stop_listening(self, mock_stop):
        """Test stopping audio listening."""
        manager = AudioManager()
        manager.stop_listening()
        
        mock_stop.assert_called_once()
    
    @patch('ai_communication_app.audio_io.AudioPlayer.stop_playback')
    def test_stop_playback(self, mock_stop):
        """Test stopping audio playback."""
        manager = AudioManager()
        manager.stop_playback()
        
        mock_stop.assert_called_once()
    
    @patch('ai_communication_app.audio_io.sd.query_devices')
    def test_get_audio_devices(self, mock_query):
        """Test getting audio devices."""
        mock_devices = [
            {'name': 'Input Device 1', 'max_input_channels': 2, 'max_output_channels': 0},
            {'name': 'Output Device 1', 'max_input_channels': 0, 'max_output_channels': 2},
            {'name': 'Both Device', 'max_input_channels': 2, 'max_output_channels': 2}
        ]
        mock_query.return_value = mock_devices
        
        manager = AudioManager()
        devices = manager.get_audio_devices()
        
        assert len(devices['input_devices']) == 2  # Input Device 1 and Both Device
        assert len(devices['output_devices']) == 2  # Output Device 1 and Both Device
        
        # Check input device structure
        input_device = devices['input_devices'][0]
        assert 'id' in input_device
        assert 'name' in input_device
        assert 'channels' in input_device
    
    @patch('ai_communication_app.audio_io.sd.query_devices')
    def test_get_audio_devices_error(self, mock_query):
        """Test getting audio devices with error."""
        mock_query.side_effect = Exception("Device query failed")
        
        manager = AudioManager()
        devices = manager.get_audio_devices()
        
        assert devices == {'input_devices': [], 'output_devices': []}


@pytest.fixture
def sample_audio_data():
    """Generate sample audio data for testing."""
    sample_rate = 16000
    duration = 1.0  # 1 second
    frequency = 440  # A4 note
    
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    audio_data = np.sin(frequency * 2 * np.pi * t)
    
    # Convert to int16
    audio_data = (audio_data * 32767).astype(np.int16)
    
    return audio_data, sample_rate


def test_wav_conversion_roundtrip(sample_audio_data):
    """Test WAV conversion roundtrip."""
    audio_data, sample_rate = sample_audio_data
    
    recorder = AudioRecorder(sample_rate=sample_rate)
    wav_bytes = recorder._numpy_to_wav_bytes(audio_data)
    
    # Read back the WAV data
    with wave.open(io.BytesIO(wav_bytes), 'rb') as wav_file:
        frames = wav_file.readframes(wav_file.getnframes())
        read_audio = np.frombuffer(frames, dtype=np.int16)
    
    # Compare original and read data
    np.testing.assert_array_equal(audio_data, read_audio)


if __name__ == "__main__":
    pytest.main([__file__])