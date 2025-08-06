# 🎤 AI Communication App

A desktop application for AI English conversation practice using **SeamlessM4T v2 Speech-to-Speech** technology. Practice speaking English with an AI assistant that can understand your speech and respond with natural-sounding audio.

## ✨ Features

- **Real-time Speech Recognition**: Captures microphone input in 0.5-second chunks using SeamlessM4T ASR
- **AI Conversation**: Powered by Llama-3-8B-Instruct via Ollama API for natural English conversation
- **Text-to-Speech**: High-quality voice synthesis using SeamlessM4T TTS
- **Streaming Audio**: Low-latency audio playback starting when 200ms+ of audio is generated
- **Conversation Memory**: Maintains context for 5 rounds with automatic summarization
- **Multi-platform**: Supports Mac, Windows, and Linux with GPU/CPU inference
- **Minimal UI**: Clean Streamlit interface focused on conversation practice

## 🏗️ Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Microphone    │───▶│   SeamlessM4T    │───▶│  Llama-3-8B     │
│     Input       │    │   Speech-to-Text │    │   (Ollama)      │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                                         │
┌─────────────────┐    ┌──────────────────┐             │
│   Speaker       │◀───│   SeamlessM4T    │◀────────────┘
│    Output       │    │   Text-to-Speech │
└─────────────────┘    └──────────────────┘
```

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- Poetry (for dependency management)
- Ollama (for LLM inference)
- PortAudio (for audio I/O)

### 1. Install System Dependencies

**macOS:**
```bash
brew install portaudio
```

**Ubuntu/Debian:**
```bash
sudo apt-get install portaudio19-dev libasound2-dev libpulse-dev ffmpeg
```

**Windows:**
```bash
# PortAudio comes with the Python sounddevice package
```

### 2. Install Ollama

```bash
# macOS/Linux
curl -fsSL https://ollama.ai/install.sh | sh

# Windows
# Download from https://ollama.ai/download
```

Start Ollama and pull the model:
```bash
ollama serve
ollama pull llama3:8b-instruct
```

### 3. Clone and Setup

```bash
git clone https://github.com/masudayuuma/ai-communication-app.git
cd ai-communication-app

# Install dependencies
poetry install

# Run the application
poetry run python main.py
```

The app will be available at `http://localhost:8501`

### 4. Using the Application

1. Click "🚀 Initialize System" to load all models
2. Click "🎤 Start" to begin listening
3. Speak into your microphone in English
4. The AI will respond with synthesized speech
5. Click "⏹️ Stop" to pause listening

## 🐳 Docker Deployment

### GPU Support (Recommended)

```bash
# Build and run with GPU support
docker-compose --profile gpu up --build

# Or build manually
docker build --build-arg BUILD_TARGET=gpu-base -t ai-communication:gpu .
docker run --gpus all -p 8501:8501 ai-communication:gpu
```

### CPU Only

```bash
# Build and run CPU version
docker-compose --profile cpu up --build

# Or build manually  
docker build --build-arg BUILD_TARGET=cpu-base -t ai-communication:cpu .
docker run -p 8502:8501 ai-communication:cpu
```

### Complete Stack with Monitoring

```bash
# Run with Ollama, monitoring, and database
docker-compose --profile gpu --profile monitoring --profile database up
```

## ⚙️ Configuration

### Environment Variables

```bash
# Model cache directory
export MODEL_CACHE_DIR="/path/to/models"

# Ollama API endpoint
export OLLAMA_BASE_URL="http://localhost:11434"

# Logging level
export LOG_LEVEL="INFO"
```

### Model Selection

Available models in the UI:
- `llama3:8b-instruct` (default, balanced performance)
- `llama3:8b-instruct-q4_0` (quantized, faster)
- `llama3:70b-instruct` (highest quality, requires more RAM)
- `codellama:7b-instruct` (code-focused)
- `mistral:7b-instruct` (alternative model)

## 🧪 Testing

Run the test suite:

```bash
# All tests
poetry run pytest

# Unit tests only
poetry run pytest -m "not integration"

# With coverage
poetry run pytest --cov=src/ai_communication_app

# Specific test file
poetry run pytest tests/test_audio_io.py -v
```

## 📊 Performance Benchmarks

### Hardware Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| RAM | 8GB | 16GB+ |
| GPU | None (CPU) | NVIDIA RTX 3060+ |
| Storage | 10GB | 50GB+ (for models) |
| CPU | 4 cores | 8+ cores |

### Latency Measurements

| Configuration | Speech Recognition | LLM Response | TTS Generation | Total Latency |
|---------------|-------------------|--------------|----------------|---------------|
| CPU Only | ~2-3s | ~5-10s | ~3-5s | ~10-18s |
| GPU (RTX 3080) | ~0.5-1s | ~1-2s | ~1-2s | ~2.5-5s |
| GPU (RTX 4090) | ~0.3-0.5s | ~0.5-1s | ~0.5-1s | ~1.3-2.5s |

*Benchmarks measured on common English phrases (10-20 words)*

### Model Download Sizes

| Model | Size | First Run Download |
|-------|------|-------------------|
| SeamlessM4T v2 Large | ~2.3GB | ~5-10 minutes |
| Llama-3-8B-Instruct | ~4.7GB | ~10-15 minutes |
| Combined First Setup | ~7GB | ~15-25 minutes |

## 🔧 CLI Usage

```bash
# Development mode with debug logging
python main.py --debug --log-level DEBUG

# Custom host/port
python main.py --host 0.0.0.0 --port 8080

# Check dependencies
python main.py --check-deps

# CLI testing mode
python main.py --cli
```

## 🐛 Troubleshooting

### Common Issues

**1. Audio Device Not Found**
```bash
# Check available devices
python -c "import sounddevice; print(sounddevice.query_devices())"

# Linux: Install ALSA/PulseAudio
sudo apt-get install alsa-utils pulseaudio
```

**2. CUDA Out of Memory**
```bash
# Reduce model size or use CPU
export CUDA_VISIBLE_DEVICES=""  # Force CPU
# Or use quantized models: llama3:8b-instruct-q4_0
```

**3. Ollama Connection Failed**
```bash
# Check Ollama status
ollama list
curl http://localhost:11434/api/tags

# Restart Ollama
pkill ollama && ollama serve
```

**4. SeamlessM4T Import Error**
```bash
# Install from source
pip install git+https://github.com/facebookresearch/seamless_communication.git

# Or use mock mode for development (limited functionality)
```

### Known Limitations

- **Audio Quality**: Depends on microphone quality and background noise
- **Language Support**: Currently optimized for English; other languages available but may have reduced quality
- **Real-time Performance**: GPU strongly recommended for responsive conversation
- **Memory Usage**: Large models require significant RAM (8GB+ recommended)
- **Network Dependency**: Initial model downloads require internet connection

## 🛠️ Development

### Project Structure

```
ai-communication-app/
├── src/ai_communication_app/
│   ├── __init__.py
│   ├── audio_io.py          # Audio recording/playback
│   ├── llm_client.py        # Ollama integration
│   ├── s2s_engine.py        # SeamlessM4T wrapper
│   └── ui.py                # Streamlit interface
├── tests/                   # Unit tests
├── main.py                  # Application entry point
├── pyproject.toml          # Poetry configuration
├── Dockerfile              # Container configuration
├── docker-compose.yml      # Multi-service setup
└── README.md               # This file
```

### Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Make changes and add tests
4. Run tests: `poetry run pytest`
5. Submit a pull request

### Adding New Features

**New TTS Voices:**
```python
# In s2s_engine.py
self.s2s_manager.set_speaker(speaker_id)  # 0-7 available
```

**Custom System Prompts:**
```python
# In llm_client.py
conversation_manager.system_prompt = "Your custom prompt here"
```

**Audio Preprocessing:**
```python
# In audio_io.py - add to AudioRecorder
def preprocess_audio(self, audio_data):
    # Add noise reduction, normalization, etc.
    return processed_audio
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Meta AI** for SeamlessM4T v2 model
- **Meta** for Llama-3 language model
- **Ollama** for local LLM inference
- **Streamlit** for the web interface
- **PortAudio** for cross-platform audio I/O

## 📬 Support

- **Issues**: [GitHub Issues](https://github.com/masudayuuma/ai-communication-app/issues)
- **Discussions**: [GitHub Discussions](https://github.com/masudayuuma/ai-communication-app/discussions)
- **Email**: masudayuuma@example.com

---

**⚡ Ready to practice English conversation with AI? Get started in minutes!**







Phi-3 miniにモデルを切り替える
macはdocker経由でgpu使えない。ローカルからしか無理っぽい
Llama3はローカルやといいけどdocker経由やと遅い
baseモデルのwhisperなかなかいい、早い