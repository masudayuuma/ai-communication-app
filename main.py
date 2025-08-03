"""Main entry point for AI Communication App."""

import os
import sys
import argparse
from pathlib import Path

from loguru import logger


def setup_logging(log_level: str = "INFO", log_file: str = None) -> None:
    """Configure logging with loguru."""
    logger.remove()  # Remove default handler
    
    # Console logging
    logger.add(
        sys.stderr,
        level=log_level,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        colorize=True
    )
    
    # File logging
    if log_file:
        logger.add(
            log_file,
            level="DEBUG",
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
            rotation="10 MB",
            retention="7 days"
        )
    
    logger.info(f"Logging configured - Level: {log_level}")


def setup_environment() -> None:
    """Set up environment variables and paths."""
    # Add src directory to Python path
    src_path = Path(__file__).parent / "src"
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))
    
    # Set default model cache directory
    if not os.environ.get("MODEL_CACHE_DIR"):
        cache_dir = Path.home() / ".cache" / "ai_communication_app"
        os.environ["MODEL_CACHE_DIR"] = str(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
    
    # Set HuggingFace cache directory
    if not os.environ.get("HF_HOME"):
        hf_cache = Path.home() / ".cache" / "huggingface"
        os.environ["HF_HOME"] = str(hf_cache)
        hf_cache.mkdir(parents=True, exist_ok=True)
    
    # Set Torch home for model caching
    if not os.environ.get("TORCH_HOME"):
        torch_cache = Path.home() / ".cache" / "torch"
        os.environ["TORCH_HOME"] = str(torch_cache)
        torch_cache.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Model cache directory: {os.environ['MODEL_CACHE_DIR']}")
    logger.info(f"HuggingFace cache: {os.environ['HF_HOME']}")


def check_dependencies() -> bool:
    """Check if required dependencies are available."""
    try:
        import torch
        logger.info(f"PyTorch version: {torch.__version__}")
        logger.info(f"CUDA available: {torch.cuda.is_available()}")
        
        if torch.cuda.is_available():
            logger.info(f"CUDA version: {torch.version.cuda}")
            logger.info(f"GPU count: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                logger.info(f"GPU {i}: {torch.cuda.get_device_name(i)}")
        
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            logger.info("Apple Silicon MPS available")
        
    except ImportError as e:
        logger.error(f"PyTorch not available: {e}")
        return False
    
    try:
        import streamlit
        logger.info(f"Streamlit version: {streamlit.__version__}")
    except ImportError as e:
        logger.error(f"Streamlit not available: {e}")
        return False
    
    try:
        import sounddevice
        logger.info(f"Sounddevice available")
    except ImportError as e:
        logger.error(f"Sounddevice not available: {e}")
        return False
    
    try:
        from seamless_communication import __version__ as seamless_version
        logger.info(f"SeamlessM4T version: {seamless_version}")
    except ImportError:
        logger.warning("SeamlessM4T not available - using mock implementation")
    
    return True


def run_streamlit_app(host: str = "localhost", port: int = 8501, debug: bool = False) -> None:
    """Run the Streamlit application."""
    import subprocess
    
    ui_file = Path(__file__).parent / "src" / "ai_communication_app" / "ui.py"
    
    if not ui_file.exists():
        logger.error(f"UI file not found: {ui_file}")
        sys.exit(1)
    
    # Streamlit command
    cmd = [
        sys.executable, "-m", "streamlit", "run",
        str(ui_file),
        "--server.address", host,
        "--server.port", str(port),
        "--browser.gatherUsageStats", "false"
    ]
    
    if debug:
        cmd.extend(["--logger.level", "debug"])
    
    logger.info(f"Starting Streamlit app on http://{host}:{port}")
    logger.info(f"Command: {' '.join(cmd)}")
    
    try:
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        logger.info("Application stopped by user")
    except subprocess.CalledProcessError as e:
        logger.error(f"Streamlit app failed: {e}")
        sys.exit(1)


def run_cli_mode() -> None:
    """Run in CLI mode for testing components."""
    logger.info("Starting CLI mode for component testing...")
    
    try:
        from ai_communication_app.audio_io import AudioManager
        from ai_communication_app.llm_client import LLMManager
        from ai_communication_app.s2s_engine import S2SManager
        
        # Test audio devices
        audio_manager = AudioManager()
        devices = audio_manager.get_audio_devices()
        logger.info(f"Available audio devices: {len(devices['input_devices'])} input, {len(devices['output_devices'])} output")
        
        # Test LLM connection
        llm_manager = LLMManager()
        if llm_manager.initialize():
            logger.info("LLM manager initialized successfully")
        else:
            logger.warning("LLM manager initialization failed")
        
        # Test S2S engine
        s2s_manager = S2SManager()
        if s2s_manager.initialize():
            logger.info("S2S manager initialized successfully")
        else:
            logger.warning("S2S manager initialization failed")
        
        logger.info("CLI mode testing completed")
        
    except ImportError as e:
        logger.error(f"Import error in CLI mode: {e}")
        logger.error("Make sure all dependencies are installed")
    except Exception as e:
        logger.error(f"CLI mode error: {e}")


def main():
    """Main application entry point."""
    parser = argparse.ArgumentParser(description="AI Communication App - English Conversation Practice")
    parser.add_argument("--host", default="localhost", help="Host address for Streamlit (default: localhost)")
    parser.add_argument("--port", type=int, default=8501, help="Port for Streamlit (default: 8501)")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"], help="Logging level")
    parser.add_argument("--log-file", help="Log file path")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    parser.add_argument("--cli", action="store_true", help="Run in CLI mode for testing")
    parser.add_argument("--check-deps", action="store_true", help="Check dependencies and exit")
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_level, args.log_file)
    
    # Setup environment
    setup_environment()
    
    logger.info("Starting AI Communication App")
    logger.info(f"Python version: {sys.version}")
    logger.info(f"Working directory: {os.getcwd()}")
    
    # Check dependencies
    if not check_dependencies():
        logger.error("Dependency check failed")
        sys.exit(1)
    
    if args.check_deps:
        logger.info("Dependencies check completed successfully")
        return
    
    try:
        if args.cli:
            run_cli_mode()
        else:
            run_streamlit_app(args.host, args.port, args.debug)
            
    except KeyboardInterrupt:
        logger.info("Application interrupted by user")
    except Exception as e:
        logger.error(f"Application error: {e}")
        if args.debug:
            logger.exception("Full traceback:")
        sys.exit(1)


if __name__ == "__main__":
    main()


# TODO: Production enhancements
# 1. Add configuration file support (YAML/JSON)
# 2. Implement proper daemon mode with systemd service
# 3. Add health check endpoints
# 4. Implement graceful shutdown handling
# 5. Add performance monitoring and metrics
# 6. Support for multiple instance deployment
# 7. Add backup and recovery mechanisms
# 8. Implement auto-update functionality
# 9. Add support for custom model loading
# 10. Create installer and packaging scripts