"""Tests for LLM client module."""

import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime

from ai_communication_app.llm_client import ConversationManager, OllamaClient, LLMManager


class TestConversationManager:
    """Test ConversationManager functionality."""
    
    def test_init(self):
        """Test ConversationManager initialization."""
        manager = ConversationManager(max_rounds=3, max_context_length=2000)
        
        assert manager.max_rounds == 3
        assert manager.max_context_length == 2000
        assert manager.conversation_history == []
        assert "helpful AI assistant" in manager.system_prompt
    
    def test_add_message(self):
        """Test adding messages to conversation."""
        manager = ConversationManager()
        
        manager.add_message("user", "Hello")
        
        assert len(manager.conversation_history) == 1
        message = manager.conversation_history[0]
        assert message["role"] == "user"
        assert message["content"] == "Hello"
        assert "timestamp" in message
    
    def test_get_context(self):
        """Test getting conversation context."""
        manager = ConversationManager()
        
        # Add some messages
        manager.add_message("user", "Hello")
        manager.add_message("assistant", "Hi there!")
        
        context = manager.get_context()
        
        # Should have system prompt + user messages
        assert len(context) == 3
        assert context[0]["role"] == "system"
        assert context[1]["role"] == "user"
        assert context[2]["role"] == "assistant"
    
    def test_conversation_trimming(self):
        """Test conversation history trimming."""
        manager = ConversationManager(max_rounds=2)
        
        # Add more messages than max_rounds * 2
        for i in range(6):
            manager.add_message("user", f"Message {i}")
            manager.add_message("assistant", f"Response {i}")
        
        # Should trigger summarization and trimming
        assert len(manager.conversation_history) <= 5  # 1 summary + 4 recent
    
    def test_clear_history(self):
        """Test clearing conversation history."""
        manager = ConversationManager()
        
        manager.add_message("user", "Hello")
        manager.add_message("assistant", "Hi!")
        
        manager.clear_history()
        
        assert len(manager.conversation_history) == 0


class TestOllamaClient:
    """Test OllamaClient functionality."""
    
    def test_init(self):
        """Test OllamaClient initialization."""
        client = OllamaClient(
            base_url="http://localhost:11434",
            model_name="llama3:8b-instruct",
            timeout=30
        )
        
        assert client.base_url == "http://localhost:11434"
        assert client.model_name == "llama3:8b-instruct"
        assert client.timeout == 30
        assert client.conversation_manager is not None
    
    @patch('ai_communication_app.llm_client.requests.post')
    @pytest.mark.asyncio
    async def test_generate_response(self, mock_post):
        """Test generating response from LLM."""
        # Mock streaming response
        mock_response = Mock()
        mock_response.raise_for_status.return_value = None
        mock_response.iter_lines.return_value = [
            b'{"message": {"content": "Hello"}, "done": false}',
            b'{"message": {"content": " there!"}, "done": true}'
        ]
        mock_post.return_value = mock_response
        
        client = OllamaClient()
        
        # Collect response chunks
        response_chunks = []
        async for chunk in client.generate_response("Hi"):
            response_chunks.append(chunk)
        
        # Verify response
        assert len(response_chunks) == 2
        assert response_chunks[0] == "Hello"
        assert response_chunks[1] == " there!"
        
        # Verify conversation was updated
        assert len(client.conversation_manager.conversation_history) == 2
        assert client.conversation_manager.conversation_history[0]["content"] == "Hi"
        assert client.conversation_manager.conversation_history[1]["content"] == "Hello there!"
    
    @patch('ai_communication_app.llm_client.requests.get')
    def test_check_model_availability(self, mock_get):
        """Test checking model availability."""
        mock_response = Mock()
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = {
            "models": [
                {"name": "llama3:8b-instruct"},
                {"name": "codellama:7b"}
            ]
        }
        mock_get.return_value = mock_response
        
        client = OllamaClient(model_name="llama3:8b-instruct")
        
        assert client.check_model_availability() is True
        
        # Test unavailable model
        client.model_name = "nonexistent:model"
        assert client.check_model_availability() is False
    
    @patch('ai_communication_app.llm_client.requests.post')
    def test_pull_model(self, mock_post):
        """Test pulling model."""
        mock_response = Mock()
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response
        
        client = OllamaClient()
        
        result = client.pull_model()
        
        assert result is True
        mock_post.assert_called_once()
    
    def test_set_model(self):
        """Test setting model."""
        client = OllamaClient()
        original_model = client.model_name
        
        client.set_model("new:model")
        
        assert client.model_name == "new:model"
        assert client.model_name != original_model
    
    def test_clear_conversation(self):
        """Test clearing conversation."""
        client = OllamaClient()
        client.conversation_manager.add_message("user", "Hello")
        
        client.clear_conversation()
        
        assert len(client.conversation_manager.conversation_history) == 0
    
    def test_get_conversation_history(self):
        """Test getting conversation history."""
        client = OllamaClient()
        client.conversation_manager.add_message("user", "Hello")
        client.conversation_manager.add_message("assistant", "Hi!")
        
        history = client.get_conversation_history()
        
        assert len(history) == 2
        assert history[0]["content"] == "Hello"
        assert history[1]["content"] == "Hi!"
        
        # Verify it's a copy (modification doesn't affect original)
        history.clear()
        assert len(client.conversation_manager.conversation_history) == 2


class TestLLMManager:
    """Test LLMManager functionality."""
    
    def test_init(self):
        """Test LLMManager initialization."""
        manager = LLMManager(
            ollama_url="http://localhost:11434",
            default_model="llama3:8b-instruct"
        )
        
        assert manager.ollama_client is not None
        assert manager.ollama_client.base_url == "http://localhost:11434"
        assert manager.ollama_client.model_name == "llama3:8b-instruct"
        assert len(manager.available_models) > 0
    
    @pytest.mark.asyncio
    async def test_process_speech_input_empty(self):
        """Test processing empty speech input."""
        manager = LLMManager()
        
        response_chunks = []
        async for chunk in manager.process_speech_input(""):
            response_chunks.append(chunk)
        
        assert len(response_chunks) == 1
        assert "didn't catch that" in response_chunks[0].lower()
    
    @patch.object(OllamaClient, 'generate_response')
    @pytest.mark.asyncio
    async def test_process_speech_input(self, mock_generate):
        """Test processing speech input."""
        # Mock async generator
        async def mock_response():
            yield "Hello"
            yield " there!"
        
        mock_generate.return_value = mock_response()
        
        manager = LLMManager()
        
        response_chunks = []
        async for chunk in manager.process_speech_input("Hi"):
            response_chunks.append(chunk)
        
        assert len(response_chunks) == 2
        assert response_chunks[0] == "Hello"
        assert response_chunks[1] == " there!"
    
    @patch.object(OllamaClient, 'check_model_availability')
    def test_initialize_success(self, mock_check):
        """Test successful initialization."""
        mock_check.return_value = True
        
        manager = LLMManager()
        result = manager.initialize()
        
        assert result is True
    
    @patch.object(OllamaClient, 'check_model_availability')
    @patch.object(OllamaClient, 'pull_model')
    def test_initialize_with_pull(self, mock_pull, mock_check):
        """Test initialization with model pulling."""
        mock_check.return_value = False
        mock_pull.return_value = True
        
        manager = LLMManager()
        result = manager.initialize()
        
        assert result is True
        mock_pull.assert_called_once()
    
    @patch.object(OllamaClient, 'check_model_availability')
    @patch.object(OllamaClient, 'pull_model')
    def test_initialize_failure(self, mock_pull, mock_check):
        """Test initialization failure."""
        mock_check.return_value = False
        mock_pull.return_value = False
        
        manager = LLMManager()
        result = manager.initialize()
        
        assert result is False
    
    def test_get_available_models(self):
        """Test getting available models."""
        manager = LLMManager()
        models = manager.get_available_models()
        
        assert isinstance(models, list)
        assert len(models) > 0
        assert "llama3:8b-instruct" in models
    
    def test_switch_model_success(self):
        """Test successful model switching."""
        manager = LLMManager()
        
        result = manager.switch_model("llama3:8b-instruct")
        
        assert result is True
        assert manager.ollama_client.model_name == "llama3:8b-instruct"
    
    def test_switch_model_failure(self):
        """Test failed model switching."""
        manager = LLMManager()
        
        result = manager.switch_model("nonexistent:model")
        
        assert result is False


# Integration tests
@pytest.mark.integration
class TestLLMIntegration:
    """Integration tests for LLM components."""
    
    @pytest.mark.asyncio
    async def test_conversation_flow(self):
        """Test complete conversation flow."""
        manager = ConversationManager()
        
        # Simulate conversation
        manager.add_message("user", "What is Python?")
        manager.add_message("assistant", "Python is a programming language.")
        manager.add_message("user", "Tell me more about it.")
        
        context = manager.get_context()
        
        # Should have system + 3 messages
        assert len(context) == 4
        assert context[0]["role"] == "system"
        assert context[-1]["content"] == "Tell me more about it."


if __name__ == "__main__":
    pytest.main([__file__])