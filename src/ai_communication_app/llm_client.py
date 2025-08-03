"""LLM client for Ollama API integration."""

import json
import asyncio
from typing import List, Dict, Optional, AsyncGenerator
from datetime import datetime

import requests
from loguru import logger


class ConversationManager:
    """Manages conversation history and context."""
    
    def __init__(self, max_rounds: int = 5, max_context_length: int = 4000):
        self.max_rounds = max_rounds
        self.max_context_length = max_context_length
        self.conversation_history: List[Dict[str, str]] = []
        self.system_prompt = (
            "You are a helpful AI assistant for English conversation practice. "
            "Respond naturally and engagingly in English. Keep responses concise "
            "but meaningful for conversation flow. Focus on helping the user "
            "practice English speaking skills."
        )
    
    def add_message(self, role: str, content: str) -> None:
        """Add a message to conversation history."""
        message = {
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat()
        }
        self.conversation_history.append(message)
        
        # Trim history if too long
        if len(self.conversation_history) > self.max_rounds * 2:
            self._summarize_and_trim()
    
    def get_context(self) -> List[Dict[str, str]]:
        """Get current conversation context for LLM."""
        messages = [{"role": "system", "content": self.system_prompt}]
        
        # Add recent conversation history
        for msg in self.conversation_history[-self.max_rounds * 2:]:
            messages.append({
                "role": msg["role"],
                "content": msg["content"]
            })
        
        return messages
    
    def _summarize_and_trim(self) -> None:
        """Summarize old conversation and trim history."""
        if len(self.conversation_history) <= self.max_rounds * 2:
            return
            
        # Keep recent messages, summarize older ones
        recent_messages = self.conversation_history[-self.max_rounds:]
        old_messages = self.conversation_history[:-self.max_rounds]
        
        # Create summary of old conversation
        summary_content = "Previous conversation summary:\n"
        for msg in old_messages:
            summary_content += f"{msg['role']}: {msg['content'][:100]}...\n"
        
        # Replace old messages with summary
        self.conversation_history = [
            {"role": "assistant", "content": summary_content, "timestamp": datetime.now().isoformat()}
        ] + recent_messages
        
        logger.info(f"Summarized conversation, kept {len(recent_messages)} recent messages")
    
    def clear_history(self) -> None:
        """Clear conversation history."""
        self.conversation_history.clear()
        logger.info("Cleared conversation history")


class OllamaClient:
    """Client for Ollama API communication."""
    
    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model_name: str = "llama3:8b-instruct",
        timeout: int = 30
    ):
        self.base_url = base_url.rstrip('/')
        self.model_name = model_name
        self.timeout = timeout
        self.conversation_manager = ConversationManager()
    
    async def generate_response(self, user_input: str) -> AsyncGenerator[str, None]:
        """Generate streaming response from LLM."""
        try:
            # Add user message to conversation
            self.conversation_manager.add_message("user", user_input)
            
            # Prepare request
            messages = self.conversation_manager.get_context()
            request_data = {
                "model": self.model_name,
                "messages": messages,
                "stream": True,
                "options": {
                    "temperature": 0.7,
                    "top_p": 0.9,
                    "max_tokens": 150
                }
            }
            
            logger.info(f"Sending request to Ollama: {user_input[:50]}...")
            
            # Make streaming request
            async for chunk in self._stream_request(request_data):
                yield chunk
                
        except Exception as e:
            logger.error(f"LLM generation error: {e}")
            yield "I'm sorry, I'm having trouble responding right now. Please try again."
    
    async def _stream_request(self, request_data: Dict) -> AsyncGenerator[str, None]:
        """Make streaming request to Ollama API."""
        url = f"{self.base_url}/api/chat"
        full_response = ""
        
        try:
            # Use asyncio to run requests in thread pool
            loop = asyncio.get_event_loop()
            
            def make_request():
                return requests.post(
                    url,
                    json=request_data,
                    stream=True,
                    timeout=self.timeout
                )
            
            response = await loop.run_in_executor(None, make_request)
            response.raise_for_status()
            
            # Process streaming response
            for line in response.iter_lines():
                if line:
                    try:
                        chunk_data = json.loads(line.decode('utf-8'))
                        
                        if 'message' in chunk_data and 'content' in chunk_data['message']:
                            content = chunk_data['message']['content']
                            full_response += content
                            yield content
                            
                        if chunk_data.get('done', False):
                            break
                            
                    except json.JSONDecodeError as e:
                        logger.warning(f"Failed to parse JSON chunk: {e}")
                        continue
            
            # Add assistant response to conversation
            if full_response.strip():
                self.conversation_manager.add_message("assistant", full_response.strip())
                logger.info(f"Generated response: {full_response[:50]}...")
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Ollama API request failed: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error in streaming: {e}")
            raise
    
    def check_model_availability(self) -> bool:
        """Check if the specified model is available."""
        try:
            url = f"{self.base_url}/api/tags"
            response = requests.get(url, timeout=5)
            response.raise_for_status()
            
            models = response.json().get('models', [])
            available_models = [model['name'] for model in models]
            
            is_available = self.model_name in available_models
            logger.info(f"Model {self.model_name} available: {is_available}")
            logger.info(f"Available models: {available_models}")
            
            return is_available
            
        except Exception as e:
            logger.error(f"Failed to check model availability: {e}")
            return False
    
    def pull_model(self) -> bool:
        """Pull the model if not available."""
        try:
            url = f"{self.base_url}/api/pull"
            request_data = {"name": self.model_name}
            
            logger.info(f"Pulling model: {self.model_name}")
            response = requests.post(url, json=request_data, timeout=300)
            response.raise_for_status()
            
            logger.info(f"Successfully pulled model: {self.model_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to pull model: {e}")
            return False
    
    def set_model(self, model_name: str) -> None:
        """Change the active model."""
        self.model_name = model_name
        logger.info(f"Changed model to: {model_name}")
    
    def clear_conversation(self) -> None:
        """Clear conversation history."""
        self.conversation_manager.clear_history()
    
    def get_conversation_history(self) -> List[Dict[str, str]]:
        """Get conversation history."""
        return self.conversation_manager.conversation_history.copy()


class LLMManager:
    """High-level LLM management interface."""
    
    def __init__(
        self,
        ollama_url: str = "http://localhost:11434",
        default_model: str = "llama3:8b"
    ):
        self.ollama_client = OllamaClient(ollama_url, default_model)
        self.available_models = [
            "llama3:8b",
            "llama3:latest",
            "llama3:8b-instruct",
            "llama3:8b-instruct-q4_0",
            "llama3:70b-instruct"
        ]
    
    async def process_speech_input(self, user_text: str) -> AsyncGenerator[str, None]:
        """Process speech input and generate text response."""
        if not user_text.strip():
            yield "I didn't catch that. Could you please repeat?"
            return
        
        try:
            logger.info(f"Processing user input: {user_text}")
            async for response_chunk in self.ollama_client.generate_response(user_text):
                yield response_chunk
                
        except Exception as e:
            logger.error(f"Speech processing error: {e}")
            yield "I'm having trouble processing your request. Please try again."
    
    def initialize(self) -> bool:
        """Initialize LLM client and check model availability."""
        try:
            if not self.ollama_client.check_model_availability():
                logger.info("Model not available, attempting to pull...")
                if not self.ollama_client.pull_model():
                    logger.error("Failed to pull model")
                    return False
            
            logger.info("LLM client initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"LLM initialization failed: {e}")
            return False
    
    def get_available_models(self) -> List[str]:
        """Get list of available models."""
        return self.available_models.copy()
    
    def switch_model(self, model_name: str) -> bool:
        """Switch to a different model."""
        if model_name not in self.available_models:
            logger.error(f"Model {model_name} not in available models")
            return False
        
        self.ollama_client.set_model(model_name)
        return True


# TODO: Production improvements
# 1. Add retry mechanism with exponential backoff
# 2. Implement request queuing and rate limiting
# 3. Add model performance monitoring and metrics
# 4. Support for multiple LLM providers (OpenAI, Anthropic, etc.)
# 5. Implement conversation persistence (database storage)
# 6. Add conversation branching and multiple contexts
# 7. Implement smart context window management
# 8. Add conversation analytics and insights
# 9. Support for custom system prompts per conversation
# 10. Add conversation export/import functionality