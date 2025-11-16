"""
Model Comparison System for OpenAI Models
Easy switching between different OpenAI models for evaluation
"""

import os
import json
import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from openai import OpenAI

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class ModelConfig:
    """Configuration for an OpenAI model"""
    name: str
    model_id: str
    description: str
    max_tokens: int
    temperature: Optional[float] = None
    is_reasoning_model: bool = False

class ModelComparison:
    """System for comparing different OpenAI models"""
    
    def __init__(self):
        self.openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        
        # Define available models - GPT-5 series prioritized
        self.models = {
            "gpt-5-nano": ModelConfig(
                name="GPT-5 Nano",
                model_id="gpt-5-nano",
                description="Fastest GPT-5 model, good for quick responses",
                max_tokens=4000,
                is_reasoning_model=True
            ),
            "gpt-5-mini": ModelConfig(
                name="GPT-5 Mini", 
                model_id="gpt-5-mini",
                description="Balanced GPT-5 model, good performance/speed ratio",
                max_tokens=6000,
                is_reasoning_model=True
            ),
            "gpt-5": ModelConfig(
                name="GPT-5",
                model_id="gpt-5",
                description="Most capable GPT-5 model, best performance",
                max_tokens=8000,
                is_reasoning_model=True
            ),
            "gpt-4o-mini": ModelConfig(
                name="GPT-4o Mini",
                model_id="gpt-4o-mini",
                description="Fast and reliable GPT-4o model (fallback)",
                max_tokens=4000,
                is_reasoning_model=False
            )
        }
    
    def get_model_config(self, model_name: str) -> ModelConfig:
        """Get configuration for a specific model"""
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found. Available models: {list(self.models.keys())}")
        return self.models[model_name]
    
    def list_available_models(self) -> Dict[str, ModelConfig]:
        """List all available models"""
        return self.models
    
    def create_model_client(self, model_name: str) -> 'ModelClient':
        """Create a model client for a specific model"""
        config = self.get_model_config(model_name)
        return ModelClient(self.openai_client, config)

class ModelClient:
    """Client for a specific OpenAI model"""
    
    def __init__(self, openai_client: OpenAI, config: ModelConfig):
        self.openai_client = openai_client
        self.config = config
        self.model_id = config.model_id
        self.name = config.name
        self.description = config.description
        self.max_tokens = config.max_tokens
        self.is_reasoning_model = config.is_reasoning_model
    
    def __str__(self):
        return f"ModelClient({self.name}, {self.model_id})"
    
    def __repr__(self):
        return f"ModelClient(name='{self.name}', model_id='{self.model_id}', max_tokens={self.max_tokens})"
    
    async def chat_completion(self, messages: List[Dict[str, str]], max_tokens_override: int = None, no_max_tokens: bool = False, **kwargs) -> Dict[str, Any]:
        """Create a chat completion with this model"""
        try:
            # Prepare parameters
            params = {
                "model": self.model_id,
                "messages": messages
            }
            
            # Handle max_tokens
            if no_max_tokens:
                # Don't set max_completion_tokens at all (unlimited)
                pass
            elif max_tokens_override is not None:
                # Use override if provided
                params["max_completion_tokens"] = max_tokens_override
            else:
                # Use model default
                params["max_completion_tokens"] = self.max_tokens
            
            # Add temperature only for non-reasoning models
            if not self.is_reasoning_model and "temperature" in kwargs:
                params["temperature"] = kwargs["temperature"]
            elif not self.is_reasoning_model:
                params["temperature"] = 0.1  # Default temperature
            
            # Add any additional parameters
            for key, value in kwargs.items():
                if key not in ["model", "messages", "max_completion_tokens", "temperature"]:
                    params[key] = value
            
            response = await asyncio.to_thread(
                self.openai_client.chat.completions.create,
                **params
            )
            
            return {
                "content": response.choices[0].message.content,
                "usage": response.usage,
                "model": self.model_id,
                "model_name": self.name
            }
            
        except Exception as e:
            logger.error(f"Error with {self.name}: {e}")
            return {
                "content": f"Error: {str(e)}",
                "usage": None,
                "model": self.model_id,
                "model_name": self.name,
                "error": str(e)
            }
    
    async def structured_completion(self, messages: List[Dict[str, str]], response_format, **kwargs) -> Dict[str, Any]:
        """Create a structured completion with this model"""
        try:
            # Prepare parameters
            params = {
                "model": self.model_id,
                "messages": messages,
                "response_format": response_format,
                "max_completion_tokens": self.max_tokens
            }
            
            # Add temperature only for non-reasoning models
            if not self.is_reasoning_model and "temperature" in kwargs:
                params["temperature"] = kwargs["temperature"]
            elif not self.is_reasoning_model:
                params["temperature"] = 0.1  # Default temperature
            
            # Add any additional parameters
            for key, value in kwargs.items():
                if key not in ["model", "messages", "response_format", "max_completion_tokens", "temperature"]:
                    params[key] = value
            
            response = await asyncio.to_thread(
                self.openai_client.beta.chat.completions.parse,
                **params
            )
            
            return {
                "content": response.choices[0].message.content,
                "parsed": response.choices[0].message.parsed,
                "usage": response.usage,
                "model": self.model_id,
                "model_name": self.name
            }
            
        except Exception as e:
            logger.error(f"Error with {self.name} structured completion: {e}")
            return {
                "content": f"Error: {str(e)}",
                "parsed": None,
                "usage": None,
                "model": self.model_id,
                "model_name": self.name,
                "error": str(e)
            }

# Import asyncio for async operations
import asyncio

# Example usage and testing
async def test_model_comparison():
    """Test the model comparison system"""
    
    comparison = ModelComparison()
    
    # List available models
    print("Available Models:")
    for name, config in comparison.list_available_models().items():
        print(f"  {name}: {config.description}")
    
    # Test different models
    test_messages = [
        {"role": "system", "content": "You are a veterinary medicine expert."},
        {"role": "user", "content": "What are the signs of mastitis in dairy cows?"}
    ]
    
    results = {}
    for model_name in ["gpt-5-nano", "gpt-5-mini", "gpt-5"]:
        print(f"\nTesting {model_name}...")
        
        model_client = comparison.create_model_client(model_name)
        result = await model_client.chat_completion(test_messages)
        
        results[model_name] = {
            "model_name": model_client.name,
            "response_length": len(result["content"]),
            "has_error": "error" in result,
            "usage": result.get("usage")
        }
        
        print(f"  Response length: {len(result['content'])}")
        print(f"  Has error: {'error' in result}")
        if result.get("usage"):
            print(f"  Tokens used: {result['usage'].total_tokens}")
    
    return results

if __name__ == "__main__":
    asyncio.run(test_model_comparison())
