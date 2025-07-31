"""
Unified LLM Interface for Multi-Agent System

This module provides a unified interface for all LLM API calls across different agents,
based on the existing MPEAF llm_caller.py implementation.
"""

import logging
import requests
import json
import time
from typing import Dict, Any, Optional

try:
    from config.config import API_KEY_GPT_4, API_URL_GPT_40
except ImportError:
    from config.config import API_KEY_GPT_4, API_URL_GPT_40


class LLMInterface:
    """
    Unified LLM interface for multi-agent system
    
    This class provides a consistent API calling interface for all agents,
    based on the existing MPEAF LLMCaller implementation with enhancements
    for multi-agent workflows.
    """
    
    def __init__(self, 
                 api_url: Optional[str] = None,
                 api_key: Optional[str] = None,
                 model_name: str = "gpt-4",
                 default_temperature: float = 0.7,
                 default_max_tokens: int = 3000,
                 timeout: int = 60,
                 max_retries: int = 3):
        """
        Initialize LLM interface
        
        Args:
            api_url: LLM API endpoint URL
            api_key: API authentication key
            model_name: Name of the model to use
            default_temperature: Default generation temperature
            default_max_tokens: Default maximum tokens
            timeout: Request timeout in seconds
            max_retries: Maximum number of retry attempts
        """
        self.api_url = api_url or API_URL_GPT_40
        self.api_key = api_key or API_KEY_GPT_4
        self.model_name = model_name
        self.default_temperature = default_temperature
        self.default_max_tokens = default_max_tokens
        self.timeout = timeout
        self.max_retries = max_retries
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
        # Setup headers
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        self.logger.info(f"LLMInterface initialized with model: {self.model_name}")
    
    def call_agent(self, 
                   agent_name: str,
                   system_prompt: str,
                   user_prompt: str,
                   temperature: Optional[float] = None,
                   max_tokens: Optional[int] = None,
                   top_p: float = 0.95) -> Dict[str, Any]:
        """
        Make an LLM API call for a specific agent
        
        Args:
            agent_name: Name of the calling agent (for logging)
            system_prompt: System prompt defining agent role
            user_prompt: User prompt with specific task
            temperature: Generation temperature (uses default if None)
            max_tokens: Maximum tokens to generate (uses default if None)
            top_p: Top-p sampling parameter
            
        Returns:
            Dictionary containing the LLM response and metadata
            
        Raises:
            Exception: If API call fails after all retries
        """
        # Use default values if not specified
        temperature = temperature if temperature is not None else self.default_temperature
        max_tokens = max_tokens if max_tokens is not None else self.default_max_tokens
        
        # Prepare messages for the API call
        messages = [
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user", 
                "content": user_prompt
            }
        ]
        
        # Prepare payload
        payload = {
            "messages": messages,
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens
        }
        
        # Make the API call with retries
        for attempt in range(self.max_retries):
            try:
                self.logger.info(f"Making LLM API call for {agent_name} (attempt {attempt + 1}/{self.max_retries})")
                
                response = requests.post(
                    self.api_url,
                    headers=self.headers,
                    json=payload,
                    timeout=self.timeout
                )
                
                # Check if request was successful
                response.raise_for_status()
                
                # Parse response
                response_data = response.json()
                
                # Extract the generated content
                if 'choices' in response_data and len(response_data['choices']) > 0:
                    generated_content = response_data['choices'][0]['message']['content']
                    
                    result = {
                        'success': True,
                        'content': generated_content,
                        'usage': response_data.get('usage', {}),
                        'model': response_data.get('model', self.model_name),
                        'temperature': temperature,
                        'max_tokens': max_tokens,
                        'attempt': attempt + 1,
                        'agent_name': agent_name
                    }
                    
                    self.logger.info(f"LLM API call successful for {agent_name} on attempt {attempt + 1}")
                    return result
                else:
                    raise ValueError("Invalid response format: missing choices")
                    
            except requests.exceptions.Timeout:
                self.logger.warning(f"Request timeout for {agent_name} on attempt {attempt + 1}")
                if attempt < self.max_retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
                    continue
                else:
                    raise Exception(f"Request timed out for {agent_name} after all retries")
                    
            except requests.exceptions.RequestException as e:
                self.logger.warning(f"Request error for {agent_name} on attempt {attempt + 1}: {str(e)}")
                if attempt < self.max_retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
                    continue
                else:
                    raise Exception(f"Request failed for {agent_name} after all retries: {str(e)}")
                    
            except json.JSONDecodeError:
                self.logger.warning(f"JSON decode error for {agent_name} on attempt {attempt + 1}")
                if attempt < self.max_retries - 1:
                    time.sleep(2 ** attempt)
                    continue
                else:
                    raise Exception(f"Failed to decode JSON response for {agent_name} after all retries")
                    
            except Exception as e:
                self.logger.error(f"Unexpected error for {agent_name} on attempt {attempt + 1}: {str(e)}")
                if attempt < self.max_retries - 1:
                    time.sleep(2 ** attempt)
                    continue
                else:
                    raise Exception(f"Unexpected error for {agent_name} after all retries: {str(e)}")
        
        # This should never be reached due to the raise statements above
        raise Exception(f"Failed to complete API call for {agent_name}")
    
    def get_api_status(self) -> Dict[str, Any]:
        """
        Check API status and configuration
        
        Returns:
            Dictionary with API status information
        """
        return {
            "api_url": self.api_url,
            "model_name": self.model_name,
            "default_temperature": self.default_temperature,
            "default_max_tokens": self.default_max_tokens,
            "timeout": self.timeout,
            "max_retries": self.max_retries,
            "has_api_key": bool(self.api_key)
        }
    
    def set_api_credentials(self, api_url: str, api_key: str):
        """
        Update API credentials
        
        Args:
            api_url: New API endpoint URL
            api_key: New API key
        """
        self.api_url = api_url
        self.api_key = api_key
        self.headers["Authorization"] = f"Bearer {self.api_key}"
        self.logger.info("API credentials updated")
