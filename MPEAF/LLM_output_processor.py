"""LLM Output Processor for Personality-driven Dialogue Transformation

This module implements a processor that takes LLM responses and extracts
the transformed utterances to populate the transformed_utterance fields
in the original dialogue structure.
"""

import json
import logging
import re
from typing import Dict, Any, List, Optional


class LLMOutputProcessor:
    """
    Process LLM responses to extract transformed utterances
    
    This class parses the LLM response JSON and maps the transformed
    utterances back to the original dialogue structure by populating
    the transformed_utterance fields.
    """
    
    def __init__(self):
        """Initialize the LLM output processor"""
        self.logger = logging.getLogger(__name__)
    
    def process_llm_response(self, 
                           original_dialogue: Dict[str, Any], 
                           llm_response: str) -> Dict[str, Any]:
        """
        Process LLM response and populate transformed_utterance fields
        
        Args:
            original_dialogue: Original dialogue with personality info
            llm_response: Raw LLM response string
            
        Returns:
            Dialogue with populated transformed_utterance fields
        """
        try:
            # Parse LLM response JSON
            transformed_data = self._parse_llm_response(llm_response)
            
            # Validate response structure
            self._validate_response_structure(transformed_data, original_dialogue)
            
            # Map transformed utterances to original dialogue
            processed_dialogue = self._map_transformed_utterances(
                original_dialogue, 
                transformed_data
            )
            
            self.logger.info(f"Successfully processed LLM response for dialogue {original_dialogue.get('dialogue_id', 'unknown')}")
            return processed_dialogue
            
        except Exception as e:
            self.logger.error(f"Failed to process LLM response: {str(e)}")
            # Return original dialogue with transformed_utterance fields as fallback
            return self._create_fallback_dialogue(original_dialogue)
    
    def _parse_llm_response(self, llm_response: str) -> Dict[str, Any]:
        """
        Parse LLM response string to extract JSON
        
        Args:
            llm_response: Raw LLM response string
            
        Returns:
            Parsed JSON data
        """
        # Try to extract JSON from response
        json_patterns = [
            r'```json\s*(\{.*?\})\s*```',  # JSON in code blocks
            r'(\{[^{}]*"dialogue_id"[^{}]*\})',  # Simple JSON pattern
            r'(\{.*?\})'  # Any JSON-like structure
        ]
        
        for pattern in json_patterns:
            matches = re.findall(pattern, llm_response, re.DOTALL)
            for match in matches:
                try:
                    return json.loads(match)
                except json.JSONDecodeError:
                    continue
        
        # If no pattern works, try parsing the entire response
        try:
            return json.loads(llm_response.strip())
        except json.JSONDecodeError:
            raise ValueError("Could not extract valid JSON from LLM response")
    
    def _validate_response_structure(self, 
                                   transformed_data: Dict[str, Any], 
                                   original_dialogue: Dict[str, Any]):
        """
        Validate that the transformed data has the expected structure
        
        Args:
            transformed_data: Parsed LLM response
            original_dialogue: Original dialogue for comparison
        """
        # Check required fields
        if 'transformed_turns' not in transformed_data:
            raise ValueError("Missing 'transformed_turns' in LLM response")
        
        transformed_turns = transformed_data['transformed_turns']
        original_turns = original_dialogue.get('turns', [])
        
        if not isinstance(transformed_turns, list):
            raise ValueError("'transformed_turns' must be a list")
        
        if len(transformed_turns) != len(original_turns):
            self.logger.warning(f"Turn count mismatch: original={len(original_turns)}, transformed={len(transformed_turns)}")
        
        # Validate each turn structure
        for i, turn in enumerate(transformed_turns):
            if not isinstance(turn, dict):
                raise ValueError(f"Turn {i} is not a dictionary")
            
            required_fields = ['turn_index', 'speaker', 'transformed']
            for field in required_fields:
                if field not in turn:
                    raise ValueError(f"Turn {i} missing required field: {field}")
    
    def _map_transformed_utterances(self, 
                                   original_dialogue: Dict[str, Any], 
                                   transformed_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Map transformed utterances to original dialogue structure
        
        Args:
            original_dialogue: Original dialogue structure
            transformed_data: Parsed transformation data
            
        Returns:
            Dialogue with populated transformed_utterance fields
        """
        # Create a copy of the original dialogue
        processed_dialogue = json.loads(json.dumps(original_dialogue))
        
        # Get transformed turns
        transformed_turns = transformed_data['transformed_turns']
        original_turns = processed_dialogue.get('turns', [])
        
        # Create mapping by turn index
        transform_map = {}
        for turn in transformed_turns:
            turn_index = turn.get('turn_index')
            transformed_utterance = turn.get('transformed', '')
            
            if turn_index is not None:
                transform_map[turn_index] = transformed_utterance
        
        # Apply transformations to original turns
        for i, turn in enumerate(original_turns):
            if i in transform_map:
                # Add transformed_utterance field
                turn['transformed_utterance'] = transform_map[i]
                self.logger.debug(f"Mapped transformation for turn {i}: {turn.get('speaker', 'unknown')}")
            else:
                # If no transformation found, use original utterance
                turn['transformed_utterance'] = turn.get('utterance', '')
                self.logger.warning(f"No transformation found for turn {i}, using original utterance")
        
        return processed_dialogue
    
    def _create_fallback_dialogue(self, original_dialogue: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create fallback dialogue with transformed_utterance fields using original utterances
        
        Args:
            original_dialogue: Original dialogue structure
            
        Returns:
            Dialogue with transformed_utterance fields filled with original utterances
        """
        fallback_dialogue = json.loads(json.dumps(original_dialogue))
        
        for turn in fallback_dialogue.get('turns', []):
            # Use original utterance as fallback
            turn['transformed_utterance'] = turn.get('utterance', '')
        
        self.logger.warning("Created fallback dialogue with original utterances")
        return fallback_dialogue

