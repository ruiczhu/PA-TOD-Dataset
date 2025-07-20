"""
Transformation Pipeline for Personality-driven Dialogue Transformation

This module implements a streamlined pipeline that:
1. Uses DialogueLoader to load SGD dialogue data
2. Uses PersonalityAdder to add personality labels to dialogues
3. Uses PromptGenerator to generate transformation prompts
4. Uses LLMCaller to get LLM transformation results
5. Uses LLMOutputProcessor to populate transformed_utterance fields
"""

import json
import logging
import os
from datetime import datetime
from typing import List, Dict, Any, Optional

try:
    from .dialogue_loader import DialogueLoader
    from .personality_adder import PersonalityAdder
    from .prompt_generator import PromptGenerator
    from .llm_caller import LLMCaller
    from .LLM_output_processor import LLMOutputProcessor
except ImportError:
    from dialogue_loader import DialogueLoader
    from personality_adder import PersonalityAdder
    from prompt_generator import PromptGenerator
    from llm_caller import LLMCaller
    from LLM_output_processor import LLMOutputProcessor


class TransformationPipeline:
    """
    Streamlined pipeline for personality-driven dialogue transformation
    
    This class orchestrates the core transformation process from loading
    dialogue data to generating transformed dialogues with populated
    transformed_utterance fields.
    """
    
    def __init__(self, 
                 data_dir: str = "datasets/sgd_processed_train",
                 output_dir: str = "output"):
        """
        Initialize the transformation pipeline
        
        Args:
            data_dir: Directory containing SGD dialogue data
            output_dir: Directory for output files
        """
        self.data_dir = data_dir
        self.output_dir = output_dir
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
        # Initialize core components
        self.dialogue_loader = DialogueLoader(data_dir=data_dir, load_all=False)
        self.personality_adder = PersonalityAdder()
        self.prompt_generator = PromptGenerator()
        self.llm_caller = LLMCaller()
        self.output_processor = LLMOutputProcessor()
        
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)
        
        self.logger.info("TransformationPipeline initialized")

    def transform_dialogues(self, 
                          dialogue_count: int = 3,
                          random_seed: Optional[int] = 42,
                          temperature: float = 0.7,
                          max_tokens: int = 4000) -> List[Dict[str, Any]]:
        """
        Transform dialogues with personality-driven modifications
        
        Args:
            dialogue_count: Number of dialogues to process
            random_seed: Random seed for reproducible results
            temperature: LLM generation temperature
            max_tokens: Maximum tokens for LLM generation
            
        Returns:
            List of transformed dialogues with populated transformed_utterance fields
        """
        self.logger.info(f"Starting transformation of {dialogue_count} dialogues")
        
        # Load dialogues
        dialogues = self.dialogue_loader.get_random_dialogues(
            count=dialogue_count, 
            random_seed=random_seed
        )
        
        if not dialogues:
            raise ValueError("No dialogues loaded")
        
        # Process each dialogue
        transformed_dialogues = []
        
        for i, dialogue in enumerate(dialogues):
            dialogue_id = dialogue.get('dialogue_id', f'dialogue_{i}')
            self.logger.info(f"Processing {i+1}/{len(dialogues)}: {dialogue_id}")
            
            try:
                # Transform single dialogue
                transformed_dialogue = self._transform_single_dialogue(
                    dialogue=dialogue,
                    temperature=temperature,
                    max_tokens=max_tokens
                )
                transformed_dialogues.append(transformed_dialogue)
                
            except Exception as e:
                self.logger.error(f"Failed to process dialogue {dialogue_id}: {str(e)}")
                # Add fallback dialogue
                fallback_dialogue = self.output_processor._create_fallback_dialogue(dialogue)
                transformed_dialogues.append(fallback_dialogue)
        
        # Save results
        self._save_transformed_dialogues(transformed_dialogues)
        
        self.logger.info(f"Transformation completed. Processed {len(transformed_dialogues)} dialogues")
        return transformed_dialogues

    def _transform_single_dialogue(self, 
                                 dialogue: Dict[str, Any],
                                 temperature: float,
                                 max_tokens: int) -> Dict[str, Any]:
        """
        Transform a single dialogue through the pipeline
        
        Args:
            dialogue: Original dialogue data
            temperature: LLM generation temperature
            max_tokens: Maximum tokens for generation
            
        Returns:
            Transformed dialogue with populated transformed_utterance fields
        """
        # Add personality labels
        dialogue_with_personality = self.personality_adder.add_random_personality(dialogue)
        
        # Generate transformation prompt
        prompt = self.prompt_generator.generate_transformation_prompt(dialogue_with_personality)
        
        # Call LLM for transformation
        llm_result = self.llm_caller.transform_dialogue(
            prompt=prompt,
            temperature=temperature,
            max_tokens=max_tokens
        )
        
        # Process LLM response to populate transformed_utterance fields
        if llm_result.get('success', False):
            transformed_dialogue = self.output_processor.process_llm_response(
                dialogue_with_personality, 
                llm_result.get('content', '')
            )
        else:
            # If LLM call failed, create fallback dialogue
            transformed_dialogue = self.output_processor._create_fallback_dialogue(dialogue_with_personality)
        
        return transformed_dialogue

    def _save_transformed_dialogues(self, transformed_dialogues: List[Dict[str, Any]]):
        """Save transformed dialogues to JSON file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"transformed_dialogues_{timestamp}.json"
        filepath = os.path.join(self.output_dir, filename)
        
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(transformed_dialogues, f, ensure_ascii=False, indent=2)
            self.logger.info(f"Saved {len(transformed_dialogues)} transformed dialogues to {filepath}")
        except Exception as e:
            self.logger.error(f"Failed to save results to {filepath}: {str(e)}")

    def get_available_services(self) -> List[str]:
        """Get list of available services in the dataset"""
        return self.dialogue_loader.get_all_services()
