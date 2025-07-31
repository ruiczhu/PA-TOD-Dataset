"""
Data Manager for Multi-Agent Pipeline

This module manages data flow and state coordination across the multi-agent
dialogue processing pipeline, ensuring data consistency and proper handoffs
between agents.
"""

import json
import logging
import time
from typing import Dict, List, Any, Optional
from datetime import datetime

# Import utilities
from multi_agents.utils.data_structures import (
    ProcessingStage, 
    ScenarioInfo, 
    UserProfile, 
    UserState, 
    EvaluationResult, 
    ProcessingMetrics, 
    EnhancedDialogue
)


class DataManager:
    """
    Central data manager for multi-agent pipeline
    
    Manages data flow, state transitions, and coordination between agents
    while maintaining data consistency and processing metrics.
    """
    
    def __init__(self):
        """Initialize data manager"""
        self.logger = logging.getLogger(__name__)
        self.processing_start_time = None
        self.processed_dialogues: List[EnhancedDialogue] = []
        self.current_batch: List[EnhancedDialogue] = []
        
        self.logger.info("DataManager initialized")
    
    def initialize_batch(self, raw_dialogues: List[Dict[str, Any]]) -> List[EnhancedDialogue]:
        """
        Initialize a batch of dialogues for processing
        
        Args:
            raw_dialogues: List of original dialogue data
            
        Returns:
            List of EnhancedDialogue objects ready for processing
        """
        self.processing_start_time = time.time()
        self.current_batch = []
        
        for dialogue_data in raw_dialogues:
            enhanced_dialogue = self._create_enhanced_dialogue(dialogue_data)
            self.current_batch.append(enhanced_dialogue)
        
        self.logger.info(f"Initialized batch with {len(self.current_batch)} dialogues")
        return self.current_batch
    
    def _create_enhanced_dialogue(self, dialogue_data: Dict[str, Any]) -> EnhancedDialogue:
        """
        Create EnhancedDialogue object from raw dialogue data
        
        Args:
            dialogue_data: Original dialogue data
            
        Returns:
            EnhancedDialogue object with initialized fields
        """
        enhanced = EnhancedDialogue()
        enhanced.dialogue_id = dialogue_data.get('dialogue_id', '')
        enhanced.services = dialogue_data.get('services', [])
        enhanced.turns = dialogue_data.get('turns', [])
        enhanced.personality = dialogue_data.get('personality', None)
        
        # Initialize processing metrics
        enhanced.processing_metrics = ProcessingMetrics(
            stage=ProcessingStage.LOADED,
            processing_time=0.0
        )
        
        return enhanced
    
    def update_dialogue_stage(self, 
                            dialogue: EnhancedDialogue, 
                            new_stage: ProcessingStage,
                            agent_name: str = "",
                            notes: str = "",
                            metrics: Dict[str, Any] = None) -> None:
        """
        Update dialogue processing stage and metrics
        
        Args:
            dialogue: Enhanced dialogue to update
            new_stage: New processing stage
            agent_name: Name of agent completing this stage
            notes: Processing notes or observations
            metrics: Additional metrics (token usage, API calls, etc.)
        """
        # Update stage
        dialogue.processing_metrics.stage = new_stage
        
        # Update timing
        if self.processing_start_time:
            dialogue.processing_metrics.processing_time = time.time() - self.processing_start_time
        
        # Update agent notes
        if agent_name and notes:
            dialogue.processing_metrics.agent_notes[agent_name] = notes
        
        # Update metrics
        if metrics:
            if 'token_usage' in metrics:
                for key, value in metrics['token_usage'].items():
                    dialogue.processing_metrics.token_usage[key] = (
                        dialogue.processing_metrics.token_usage.get(key, 0) + value
                    )
            
            if 'api_calls' in metrics:
                dialogue.processing_metrics.api_calls += metrics['api_calls']
        
        self.logger.debug(f"Updated dialogue {dialogue.dialogue_id} to stage {new_stage.value}")
    
    def add_scenario_info(self, dialogue: EnhancedDialogue, scenario_info: ScenarioInfo) -> None:
        """
        Add scenario information to dialogue
        
        Args:
            dialogue: Enhanced dialogue to update
            scenario_info: Generated scenario information
        """
        dialogue.scenario_info = scenario_info
        self.update_dialogue_stage(
            dialogue, 
            ProcessingStage.SCENARIO_ENHANCED,
            agent_name="ScenarioExpert",
            notes=f"Generated scenario: {scenario_info.location} - {scenario_info.time_of_day}"
        )
    
    def add_user_profile(self, dialogue: EnhancedDialogue, user_profile: UserProfile) -> None:
        """
        Add user profile information to dialogue
        
        Args:
            dialogue: Enhanced dialogue to update
            user_profile: Generated user profile
        """
        dialogue.user_profile = user_profile
        self.update_dialogue_stage(
            dialogue,
            ProcessingStage.PROFILE_GENERATED,
            agent_name="ProfileExpert", 
            notes=f"Generated profile: {user_profile.age_range}, {user_profile.occupation}"
        )
    
    def add_user_state(self, dialogue: EnhancedDialogue, user_state: UserState) -> None:
        """
        Add user state information to dialogue
        
        Args:
            dialogue: Enhanced dialogue to update
            user_state: Generated user state
        """
        dialogue.user_state = user_state
        self.update_dialogue_stage(
            dialogue,
            ProcessingStage.STATE_SIMULATED,
            agent_name="SimulationExpert",
            notes=f"Generated state: {user_state.emotional_state}, {user_state.stress_level}"
        )
    
    def add_transformed_dialogue(self, 
                               dialogue: EnhancedDialogue, 
                               transformed_turns: List[Dict[str, Any]]) -> None:
        """
        Add transformed dialogue turns
        
        Args:
            dialogue: Enhanced dialogue to update
            transformed_turns: List of transformed turn data
        """
        dialogue.transformed_turns = transformed_turns
        
        # Also update original turns with transformed_utterance field
        for i, turn in enumerate(dialogue.turns):
            if i < len(transformed_turns):
                turn['transformed_utterance'] = transformed_turns[i].get('transformed', turn.get('utterance', ''))
            else:
                turn['transformed_utterance'] = turn.get('utterance', '')
        
        self.update_dialogue_stage(
            dialogue,
            ProcessingStage.TRANSFORMED,
            agent_name="TransformationExpert",
            notes=f"Transformed {len(transformed_turns)} turns"
        )
    
    def add_evaluation_result(self, 
                            dialogue: EnhancedDialogue, 
                            evaluation: EvaluationResult,
                            is_final: bool = False) -> None:
        """
        Add personality evaluation result
        
        Args:
            dialogue: Enhanced dialogue to update
            evaluation: Evaluation result
            is_final: Whether this is the final evaluation
        """
        if is_final:
            dialogue.final_evaluation = evaluation
            self.update_dialogue_stage(
                dialogue,
                ProcessingStage.FINAL_EVALUATED,
                agent_name="PersonalityEvaluator",
                notes="Final personality evaluation completed"
            )
        else:
            dialogue.initial_evaluation = evaluation
            self.update_dialogue_stage(
                dialogue,
                ProcessingStage.EVALUATED,
                agent_name="PersonalityEvaluator", 
                notes="Initial personality evaluation completed"
            )
    
    def add_optimized_dialogue(self, 
                             dialogue: EnhancedDialogue, 
                             optimized_turns: List[Dict[str, Any]]) -> None:
        """
        Add optimized dialogue turns
        
        Args:
            dialogue: Enhanced dialogue to update
            optimized_turns: List of optimized turn data
        """
        dialogue.optimized_turns = optimized_turns
        self.update_dialogue_stage(
            dialogue,
            ProcessingStage.OPTIMIZED,
            agent_name="OptimizationExpert",
            notes=f"Optimized {len(optimized_turns)} turns"
        )
    
    def mark_dialogue_completed(self, dialogue: EnhancedDialogue, success: bool = True, error_message: str = "") -> None:
        """
        Mark dialogue as completed
        
        Args:
            dialogue: Enhanced dialogue to mark as completed
            success: Whether processing was successful
            error_message: Error message if processing failed
        """
        dialogue.processing_metrics.success = success
        dialogue.processing_metrics.error_message = error_message
        
        self.update_dialogue_stage(
            dialogue,
            ProcessingStage.COMPLETED,
            notes="Dialogue processing completed"
        )
        
        # Add to processed dialogues list
        self.processed_dialogues.append(dialogue)
    
    def get_processing_summary(self) -> Dict[str, Any]:
        """
        Get summary of current processing status
        
        Returns:
            Dictionary with processing summary information
        """
        if not self.current_batch:
            return {"status": "No active batch"}
        
        # Count dialogues by stage
        stage_counts = {}
        total_tokens = 0
        total_api_calls = 0
        errors = 0
        
        for dialogue in self.current_batch:
            stage = dialogue.processing_metrics.stage
            stage_counts[stage.value] = stage_counts.get(stage.value, 0) + 1
            
            # Aggregate metrics
            total_tokens += sum(dialogue.processing_metrics.token_usage.values())
            total_api_calls += dialogue.processing_metrics.api_calls
            
            if not dialogue.processing_metrics.success and dialogue.processing_metrics.error_message:
                errors += 1
        
        # Calculate timing
        elapsed_time = time.time() - self.processing_start_time if self.processing_start_time else 0
        
        return {
            "total_dialogues": len(self.current_batch),
            "stage_distribution": stage_counts,
            "total_tokens_used": total_tokens,
            "total_api_calls": total_api_calls,
            "errors": errors,
            "elapsed_time": elapsed_time,
            "processing_rate": len(self.current_batch) / max(elapsed_time, 1) * 60  # dialogues per minute
        }
    
    def save_batch_results(self, output_path: str) -> None:
        """
        Save processed batch results to file
        
        Args:
            output_path: Path to save results
        """
        try:
            # Convert to serializable format
            results_data = {
                "processing_summary": self.get_processing_summary(),
                "timestamp": datetime.now().isoformat(),
                "dialogues": [dialogue.to_dict() for dialogue in self.current_batch]
            }
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(results_data, f, ensure_ascii=False, indent=2)
            
            self.logger.info(f"Saved batch results to {output_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to save batch results: {str(e)}")
            raise
    
    def get_dialogue_by_id(self, dialogue_id: str) -> Optional[EnhancedDialogue]:
        """
        Get dialogue by ID from current batch
        
        Args:
            dialogue_id: ID of dialogue to retrieve
            
        Returns:
            EnhancedDialogue object if found, None otherwise
        """
        for dialogue in self.current_batch:
            if dialogue.dialogue_id == dialogue_id:
                return dialogue
        return None
    
    def get_dialogues_by_stage(self, stage: ProcessingStage) -> List[EnhancedDialogue]:
        """
        Get all dialogues currently at a specific processing stage
        
        Args:
            stage: Processing stage to filter by
            
        Returns:
            List of dialogues at the specified stage
        """
        return [
            dialogue for dialogue in self.current_batch 
            if dialogue.processing_metrics.stage == stage
        ]
