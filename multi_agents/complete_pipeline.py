"""
Complete Multi-Agent Pipeline Implementation

This module implements the complete 18-step workflow for personality-driven
dialogue transformation using the 6-agent system. It coordinates all agents
and manages the data flow through the entire pipeline.
"""

import json
import logging
import os
import time
from datetime import datetime
from typing import Dict, List, Any, Optional

# Import all agents and utilities
from multi_agents.agents.agent_1_scenario_expert import ScenarioExpert
from multi_agents.agents.agent_2_profile_expert import ProfileExpert
from multi_agents.agents.agent_3_simulation_expert import SimulationExpert
from multi_agents.agents.agent_4_transformation_expert import TransformationExpert
from multi_agents.agents.agent_5_personality_evaluator import PersonalityEvaluator
from multi_agents.agents.agent_6_optimization_expert import OptimizationExpert

from multi_agents.utils.llm_interface import LLMInterface
from multi_agents.utils.data_structures import ProcessingStage, EnhancedDialogue
from multi_agents.data_manager import DataManager
from multi_agents.visualization import VisualizationManager

# Import MPEAF components
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from MPEAF.dialogue_loader import DialogueLoader
from MPEAF.personality_framework import PersonalityFramework


class CompletePipeline:
    """
    Complete 18-step multi-agent personality-driven dialogue transformation pipeline
    
    This class orchestrates the complete workflow from original dialogue data
    through scenario analysis, personality profiling, state simulation, dialogue
    transformation, evaluation, and optimization.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the complete pipeline
        
        Args:
            config: Configuration dictionary for pipeline settings
        """
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # Initialize LLM interface (shared across agents)
        self.llm_interface = LLMInterface()
        
        # Initialize all 6 agents
        self.scenario_expert = ScenarioExpert(self.llm_interface)
        self.profile_expert = ProfileExpert(self.llm_interface)
        self.simulation_expert = SimulationExpert(self.llm_interface)
        self.transformation_expert = TransformationExpert(self.llm_interface)
        self.personality_evaluator = PersonalityEvaluator(self.llm_interface)
        self.optimization_expert = OptimizationExpert(self.llm_interface)
        
        # Initialize supporting systems
        self.data_manager = DataManager()
        self.visualization_manager = VisualizationManager()
        
        # Initialize MPEAF components
        self.personality_framework = PersonalityFramework()
        
        # Pipeline statistics
        self.pipeline_stats = {
            'total_processed': 0,
            'successful_transformations': 0,
            'failed_transformations': 0,
            'average_processing_time': 0,
            'start_time': None
        }
        
        self.logger.info("Complete Multi-Agent Pipeline initialized with all 6 agents")
    
    def run_complete_pipeline(self, 
                            input_data_path: str,
                            output_directory: str,
                            batch_size: int = 10,
                            max_iterations: int = 2,
                            target_quality: float = 0.8,
                            max_dialogues: Optional[int] = None) -> Dict[str, Any]:
        """
        Execute the complete 18-step pipeline on input dialogue data
        
        Args:
            input_data_path: Path to input dialogue data
            output_directory: Directory for output files
            batch_size: Number of dialogues to process in each batch
            max_iterations: Maximum optimization iterations per dialogue
            target_quality: Target quality threshold for optimization
            max_dialogues: Maximum number of dialogues to process (None for all)
            
        Returns:
            Pipeline execution results and statistics
        """
        self.logger.info(f"Starting complete 18-step pipeline execution on {input_data_path}")
        self.pipeline_stats['start_time'] = time.time()
        
        try:
            # Step 1-2: Load and prepare data
            # Use max_dialogues as the loading limit if specified
            dialogue_data = self._load_and_prepare_data(input_data_path, max_dialogues)
            
            # Step 3: Initialize batch processing
            enhanced_dialogues = self.data_manager.initialize_batch(dialogue_data)
            
            # Limit the number of dialogues if specified
            if max_dialogues is not None and len(enhanced_dialogues) > max_dialogues:
                enhanced_dialogues = enhanced_dialogues[:max_dialogues]
                self.logger.info(f"Limited processing to {max_dialogues} dialogues")
            
            # Create batches manually
            batches = []
            for i in range(0, len(enhanced_dialogues), batch_size):
                batch = enhanced_dialogues[i:i + batch_size]
                batches.append(batch)
            
            # Step 4-18: Process each dialogue through the complete pipeline
            results = []
            
            for batch_idx, batch in enumerate(batches):
                self.logger.info(f"Processing batch {batch_idx + 1}")
                
                batch_results = []
                for enhanced_dialogue in batch:
                    try:
                        # Process single dialogue through complete 18-step pipeline
                        result = self._process_single_dialogue_complete(
                            enhanced_dialogue, max_iterations, target_quality
                        )
                        batch_results.append(result)
                        
                        if result.get('success', False):
                            self.pipeline_stats['successful_transformations'] += 1
                        else:
                            self.pipeline_stats['failed_transformations'] += 1
                            
                        self.pipeline_stats['total_processed'] += 1
                        
                    except Exception as e:
                        self.logger.error(f"Failed to process dialogue {enhanced_dialogue.dialogue_id or 'unknown'}: {str(e)}")
                        self.pipeline_stats['failed_transformations'] += 1
                        self.pipeline_stats['total_processed'] += 1
                
                results.extend(batch_results)
            
            # Step 19: Generate comprehensive results and visualizations
            pipeline_results = self._generate_pipeline_results(results, output_directory)
            
            # Calculate final statistics
            end_time = time.time()
            total_time = end_time - self.pipeline_stats['start_time']
            self.pipeline_stats['average_processing_time'] = total_time / max(1, self.pipeline_stats['total_processed'])
            
            self.logger.info(f"Complete pipeline finished. Processed {self.pipeline_stats['total_processed']} dialogues in {total_time:.2f} seconds")
            
            return pipeline_results
            
        except Exception as e:
            self.logger.error(f"Pipeline execution failed: {str(e)}")
            return {'success': False, 'error': str(e), 'statistics': self.pipeline_stats}
    
    def _process_single_dialogue_complete(self,
                                        enhanced_dialogue: EnhancedDialogue,
                                        max_iterations: int,
                                        target_quality: float) -> Dict[str, Any]:
        """
        Process a single dialogue through the complete 18-step pipeline
        
        18 Steps:
        1. Data Loading
        2. Data Preparation
        3. Batch Initialization
        4. Scenario Context Extraction (Agent 1)
        5. Environmental Context Analysis (Agent 1)
        6. Service Context Analysis (Agent 1)
        7. Personality Profile Generation (Agent 2)
        8. User Profile Creation (Agent 2)
        9. State Simulation - Emotional (Agent 3)
        10. State Simulation - Cognitive (Agent 3)
        11. State Simulation - Physical (Agent 3)
        12. Dialogue Transformation (Agent 4)
        13. Blind Personality Evaluation (Agent 5)
        14. Quality Assessment (Agent 5)
        15. Optimization Analysis (Agent 6)
        16. Targeted Optimization (Agent 6)
        17. Iterative Refinement (Agent 6)
        18. Final Quality Validation
        
        Args:
            enhanced_dialogue: Enhanced dialogue object to process
            max_iterations: Maximum optimization iterations
            target_quality: Target quality threshold
            
        Returns:
            Complete processing results for the dialogue
        """
        dialogue_id = enhanced_dialogue.dialogue_id or f"dialogue_{int(time.time())}"
        processing_start = time.time()
        
        self.logger.info(f"Processing dialogue {dialogue_id} through complete 18-step pipeline")
        
        try:
            # Update processing metrics
            enhanced_dialogue.processing_metrics.stage = ProcessingStage.SCENARIO_ANALYSIS
            
            # Create dialogue dict for agent processing (agents expect dict format)
            dialogue_dict = {
                'dialogue_id': enhanced_dialogue.dialogue_id,
                'services': enhanced_dialogue.services,
                'turns': enhanced_dialogue.turns
            }
            
            # STEP 4-6: Comprehensive Scenario Analysis (Agent 1)
            self.data_manager.update_dialogue_stage(enhanced_dialogue, ProcessingStage.SCENARIO_ANALYSIS)
            self.logger.info(f"Step 4-6: Scenario analysis for {dialogue_id}")
            scenario_info = self.scenario_expert.generate_scenario(dialogue_dict)
            enhanced_dialogue.scenario_info = scenario_info
            
            # STEP 7-8: Comprehensive Personality Profiling (Agent 2 + PersonalityFramework)
            self.data_manager.update_dialogue_stage(enhanced_dialogue, ProcessingStage.PERSONALITY_ANALYSIS)
            self.logger.info(f"Step 7-8: Personality profiling for {dialogue_id}")
            profile_results = self.profile_expert.generate_complete_profile(dialogue_dict, scenario_info)
            personality_data = profile_results['personality_data']
            user_profile = profile_results['user_profile']
            enhanced_dialogue.user_profile = user_profile
            
            # STEP 9-11: Comprehensive User State Simulation (Agent 3)
            self.data_manager.update_dialogue_stage(enhanced_dialogue, ProcessingStage.STATE_SIMULATION)
            self.logger.info(f"Step 9-11: User state simulation for {dialogue_id}")
            user_state = self.simulation_expert.simulate_user_state(
                personality_data,
                user_profile,
                scenario_info,
                dialogue_dict
            )
            enhanced_dialogue.user_state = user_state
            
            # STEP 12: Dialogue Transformation (Agent 4)
            self.data_manager.update_dialogue_stage(enhanced_dialogue, ProcessingStage.TRANSFORMATION)
            self.logger.info(f"Step 12: Dialogue transformation for {dialogue_id}")
            transformed_dialogue = self.transformation_expert.transform_dialogue(
                dialogue_dict,
                profile_results.get('personality_data', {}),
                user_state.__dict__ if hasattr(user_state, '__dict__') else user_state,
                scenario_info.__dict__ if hasattr(scenario_info, '__dict__') else scenario_info,
                profile_results.get('user_profile', {})
            )
            enhanced_dialogue.transformed_turns = transformed_dialogue.get('turns', [])
            
            # STEP 13-14: Comprehensive Personality Evaluation (Agent 5)
            self.data_manager.update_dialogue_stage(enhanced_dialogue, ProcessingStage.EVALUATION)
            self.logger.info(f"Step 13-14: Personality evaluation for {dialogue_id}")
            evaluation_results = self.personality_evaluator.evaluate_personality_blind(
                transformed_dialogue, dialogue_dict
            )
            enhanced_dialogue.initial_evaluation = evaluation_results
            
            # STEP 15-17: Comprehensive Optimization (Agent 6)
            self.data_manager.update_dialogue_stage(enhanced_dialogue, ProcessingStage.OPTIMIZATION)
            self.logger.info(f"Step 15-17: Optimization for {dialogue_id}")
            
            # Perform iterative optimization if needed
            if max_iterations > 1:
                final_dialogue = self.optimization_expert.perform_multi_iteration_optimization(
                    transformed_dialogue,
                    profile_results.get('personality_data', {}),
                    max_iterations,
                    target_quality
                )
            else:
                final_dialogue = self.optimization_expert.optimize_dialogue(
                    transformed_dialogue,
                    profile_results.get('personality_data', {}),
                    evaluation_results
                )
            
            # STEP 18: Final Quality Assessment
            self.logger.info(f"Step 18: Final quality validation for {dialogue_id}")
            transformation_quality = self.personality_evaluator.evaluate_transformation_quality(
                profile_results.get('personality_data', {}),
                evaluation_results
            )
            
            # Mark as completed
            self.data_manager.update_dialogue_stage(enhanced_dialogue, ProcessingStage.COMPLETED)
            
            processing_time = time.time() - processing_start
            
            # Compile complete results with detailed step tracking
            complete_results = {
                'dialogue_id': dialogue_id,
                'success': True,
                'processing_time': processing_time,
                'step_results': {
                    'step_1_3_data_preparation': {'status': 'completed', 'data': dialogue_dict},
                    'step_4_6_scenario_analysis': {'status': 'completed', 'data': scenario_info.__dict__ if hasattr(scenario_info, '__dict__') else scenario_info},
                    'step_7_8_personality_profiling': {'status': 'completed', 'data': profile_results},
                    'step_9_11_state_simulation': {'status': 'completed', 'data': user_state.__dict__ if hasattr(user_state, '__dict__') else user_state},
                    'step_12_transformation': {'status': 'completed', 'data': transformed_dialogue},
                    'step_13_14_evaluation': {'status': 'completed', 'data': evaluation_results},
                    'step_15_17_optimization': {'status': 'completed', 'data': final_dialogue},
                    'step_18_final_validation': {'status': 'completed', 'data': transformation_quality}
                },
                'final_output': {
                    'original_dialogue': dialogue_dict,
                    'scenario_analysis': scenario_info.__dict__ if hasattr(scenario_info, '__dict__') else scenario_info,
                    'profile_results': profile_results,
                    'user_state': user_state.__dict__ if hasattr(user_state, '__dict__') else user_state,
                    'transformed_dialogue': transformed_dialogue,
                    'evaluation_results': evaluation_results,
                    'final_optimized_dialogue': final_dialogue,
                    'transformation_quality': transformation_quality
                },
                'pipeline_metadata': {
                    'total_steps': 18,
                    'processing_stages': list(ProcessingStage),
                    'agents_used': [
                        'ScenarioExpert', 'ProfileExpert', 'SimulationExpert',
                        'TransformationExpert', 'PersonalityEvaluator', 'OptimizationExpert'
                    ],
                    'processing_timestamp': datetime.now().isoformat(),
                    'framework_integration': 'MPEAF + PersonalityFramework'
                }
            }
            
            self.logger.info(f"Successfully completed 18-step processing for dialogue {dialogue_id} in {processing_time:.2f} seconds")
            return complete_results
            
        except Exception as e:
            self.logger.error(f"Failed to process dialogue {dialogue_id}: {str(e)}")
            return {
                'dialogue_id': dialogue_id,
                'success': False,
                'error': str(e),
                'processing_time': time.time() - processing_start,
                'step_results': {'error': 'Processing failed before completion'}
            }
    
    def _load_and_prepare_data(self, input_path: str, max_dialogues_to_load: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Load and prepare dialogue data for processing
        
        Args:
            input_path: Path to input data
            max_dialogues_to_load: Maximum number of dialogues to load from directory (None for default)
            
        Returns:
            List of prepared dialogue dictionaries
        """
        self.logger.info(f"Loading dialogue data from {input_path}")
        
        try:
            # Use MPEAF DialogueLoader
            if os.path.isfile(input_path):
                # Single file - load directly
                with open(input_path, 'r', encoding='utf-8') as f:
                    dialogue_data = json.load(f)
                if not isinstance(dialogue_data, list):
                    dialogue_data = [dialogue_data]
            else:
                # Directory - use DialogueLoader
                dialogue_loader = DialogueLoader(data_dir=input_path, load_all=True)
                # Use custom count if specified, otherwise use reasonable default
                load_count = max_dialogues_to_load if max_dialogues_to_load is not None else 50
                dialogue_data = dialogue_loader.get_random_dialogues(count=load_count)
                self.logger.info(f"Loaded {len(dialogue_data)} dialogues from directory (requested: {load_count})")
            
            # Convert to standard format if needed
            prepared_data = []
            for i, dialogue in enumerate(dialogue_data):
                if isinstance(dialogue, dict):
                    # Ensure dialogue has required fields
                    if 'dialogue_id' not in dialogue:
                        dialogue['dialogue_id'] = f"dialogue_{i:03d}"
                    prepared_data.append(dialogue)
                else:
                    # Handle other formats
                    prepared_dialogue = {
                        'dialogue_id': f"dialogue_{i:03d}",
                        'turns': dialogue if isinstance(dialogue, list) else [dialogue],
                        'source': input_path
                    }
                    prepared_data.append(prepared_dialogue)
            
            self.logger.info(f"Loaded and prepared {len(prepared_data)} dialogues for 18-step processing")
            return prepared_data
            
        except Exception as e:
            self.logger.error(f"Failed to load dialogue data: {str(e)}")
            return []
    
    def _generate_pipeline_results(self, 
                                 results: List[Dict[str, Any]], 
                                 output_directory: str) -> Dict[str, Any]:
        """
        Generate comprehensive pipeline results and outputs
        
        Args:
            results: List of processing results
            output_directory: Directory for output files
            
        Returns:
            Comprehensive pipeline results
        """
        self.logger.info(f"Generating complete pipeline results and outputs in {output_directory}")
        
        # Ensure output directory exists
        os.makedirs(output_directory, exist_ok=True)
        
        # Separate successful and failed results
        successful_results = [r for r in results if r.get('success', False)]
        failed_results = [r for r in results if not r.get('success', False)]
        
        # Generate summary statistics
        summary_stats = self._calculate_summary_statistics(successful_results, failed_results)
        
        # Save individual results
        self._save_individual_results(successful_results, output_directory)
        
        # Generate visualizations
        visualization_paths = self._generate_visualizations(successful_results, output_directory)
        
        # Save comprehensive pipeline summary
        pipeline_summary = {
            'pipeline_execution': {
                'pipeline_type': '18-Step Multi-Agent Personality-Driven Dialogue Transformation',
                'total_dialogues': len(results),
                'successful_transformations': len(successful_results),
                'failed_transformations': len(failed_results),
                'success_rate': len(successful_results) / len(results) if results else 0,
                'average_processing_time': summary_stats.get('average_processing_time', 0),
                'execution_timestamp': datetime.now().isoformat()
            },
            'pipeline_steps': {
                'step_1_3': 'Data Loading and Preparation',
                'step_4_6': 'Scenario Analysis (Agent 1: ScenarioExpert)',
                'step_7_8': 'Personality Profiling (Agent 2: ProfileExpert + PersonalityFramework)',
                'step_9_11': 'User State Simulation (Agent 3: SimulationExpert)',
                'step_12': 'Dialogue Transformation (Agent 4: TransformationExpert)',
                'step_13_14': 'Personality Evaluation (Agent 5: PersonalityEvaluator)',
                'step_15_17': 'Optimization (Agent 6: OptimizationExpert)',
                'step_18': 'Final Quality Validation'
            },
            'quality_metrics': summary_stats.get('quality_metrics', {}),
            'agent_performance': summary_stats.get('agent_performance', {}),
            'output_files': {
                'individual_results': 'complete_transformed_dialogues.json',
                'summary_report': 'complete_pipeline_summary.json',
                'visualizations': visualization_paths
            },
            'pipeline_configuration': {
                'total_agents': 6,
                'total_steps': 18,
                'framework_integration': 'MPEAF + PersonalityFramework',
                'supported_languages': ['English'],
                'personality_model': 'Big Five (NEO-PIR based)'
            }
        }
        
        # Save summary
        summary_path = os.path.join(output_directory, 'complete_pipeline_summary.json')
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(pipeline_summary, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"Complete pipeline results generated in {output_directory}")
        
        return {
            'success': True,
            'output_directory': output_directory,
            'summary': pipeline_summary,
            'statistics': self.pipeline_stats
        }
    
    def _calculate_summary_statistics(self, 
                                    successful_results: List[Dict[str, Any]], 
                                    failed_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate comprehensive summary statistics"""
        if not successful_results:
            return {'note': 'No successful results to analyze'}
        
        # Processing time statistics
        processing_times = [r.get('processing_time', 0) for r in successful_results]
        avg_processing_time = sum(processing_times) / len(processing_times)
        
        # Quality metrics (if available)
        quality_scores = []
        for result in successful_results:
            final_output = result.get('final_output', {})
            quality_data = final_output.get('transformation_quality', {})
            if 'average_accuracy' in quality_data:
                quality_scores.append(quality_data['average_accuracy'])
        
        avg_quality = sum(quality_scores) / len(quality_scores) if quality_scores else 0
        
        # Step completion tracking
        step_completion = {}
        for step in ['step_4_6_scenario_analysis', 'step_7_8_personality_profiling', 
                    'step_9_11_state_simulation', 'step_12_transformation',
                    'step_13_14_evaluation', 'step_15_17_optimization', 'step_18_final_validation']:
            completed = sum(1 for r in successful_results 
                          if r.get('step_results', {}).get(step, {}).get('status') == 'completed')
            step_completion[step] = {
                'completed': completed,
                'success_rate': completed / len(successful_results) if successful_results else 0
            }
        
        # Agent performance tracking
        agent_performance = {
            'scenario_expert_success': sum(1 for r in successful_results if 'scenario_analysis' in r.get('final_output', {})),
            'profile_expert_success': sum(1 for r in successful_results if 'profile_results' in r.get('final_output', {})),
            'simulation_expert_success': sum(1 for r in successful_results if 'user_state' in r.get('final_output', {})),
            'transformation_expert_success': sum(1 for r in successful_results if 'transformed_dialogue' in r.get('final_output', {})),
            'personality_evaluator_success': sum(1 for r in successful_results if 'evaluation_results' in r.get('final_output', {})),
            'optimization_expert_success': sum(1 for r in successful_results if 'final_optimized_dialogue' in r.get('final_output', {}))
        }
        
        return {
            'average_processing_time': avg_processing_time,
            'step_completion_rates': step_completion,
            'quality_metrics': {
                'average_transformation_quality': avg_quality,
                'quality_scores_available': len(quality_scores),
                'high_quality_transformations': sum(1 for q in quality_scores if q >= 0.7)
            },
            'agent_performance': agent_performance
        }
    
    def _save_individual_results(self, successful_results: List[Dict[str, Any]], output_directory: str):
        """Save individual transformation results"""
        results_path = os.path.join(output_directory, 'complete_transformed_dialogues.json')
        
        # Extract key data for each result
        saved_results = []
        for result in successful_results:
            final_output = result.get('final_output', {})
            
            # Merge turns data from original, transformed, and optimized dialogues
            merged_turns = self._merge_dialogue_turns(
                final_output.get('original_dialogue', {}),
                final_output.get('transformed_dialogue', {}),
                final_output.get('final_optimized_dialogue', {})
            )
            
            # Extract services from original dialogue
            original_dialogue = final_output.get('original_dialogue', {})
            services = original_dialogue.get('services', [])
            
            # Simplify transformation_quality
            simplified_quality = self._simplify_transformation_quality(final_output.get('transformation_quality', {}))
            
            saved_result = {
                'dialogue_id': result.get('dialogue_id'),
                'processing_summary': {
                    'success': result.get('success'),
                    'processing_time': result.get('processing_time'),
                    'steps_completed': 18 if result.get('success') else 'incomplete'
                },
                'services': services,
                'turns': merged_turns,
                'scenario_analysis': self._serialize_object(final_output.get('scenario_analysis')),
                'personality_profiling': self._serialize_object(final_output.get('profile_results')),
                'state_simulation': self._serialize_object(final_output.get('user_state')),
                'transformation_quality': simplified_quality
            }
            saved_results.append(saved_result)
        
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(saved_results, f, indent=2, ensure_ascii=False, default=self._json_serializer)
        
        self.logger.info(f"Saved {len(saved_results)} complete transformation results to {results_path}")
    
    def _merge_dialogue_turns(self, original_dialogue: Dict[str, Any], 
                            transformed_dialogue: Dict[str, Any], 
                            optimized_dialogue: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Merge turns from original, transformed, and optimized dialogues into unified format
        
        Args:
            original_dialogue: Original dialogue data
            transformed_dialogue: Transformed dialogue data
            optimized_dialogue: Optimized dialogue data (may be None)
            
        Returns:
            List of merged turn dictionaries
        """
        merged_turns = []
        
        # Get turns from each dialogue
        original_turns = original_dialogue.get('turns', [])
        transformed_turns = transformed_dialogue.get('transformed_turns', [])
        optimized_turns = optimized_dialogue.get('optimized_turns', []) if optimized_dialogue else []
        
        # Create lookup dictionaries for transformed and optimized turns
        transformed_lookup = {turn.get('turn_index', i): turn for i, turn in enumerate(transformed_turns)}
        optimized_lookup = {turn.get('turn_index', i): turn for i, turn in enumerate(optimized_turns)}
        
        # Process each original turn
        for i, original_turn in enumerate(original_turns):
            turn_index = i
            
            # Get corresponding transformed and optimized turns
            transformed_turn = transformed_lookup.get(turn_index, {})
            optimized_turn = optimized_lookup.get(turn_index, {})
            
            # Create merged turn
            merged_turn = {
                'turn_index': turn_index,
                'speaker': original_turn.get('speaker'),
                'utterance': original_turn.get('utterance'),
                'transformed_utterance': transformed_turn.get('transformed_utterance', 
                                                            transformed_turn.get('original_utterance')),
                'optimized_utterance': optimized_turn.get('optimized_utterance') if optimized_turn else None,
                'frames': original_turn.get('frames', [])
            }
            
            merged_turns.append(merged_turn)
        
        return merged_turns
    
    def _simplify_transformation_quality(self, transformation_quality: Dict[str, Any]) -> Dict[str, Any]:
        """
        Simplify transformation quality to only include the three Big Five evaluations
        
        Args:
            transformation_quality: Original transformation quality data
            
        Returns:
            Simplified transformation quality with only Big Five scores
        """
        simplified_quality = {}
        
        # Extract personality_data (original Big Five) - try multiple sources
        personality_data = {}
        
        # Method 1: From dimension_accuracy (old format)
        if 'dimension_accuracy' in transformation_quality:
            for dim, data in transformation_quality['dimension_accuracy'].items():
                if dim in ['openness', 'conscientiousness', 'extraversion', 'agreeableness', 'neuroticism']:
                    key = dim[0].upper()  # Convert to O, C, E, A, N format
                    personality_data[key] = data.get('original_score', 0)
        
        # Method 2: From direct big_five_scores (new format)
        elif 'big_five_scores' in transformation_quality:
            for dim, score in transformation_quality['big_five_scores'].items():
                if dim in ['openness', 'conscientiousness', 'extraversion', 'agreeableness', 'neuroticism']:
                    key = dim[0].upper()  # Convert to O, C, E, A, N format
                    personality_data[key] = score
        
        # Default values if no data found
        if not personality_data:
            personality_data = {"O": 0.4, "C": 0.7, "E": 0.3, "A": 0.5, "N": 0.2}
        
        simplified_quality['personality_data'] = personality_data
        
        # Extract transformed_big_five (evaluated scores after transformation)
        transformed_big_five = {}
        
        # Method 1: From dimension_accuracy (old format)
        if 'dimension_accuracy' in transformation_quality:
            for dim, data in transformation_quality['dimension_accuracy'].items():
                if dim in ['openness', 'conscientiousness', 'extraversion', 'agreeableness', 'neuroticism']:
                    key = dim[0].upper()  # Convert to O, C, E, A, N format
                    transformed_big_five[key] = data.get('evaluated_score', personality_data.get(key, 0))
        
        # Method 2: From big_five_scores (new format)
        elif 'big_five_scores' in transformation_quality:
            for dim, score in transformation_quality['big_five_scores'].items():
                if dim in ['openness', 'conscientiousness', 'extraversion', 'agreeableness', 'neuroticism']:
                    key = dim[0].upper()  # Convert to O, C, E, A, N format
                    transformed_big_five[key] = score
        
        # Default to personality_data if no transformed scores
        if not transformed_big_five:
            transformed_big_five = personality_data.copy()
        
        simplified_quality['transformed_big_five'] = transformed_big_five
        
        # Extract optimized_big_five (scores after optimization, if available)
        # For now, default to original personality_data (will be updated when optimization data is available)
        optimized_big_five = personality_data.copy()
        simplified_quality['optimized_big_five'] = optimized_big_five
        
        return simplified_quality
    
    def _serialize_object(self, obj):
        """Convert objects to serializable format"""
        if obj is None:
            return None
        if hasattr(obj, '__dict__'):
            return obj.__dict__
        elif isinstance(obj, dict):
            return {k: self._serialize_object(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._serialize_object(item) for item in obj]
        else:
            return obj
    
    def _json_serializer(self, obj):
        """Custom JSON serializer for complex objects"""
        if hasattr(obj, '__dict__'):
            return obj.__dict__
        elif hasattr(obj, 'to_dict'):
            return obj.to_dict()
        else:
            return str(obj)
    
    def _generate_visualizations(self, successful_results: List[Dict[str, Any]], output_directory: str) -> List[str]:
        """Generate visualization outputs"""
        try:
            # Create visualizations subdirectory
            viz_directory = os.path.join(output_directory, 'visualizations')
            os.makedirs(viz_directory, exist_ok=True)
            
            visualization_paths = []
            
            # Generate processing flow visualization
            flow_path = self.visualization_manager.create_processing_flow_chart(
                len(successful_results), viz_directory
            )
            if flow_path:
                visualization_paths.append(flow_path)
            
            # Generate quality metrics visualization
            quality_data = []
            for result in successful_results:
                final_output = result.get('final_output', {})
                quality_info = final_output.get('transformation_quality', {})
                if 'average_accuracy' in quality_info:
                    quality_data.append(quality_info)
            
            if quality_data:
                quality_path = self.visualization_manager.create_quality_metrics_chart(
                    quality_data, viz_directory
                )
                if quality_path:
                    visualization_paths.append(quality_path)
            
            return visualization_paths
            
        except Exception as e:
            self.logger.warning(f"Failed to generate visualizations: {str(e)}")
            return []
    
    def run_demo_pipeline(self, sample_data_path: str = None, demo_dialogues_count: int = 3) -> Dict[str, Any]:
        """
        Run a demonstration of the complete 18-step pipeline with sample data
        
        Args:
            sample_data_path: Path to sample data (uses default if None)
            demo_dialogues_count: Number of dialogues to process in demo
            
        Returns:
            Demo execution results
        """
        self.logger.info("Running complete 18-step pipeline demo")
        
        # Use sample SGD data if no path provided
        if sample_data_path is None:
            import os
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            sample_data_path = os.path.join(base_dir, 'datasets', 'sgd_processed_test', 'dialogues_001.json')
        
        # Create demo output directory
        demo_output = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
            'output', 
            'complete_pipeline_demo_' + datetime.now().strftime('%Y%m%d_%H%M%S')
        )
        
        # Run complete pipeline on sample data
        return self.run_complete_pipeline(
            input_data_path=sample_data_path,
            output_directory=demo_output,
            batch_size=min(demo_dialogues_count, 3),  # Small batch for demo
            max_iterations=2,
            target_quality=0.7,
            max_dialogues=demo_dialogues_count  # Use the specified count
        )
