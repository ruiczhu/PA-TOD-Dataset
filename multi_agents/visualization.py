"""
Visualization and Progress Tracking for Multi-Agent Pipeline

This module provides real-time visualization and progress tracking capabilities
for the multi-agent dialogue processing pipeline.
"""

import time
import sys
import os
import json
from typing import Dict, List, Any, Optional
from datetime import datetime
from .utils.data_structures import ProcessingStage, EnhancedDialogue


class VisualizationManager:
    """
    Main visualization manager for multi-agent pipeline
    
    Handles creation of charts, graphs, and visual reports for pipeline results.
    """
    
    def __init__(self):
        """Initialize visualization manager"""
        self.output_dir = None
        
    def create_processing_flow_chart(self, num_dialogues: int, output_dir: str) -> Optional[str]:
        """
        Create a processing flow chart visualization
        
        Args:
            num_dialogues: Number of dialogues processed
            output_dir: Output directory for visualization
            
        Returns:
            Path to created visualization file or None if failed
        """
        try:
            flow_data = {
                "processing_flow": {
                    "total_dialogues": num_dialogues,
                    "stages": [
                        "Data Loading",
                        "Scenario Analysis", 
                        "Personality Profiling",
                        "State Simulation",
                        "Dialogue Transformation",
                        "Personality Evaluation",
                        "Optimization",
                        "Final Validation"
                    ],
                    "agents": [
                        "ScenarioExpert",
                        "ProfileExpert", 
                        "SimulationExpert",
                        "TransformationExpert",
                        "PersonalityEvaluator",
                        "OptimizationExpert"
                    ]
                }
            }
            
            flow_path = os.path.join(output_dir, 'processing_flow.json')
            with open(flow_path, 'w') as f:
                json.dump(flow_data, f, indent=2)
                
            return flow_path
            
        except Exception:
            return None
    
    def create_quality_metrics_chart(self, quality_data: List[Dict[str, Any]], output_dir: str) -> Optional[str]:
        """
        Create quality metrics visualization
        
        Args:
            quality_data: List of quality metric dictionaries
            output_dir: Output directory for visualization
            
        Returns:
            Path to created visualization file or None if failed
        """
        try:
            metrics_summary = {
                "quality_metrics": {
                    "total_evaluations": len(quality_data),
                    "average_quality": sum(q.get('average_accuracy', 0) for q in quality_data) / len(quality_data) if quality_data else 0,
                    "high_quality_count": sum(1 for q in quality_data if q.get('average_accuracy', 0) >= 0.7),
                    "quality_distribution": {
                        "excellent": sum(1 for q in quality_data if q.get('average_accuracy', 0) >= 0.8),
                        "good": sum(1 for q in quality_data if 0.6 <= q.get('average_accuracy', 0) < 0.8),
                        "fair": sum(1 for q in quality_data if 0.4 <= q.get('average_accuracy', 0) < 0.6),
                        "poor": sum(1 for q in quality_data if q.get('average_accuracy', 0) < 0.4)
                    }
                }
            }
            
            metrics_path = os.path.join(output_dir, 'quality_metrics.json')
            with open(metrics_path, 'w') as f:
                json.dump(metrics_summary, f, indent=2)
                
            return metrics_path
            
        except Exception:
            return None


class ProgressTracker:
    """
    Real-time progress tracking for multi-agent pipeline
    
    Provides progress bars, stage tracking, and performance metrics
    during the dialogue processing workflow.
    """
    
    def __init__(self, total_dialogues: int):
        """
        Initialize progress tracker
        
        Args:
            total_dialogues: Total number of dialogues to process
        """
        self.total_dialogues = total_dialogues
        self.current_dialogue = 0
        self.start_time = time.time()
        self.stage_times = {}
        self.current_stage = ProcessingStage.LOADED
        self.error_count = 0
        self.success_count = 0
        
        # Stage display names
        self.stage_names = {
            ProcessingStage.LOADED: "Loading Data",
            ProcessingStage.SCENARIO_ENHANCED: "Scenario Enhancement",
            ProcessingStage.PROFILE_GENERATED: "Profile Generation", 
            ProcessingStage.STATE_SIMULATED: "State Simulation",
            ProcessingStage.TRANSFORMED: "Dialogue Transformation",
            ProcessingStage.EVALUATED: "Initial Evaluation",
            ProcessingStage.OPTIMIZED: "Dialogue Optimization",
            ProcessingStage.FINAL_EVALUATED: "Final Evaluation",
            ProcessingStage.COMPLETED: "Completed"
        }
        
        print("Multi-Agent Dialogue Processing Pipeline Started")
        print(f"Total dialogues to process: {total_dialogues}")
        print("-" * 60)
    
    def start_dialogue(self, dialogue_id: str):
        """
        Start processing a new dialogue
        
        Args:
            dialogue_id: ID of the dialogue being processed
        """
        self.current_dialogue += 1
        self.current_dialogue_id = dialogue_id
        self.dialogue_start_time = time.time()
        
        progress_percent = (self.current_dialogue / self.total_dialogues) * 100
        
        print(f"\nProcessing Dialogue {self.current_dialogue}/{self.total_dialogues} ({progress_percent:.1f}%)")
        print(f"Dialogue ID: {dialogue_id}")
        print(f"Started at: {datetime.now().strftime('%H:%M:%S')}")
    
    def update_stage(self, stage: ProcessingStage, agent_name: str = "", details: str = ""):
        """
        Update current processing stage
        
        Args:
            stage: Current processing stage
            agent_name: Name of the agent handling this stage
            details: Additional details about the stage
        """
        self.current_stage = stage
        stage_name = self.stage_names.get(stage, stage.value)
        
        # Calculate stage timing
        current_time = time.time()
        if hasattr(self, 'dialogue_start_time'):
            stage_duration = current_time - self.dialogue_start_time
        else:
            stage_duration = 0
        
        # Display stage update
        status_icon = "[PROCESSING]" if stage != ProcessingStage.COMPLETED else "[COMPLETED]"
        print(f"  {status_icon} {stage_name}", end="")
        
        if agent_name:
            print(f" ({agent_name})", end="")
        
        if details:
            print(f" - {details}", end="")
            
        print(f" [{stage_duration:.1f}s]")
        
        # Update progress bar
        self._update_progress_bar()
    
    def _update_progress_bar(self):
        """Update the progress bar display"""
        # Calculate overall progress
        stage_weights = {
            ProcessingStage.LOADED: 0.05,
            ProcessingStage.SCENARIO_ENHANCED: 0.15,
            ProcessingStage.PROFILE_GENERATED: 0.25,
            ProcessingStage.STATE_SIMULATED: 0.35,
            ProcessingStage.TRANSFORMED: 0.50,
            ProcessingStage.EVALUATED: 0.65,
            ProcessingStage.OPTIMIZED: 0.80,
            ProcessingStage.FINAL_EVALUATED: 0.95,
            ProcessingStage.COMPLETED: 1.0
        }
        
        dialogue_progress = stage_weights.get(self.current_stage, 0)
        overall_progress = ((self.current_dialogue - 1) + dialogue_progress) / self.total_dialogues
        
        # Create progress bar
        bar_length = 40
        filled_length = int(bar_length * overall_progress)
        bar = "=" * filled_length + "-" * (bar_length - filled_length)
        
        # Calculate ETA
        elapsed_time = time.time() - self.start_time
        if overall_progress > 0:
            eta_seconds = (elapsed_time / overall_progress) * (1 - overall_progress)
            eta_minutes = int(eta_seconds // 60)
            eta_seconds = int(eta_seconds % 60)
            eta_str = f"{eta_minutes:02d}:{eta_seconds:02d}"
        else:
            eta_str = "--:--"
        
        print(f"  Progress: |{bar}| {overall_progress*100:.1f}% | ETA: {eta_str}")
    
    def log_error(self, error_message: str, agent_name: str = ""):
        """
        Log an error during processing
        
        Args:
            error_message: Error message to log
            agent_name: Name of the agent where error occurred
        """
        self.error_count += 1
        agent_info = f" in {agent_name}" if agent_name else ""
        print(f"  [ERROR]{agent_info}: {error_message}")
    
    def log_success(self, message: str = "", metrics: Dict[str, Any] = None):
        """
        Log successful completion of a dialogue
        
        Args:
            message: Success message
            metrics: Processing metrics to display
        """
        self.success_count += 1
        
        dialogue_duration = time.time() - self.dialogue_start_time
        print(f"  [SUCCESS] Completed in {dialogue_duration:.1f}s")
        
        if message:
            print(f"  Note: {message}")
            
        if metrics:
            if 'token_usage' in metrics:
                total_tokens = sum(metrics['token_usage'].values())
                print(f"  Tokens used: {total_tokens}")
            if 'api_calls' in metrics:
                print(f"  API calls: {metrics['api_calls']}")
    
    def display_summary(self):
        """Display final processing summary"""
        total_time = time.time() - self.start_time
        minutes = int(total_time // 60)
        seconds = int(total_time % 60)
        
        print("\n" + "=" * 60)
        print("Multi-Agent Pipeline Summary")
        print("=" * 60)
        print(f"Total dialogues processed: {self.current_dialogue}")
        print(f"Successful: {self.success_count}")
        print(f"Errors: {self.error_count}")
        print(f"Total time: {minutes:02d}:{seconds:02d}")
        
        if self.success_count > 0:
            avg_time = total_time / self.success_count
            print(f"Average time per dialogue: {avg_time:.1f}s")
        
        success_rate = (self.success_count / max(self.current_dialogue, 1)) * 100
        print(f"Success rate: {success_rate:.1f}%")
        print("=" * 60)


class StageVisualizer:
    """
    Visual representation of processing stages
    
    Provides a visual overview of which stage each dialogue is in
    and overall pipeline status.
    """
    
    def __init__(self):
        """Initialize stage visualizer"""
        self.stage_icons = {
            ProcessingStage.LOADED: "[LOADED]",
            ProcessingStage.SCENARIO_ENHANCED: "[SCENARIO]", 
            ProcessingStage.PROFILE_GENERATED: "[PROFILE]",
            ProcessingStage.STATE_SIMULATED: "[STATE]",
            ProcessingStage.TRANSFORMED: "[TRANSFORM]",
            ProcessingStage.EVALUATED: "[EVAL]",
            ProcessingStage.OPTIMIZED: "[OPTIMIZE]",
            ProcessingStage.FINAL_EVALUATED: "[FINAL_EVAL]",
            ProcessingStage.COMPLETED: "[COMPLETE]"
        }
    
    def display_pipeline_status(self, dialogues: List[EnhancedDialogue]):
        """
        Display current status of all dialogues in pipeline
        
        Args:
            dialogues: List of enhanced dialogues with their current stages
        """
        print("\nPipeline Status Overview")
        print("-" * 40)
        
        # Count dialogues by stage
        stage_counts = {}
        for dialogue in dialogues:
            stage = dialogue.processing_metrics.stage
            stage_counts[stage] = stage_counts.get(stage, 0) + 1
        
        # Display stage summary
        for stage in ProcessingStage:
            count = stage_counts.get(stage, 0)
            if count > 0:
                icon = self.stage_icons.get(stage, "[UNKNOWN]")
                stage_name = stage.value.replace("_", " ").title()
                print(f"{icon} {stage_name}: {count} dialogues")
        
        print("-" * 40)
