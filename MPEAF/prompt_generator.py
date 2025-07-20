"""
Prompt Generator for Personality-driven Dialogue Transformation

This module implements a Chain-of-Thought (CoT) based prompt generator for 
transforming dialogue utterances according to specific personality profiles
using the NEO-PIR Big Five personality framework.
"""

import logging
from typing import Dict, Any

try:
    from .personality_framework import PersonalityFramework
except ImportError:
    from personality_framework import PersonalityFramework


class PromptGenerator:
    """
    Generate Chain-of-Thought prompts for personality-driven dialogue transformation
    
    This class creates structured prompts that guide language models to transform
    dialogue utterances according to specific personality profiles while maintaining
    coherence and task-oriented dialogue flow.
    """
    
    def __init__(self):
        """Initialize the prompt generator with personality framework"""
        self.personality_framework = PersonalityFramework()
        self.logger = logging.getLogger(__name__)
        
        # CoT prompt template for personality transformation of entire dialogue
        self.COT_TEMPLATE = """You are an expert dialogue analyst and personality psychologist. Your task is to transform ALL dialogue utterances to match specific personality profiles while maintaining dialogue coherence and task completion.

## Context Information
**Dialogue ID:** {dialogue_id}
**Services:** {services}
**Total Turns:** {total_turns}

## Target Personality Profile
**Big Five Dimensions:**
{personality_dimensions}

**Detailed Personality Traits:**
{personality_traits}

**Linguistic Markers to Apply:**
{linguistic_markers}

## Original Dialogue
{dialogue_context}

## Transformation Task
Transform ALL utterances in this dialogue to match the target personality profile. Consider:

### Step 1: Context Analysis
- Analyze the overall dialogue goal and task-oriented flow
- Identify the services being discussed and their progression
- Understand each speaker's role and information exchange patterns

### Step 2: Personality Application
- Apply the personality profile consistently across all turns
- Ensure personality traits influence language style throughout the dialogue
- Maintain personality consistency while preserving functional meaning

### Step 3: Linguistic Transformation
- Apply linguistic markers systematically to all utterances
- Adjust vocabulary, phrasing, and emotional expressions
- Ensure natural personality-consistent communication style

### Step 4: Coherence Check
- Verify all transformed utterances maintain dialogue flow
- Ensure task-oriented progression is preserved
- Check that the dialogue remains natural and realistic

## Output Format
Provide the transformed dialogue in the following JSON format:

{{
    "dialogue_id": "{dialogue_id}",
    "transformed_turns": [
        {{
            "turn_index": 0,
            "speaker": "USER",
            "original": "original utterance here",
            "transformed": "personality-matched transformation here"
        }},
        {{
            "turn_index": 1,
            "speaker": "SYSTEM",
            "original": "original utterance here", 
            "transformed": "personality-matched transformation here"
        }}
    ]
}}

Remember: Transform ALL utterances while maintaining the same functional meaning and task-oriented dialogue progression."""

    def generate_transformation_prompt(self, dialogue: Dict[str, Any]) -> str:
        """
        Generate a complete CoT prompt for transforming all turns in a dialogue
        
        Args:
            dialogue: Complete dialogue data with personality information
            
        Returns:
            Complete prompt string for LLM processing
            
        Raises:
            ValueError: If dialogue structure is invalid
        """
        try:
            # Validate input
            self._validate_dialogue_structure(dialogue)
            
            # Prepare dialogue context (all turns)
            dialogue_context = self._prepare_full_dialogue_context(dialogue)
            
            # Analyze personality profile
            personality_analysis = self._analyze_personality_profile(dialogue.get('personality', {}))
            
            # Format the complete prompt
            prompt = self.COT_TEMPLATE.format(
                dialogue_id=dialogue.get('dialogue_id', 'unknown'),
                services=', '.join(dialogue.get('services', [])),
                total_turns=len(dialogue['turns']),
                personality_dimensions=personality_analysis['dimensions_text'],
                personality_traits=personality_analysis['traits_text'],
                linguistic_markers=personality_analysis['markers_text'],
                dialogue_context=dialogue_context
            )
            
            return prompt
            
        except Exception as e:
            self.logger.error(f"Error generating prompt for dialogue {dialogue.get('dialogue_id', 'unknown')}: {str(e)}")
            raise



    def _validate_dialogue_structure(self, dialogue: Dict[str, Any]) -> None:
        """Validate that dialogue has required structure"""
        required_fields = ['turns']
        for field in required_fields:
            if field not in dialogue:
                raise ValueError(f"Missing required field: {field}")
        
        if not isinstance(dialogue['turns'], list) or len(dialogue['turns']) == 0:
            raise ValueError("Dialogue must have at least one turn")
        
        for i, turn in enumerate(dialogue['turns']):
            if 'utterance' not in turn or 'speaker' not in turn:
                raise ValueError(f"Turn {i} missing required fields: utterance, speaker")

    def _prepare_full_dialogue_context(self, dialogue: Dict[str, Any]) -> str:
        """
        Prepare complete dialogue context for the prompt
        
        Args:
            dialogue: Complete dialogue data
            
        Returns:
            Formatted dialogue context string with all turns
        """
        context_lines = []
        
        for i, turn in enumerate(dialogue['turns']):
            speaker = turn['speaker']
            utterance = turn['utterance']
            context_lines.append(f"Turn {i} [{speaker}]: \"{utterance}\"")
        
        return '\n'.join(context_lines)

    def _analyze_personality_profile(self, personality: Dict[str, Any]) -> Dict[str, str]:
        """
        Analyze personality profile and prepare formatted text
        
        Args:
            personality: Personality data from dialogue
            
        Returns:
            Dictionary with formatted personality analysis
        """
        analysis = {
            'dimensions_text': '',
            'traits_text': '',
            'markers_text': ''
        }
        
        if not personality:
            analysis['dimensions_text'] = "No personality profile provided - use neutral style"
            analysis['traits_text'] = "No specific traits to apply"
            analysis['markers_text'] = "Use standard conversational markers"
            return analysis
        
        # Process Big Five dimensions
        big_five = personality.get('big_five', {})
        if big_five:
            dimension_lines = []
            for dim, score in big_five.items():
                if isinstance(score, (int, float)):
                    dim_info = self.personality_framework.NEO_PIR_FACETS.get(dim, {})
                    dim_name = dim_info.get('name', dim)
                    level = self._score_to_level(score)
                    dimension_lines.append(f"- {dim_name} ({dim}): {score:.3f} ({level})")
            analysis['dimensions_text'] = '\n'.join(dimension_lines)
        
        # Process detailed facets
        facets = personality.get('facets', {})
        if facets:
            trait_lines = []
            markers_list = []
            
            for dimension, traits in facets.items():
                if isinstance(traits, dict):
                    for trait_code, score in traits.items():
                        if isinstance(score, (int, float)):
                            trait_info = self._get_trait_info(trait_code)
                            level = self._score_to_level(score)
                            trait_lines.append(f"- {trait_info['name']} ({trait_code}): {score:.3f} ({level})")
                            
                            # Collect linguistic markers for high-scoring traits
                            if score > 0.6:  # High score threshold
                                markers = self.personality_framework.get_trait_markers(trait_code)
                                markers_list.extend(markers)
            
            analysis['traits_text'] = '\n'.join(trait_lines) if trait_lines else "No detailed traits available"
            analysis['markers_text'] = '\n'.join(f"- {marker}" for marker in markers_list[:10])  # Limit to top 10
        
        return analysis

    def _get_trait_info(self, trait_code: str) -> Dict[str, str]:
        """Get trait information from personality framework"""
        all_traits = self.personality_framework.get_all_traits_with_markers()
        return all_traits.get(trait_code, {
            'name': trait_code,
            'dimension': trait_code[0] if trait_code else 'Unknown'
        })

    def _score_to_level(self, score: float) -> str:
        """Convert numeric score to descriptive level"""
        if score >= 0.7:
            return "High"
        elif score >= 0.4:
            return "Moderate"
        else:
            return "Low"




