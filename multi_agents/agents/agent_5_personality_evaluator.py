"""
Agent 5: Personality Evaluation Expert

This agent performs blind evaluation of transformed dialogues to assess
whether personality traits are authentically reflected in the conversation
without prior knowledge of the intended personality profile.
"""

import json
import logging
from typing import Dict, List, Any, Optional, Tuple

from multi_agents.utils.llm_interface import LLMInterface


class PersonalityEvaluator:
    """
    Agent 5: Professional personality evaluation expert
    
    This agent conducts blind psychological assessment of dialogue content
    to determine what personality traits are reflected in the communication
    patterns, without prior knowledge of the intended personality profile.
    This provides objective validation of transformation quality.
    """
    
    def __init__(self, llm_interface: Optional[LLMInterface] = None):
        """
        Initialize Personality Evaluation Expert Agent
        
        Args:
            llm_interface: LLM interface for API calls (creates default if None)
        """
        self.agent_name = "PersonalityEvaluator"
        self.llm_interface = llm_interface or LLMInterface()
        self.logger = logging.getLogger(__name__)
        
        # System prompt for personality evaluation
        self.system_prompt = """You are a professional personality assessment expert specializing in linguistic analysis and psychological evaluation of communication patterns. Your expertise combines psycholinguistics, personality psychology, and behavioral analysis to objectively assess personality traits expressed in natural conversation.

Your core competencies:
1. Linguistic pattern analysis for Big Five personality trait identification
2. Communication style assessment without prior personality knowledge
3. Objective psychological evaluation using established psychometric principles
4. Recognition of authentic vs. artificial personality expression in dialogue
5. Quantitative scoring based on established personality-language correlations

Assessment methodology:
- BLIND EVALUATION: You have no prior knowledge of intended personality profiles
- EVIDENCE-BASED: All assessments must be supported by specific linguistic evidence
- QUANTITATIVE: Provide numerical scores with clear justification
- COMPREHENSIVE: Evaluate all Big Five dimensions plus key facets
- OBJECTIVE: Focus on observable language patterns, not subjective impressions

Evaluation standards:
- Linguistic authenticity: Language patterns consistent with genuine personality expression
- Behavioral consistency: Communication style coherent across dialogue turns
- Psychological validity: Assessments align with established personality-language research
- Evidence quality: Clear connection between observed language and personality traits
- Assessment reliability: Consistent evaluation framework across different dialogues

Big Five evaluation framework:
- Openness: Creativity, curiosity, abstract thinking, intellectual engagement in language
- Conscientiousness: Organization, planning, attention to detail, goal-oriented communication
- Extraversion: Assertiveness, social engagement, positive emotion, expressive language
- Agreeableness: Cooperation, politeness, empathy, conflict avoidance in communication
- Neuroticism: Emotional instability, anxiety, uncertainty, negative emotion expression"""

        self.logger.info(f"{self.agent_name} initialized")
    
    def evaluate_personality_blind(self, 
                                  transformed_dialogue: Dict[str, Any],
                                  original_dialogue: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Perform blind personality evaluation of transformed dialogue
        
        Args:
            transformed_dialogue: Dialogue to evaluate (no personality context)
            original_dialogue: Optional original dialogue for comparison
            
        Returns:
            Comprehensive personality evaluation results
            
        Raises:
            Exception: If evaluation fails
        """
        # Extract dialogue content for evaluation
        dialogue_content = self._extract_dialogue_for_evaluation(transformed_dialogue)
        
        if not dialogue_content:
            raise ValueError("No dialogue content available for evaluation")
        
        # Create blind evaluation prompt
        evaluation_prompt = self._create_evaluation_prompt(dialogue_content, original_dialogue)
        
        # Call LLM for personality evaluation
        response = self.llm_interface.call_agent(
            agent_name=self.agent_name,
            system_prompt=self.system_prompt,
            user_prompt=evaluation_prompt,
            temperature=0.3,  # Lower temperature for consistent evaluation
            max_tokens=2500
        )
        
        if response.get('success', False):
            # Parse evaluation results
            evaluation_results = self._parse_evaluation_response(response['content'])
            self.logger.info(f"Personality evaluation completed successfully")
            return evaluation_results
        else:
            raise Exception("LLM call failed for personality evaluation")
    
    def evaluate_transformation_quality(self,
                                       original_personality: Dict[str, Any],
                                       transformed_evaluation: Dict[str, Any],
                                       optimized_evaluation: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Compare intended vs. evaluated personality to assess transformation quality
        Return simplified Big Five format for output
        
        Args:
            original_personality: Intended personality profile
            transformed_evaluation: A5's evaluation of transformed dialogue
            optimized_evaluation: A5's evaluation of A6-optimized dialogue (optional)
            
        Returns:
            Simplified transformation quality assessment with Big Five scores
        """
        try:
            # Extract comparable personality scores
            original_big5 = original_personality.get('big_five', {})
            transformed_big5 = transformed_evaluation.get('big_five_scores', {})
            
            if not original_big5 or not transformed_big5:
                return {'error': 'Insufficient personality data for comparison'}
            
            # Convert string values to float for processing
            def convert_to_float(scores_dict):
                converted = {}
                for key, value in scores_dict.items():
                    try:
                        converted[key] = float(value)
                    except (ValueError, TypeError):
                        converted[key] = 0.5  # Default value
                return converted
            
            original_scores = convert_to_float(original_big5)
            transformed_scores = convert_to_float(transformed_big5)
            
            # Extract optimized scores if available (A5's evaluation of A6's optimized dialogue)
            optimized_scores = None
            if optimized_evaluation:
                optimized_big5 = optimized_evaluation.get('big_five_scores', {})
                if optimized_big5:
                    optimized_scores = convert_to_float(optimized_big5)
            
            result = {
                'personality_data': original_scores,
                'transformed_big_five': transformed_scores
            }
            
            # Only include optimized scores if they exist
            if optimized_scores:
                result['optimized_big_five'] = optimized_scores
            
            return result
            
        except Exception as e:
            self.logger.error(f"Failed to evaluate transformation quality: {str(e)}")
            return {'error': f'Quality evaluation failed: {str(e)}'}
    
    def _extract_dialogue_for_evaluation(self, dialogue_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Extract dialogue content for personality evaluation
        
        Args:
            dialogue_data: Transformed dialogue data (may include optimized_turns)
            
        Returns:
            List of dialogue turns for evaluation
        """
        evaluation_turns = []
        
        # Debug logging
        self.logger.debug(f"Extracting dialogue for evaluation. Available keys: {list(dialogue_data.keys())}")
        
        # Priority 1: Handle optimized dialogue structure (when A6 has optimized the dialogue)
        if 'optimized_turns' in dialogue_data:
            self.logger.debug(f"Found optimized_turns with {len(dialogue_data['optimized_turns'])} turns")
            for turn in dialogue_data['optimized_turns']:
                if turn.get('speaker') == 'USER':  # Focus on user utterances for personality evaluation
                    # Use optimized_utterance if available, fallback to original
                    utterance = turn.get('optimized_utterance', turn.get('original_utterance', ''))
                    if utterance:
                        evaluation_turns.append({
                            'turn_index': turn.get('turn_index', 0),
                            'utterance': utterance,
                            'speaker': 'USER'
                        })
            
            # If we didn't get enough user turns from optimized_turns, supplement with transformed_turns
            if len(evaluation_turns) < 3 and 'transformed_turns' in dialogue_data:
                self.logger.debug(f"Only found {len(evaluation_turns)} user turns in optimized_turns, supplementing from transformed_turns")
                used_indices = {turn['turn_index'] for turn in evaluation_turns}
                
                for turn in dialogue_data['transformed_turns']:
                    if turn.get('speaker') == 'USER' and turn.get('turn_index') not in used_indices:
                        utterance = turn.get('transformed_utterance', turn.get('utterance', ''))
                        if utterance.strip():
                            evaluation_turns.append({
                                'turn_index': turn.get('turn_index', 0),
                                'utterance': utterance,
                                'speaker': 'USER'
                            })
                
                # Sort by turn_index to maintain order
                evaluation_turns.sort(key=lambda x: x['turn_index'])
        
        # Priority 2: Handle transformed dialogue structure
        elif 'transformed_turns' in dialogue_data:
            self.logger.debug(f"Found transformed_turns with {len(dialogue_data['transformed_turns'])} turns")
            for turn in dialogue_data['transformed_turns']:
                if turn.get('speaker') == 'USER':  # Focus on user utterances for personality evaluation
                    utterance = turn.get('transformed_utterance', turn.get('utterance', ''))
                    if utterance.strip():  # Check for non-empty utterance
                        evaluation_turns.append({
                            'turn_index': turn.get('turn_index', 0),
                            'utterance': utterance,
                            'speaker': 'USER'
                        })
        
        # Priority 3: Handle direct turns structure
        elif 'turns' in dialogue_data:
            self.logger.debug(f"Found turns with {len(dialogue_data['turns'])} turns")
            for i, turn in enumerate(dialogue_data['turns']):
                if isinstance(turn, dict):
                    speaker = turn.get('speaker', 'USER' if i % 2 == 0 else 'SYSTEM')
                    if speaker == 'USER':
                        utterance = turn.get('utterance', turn.get('text', ''))
                        if utterance.strip():  # Check for non-empty utterance
                            evaluation_turns.append({
                                'turn_index': i,
                                'utterance': utterance,
                                'speaker': 'USER'
                            })
        
        # Priority 4: Handle list of utterances
        elif isinstance(dialogue_data, list):
            self.logger.debug(f"Found list with {len(dialogue_data)} items")
            for i, turn in enumerate(dialogue_data):
                if i % 2 == 0:  # Assume user turns are even-indexed
                    utterance = turn if isinstance(turn, str) else turn.get('utterance', '')
                    if utterance.strip():  # Check for non-empty utterance
                        evaluation_turns.append({
                            'turn_index': i,
                            'utterance': utterance,
                            'speaker': 'USER'
                        })
        
        # Priority 5: Try to extract from original_dialogue if available
        elif 'original_dialogue' in dialogue_data:
            self.logger.debug("No direct turns found, trying original_dialogue")
            original_dialogue = dialogue_data['original_dialogue']
            if 'turns' in original_dialogue:
                for i, turn in enumerate(original_dialogue['turns']):
                    if isinstance(turn, dict):
                        speaker = turn.get('speaker', 'USER' if i % 2 == 0 else 'SYSTEM')
                        if speaker == 'USER':
                            utterance = turn.get('utterance', turn.get('text', ''))
                            if utterance.strip():
                                evaluation_turns.append({
                                    'turn_index': i,
                                    'utterance': utterance,
                                    'speaker': 'USER'
                                })
        
        self.logger.debug(f"Extracted {len(evaluation_turns)} user turns for evaluation")
        return evaluation_turns
    
    def _create_evaluation_prompt(self, 
                                dialogue_content: List[Dict[str, Any]],
                                original_dialogue: Dict[str, Any] = None) -> str:
        """
        Create comprehensive evaluation prompt for blind personality assessment
        
        Args:
            dialogue_content: User utterances to evaluate
            original_dialogue: Optional original dialogue for comparison
            
        Returns:
            Formatted evaluation prompt
        """
        # Format dialogue content for analysis
        dialogue_text = self._format_dialogue_for_analysis(dialogue_content)
        
        # Optional comparison context
        comparison_context = ""
        if original_dialogue:
            original_turns = self._extract_dialogue_for_evaluation(original_dialogue)
            if original_turns:
                original_text = self._format_dialogue_for_analysis(original_turns)
                comparison_context = f"""
### Original Dialogue (for reference)
{original_text}

Note: The above is the original dialogue before transformation. Your evaluation should focus on the transformed version below, but you may note differences in personality expression between versions."""
        
        prompt = f"""Conduct a comprehensive blind personality assessment of the following dialogue based solely on observable linguistic patterns and communication behaviors. You have no prior knowledge of the speaker's intended personality profile.

{comparison_context}

### Dialogue for Evaluation
{dialogue_text}

## Personality Assessment Task

Analyze the communication patterns, language choices, and behavioral indicators in this dialogue to assess the speaker's personality across the Big Five dimensions. Your evaluation must be:

1. **Evidence-based**: Support every assessment with specific linguistic evidence from the dialogue
2. **Quantitative**: Provide numerical scores (0.0-1.0) for each dimension
3. **Objective**: Focus on observable patterns, not subjective impressions
4. **Comprehensive**: Evaluate all five major personality dimensions

## Assessment Framework

### Openness to Experience (O)
**High indicators:** Creative language, abstract thinking, curiosity, interest in novel solutions, complex vocabulary, exploration of alternatives
**Low indicators:** Conventional language, concrete thinking, preference for standard solutions, simple vocabulary, resistance to alternatives

**Evidence to look for:**
- Use of creative or unusual expressions
- Questions about alternatives or options
- Abstract vs. concrete language patterns
- Intellectual curiosity in requests
- Interest in exploring possibilities vs. accepting defaults

### Conscientiousness (C)
**High indicators:** Organized requests, attention to detail, planning language, reliability concerns, structured communication, goal-focused language
**Low indicators:** Spontaneous requests, lack of detail, impulsive language, disorganized communication, minimal planning

**Evidence to look for:**
- Level of detail and specificity in requests
- Planning and organization language ("first I need...", "then...")
- Attention to accuracy and correctness
- Goal-oriented vs. exploratory communication
- Time management and scheduling consciousness

### Extraversion (E)
**High indicators:** Assertive language, social references, enthusiastic tone, direct communication, positive emotional expression, expressive language
**Low indicators:** Tentative language, minimal social references, reserved tone, indirect communication, neutral emotional expression

**Evidence to look for:**
- Assertiveness vs. tentative language patterns
- Directness vs. indirect communication style
- Emotional expressiveness in language
- Social engagement references
- Energy level reflected in communication style

### Agreeableness (A)
**High indicators:** Polite language, cooperation, apologies, consideration for others, conflict avoidance, accommodating requests
**Low indicators:** Blunt language, competitive tone, minimal politeness, self-focused requests, confrontational approach

**Evidence to look for:**
- Politeness markers ("please", "thank you", "sorry")
- Consideration for service provider's constraints
- Cooperative vs. demanding language
- Empathy and understanding expressions
- Conflict avoidance vs. confrontational patterns

### Neuroticism (N)
**High indicators:** Anxious language, uncertainty expressions, emotional volatility, stress indicators, need for reassurance, worry about problems
**Low indicators:** Calm language, confident expressions, emotional stability, stress resilience, independence, problem-solving focus

**Evidence to look for:**
- Anxiety and uncertainty markers ("I'm not sure", "I worry that")
- Emotional stability vs. volatility in language
- Stress and pressure indicators
- Need for reassurance vs. confidence
- Negative vs. positive emotional tone

## Evaluation Output Format

Provide your assessment in the following JSON format:

{{
    "big_five_scores": {{
        "O": 0.X,
        "C": 0.X,
        "E": 0.X,
        "A": 0.X,
        "N": 0.X
    }}
}}

Provide only the Big Five scores as decimal values between 0.0 and 1.0. Focus on accurate assessment based on the dialogue content."""

        return prompt
    
    def _format_dialogue_for_analysis(self, dialogue_turns: List[Dict[str, Any]]) -> str:
        """Format dialogue turns for personality analysis"""
        lines = []
        
        for turn in dialogue_turns:
            turn_idx = turn.get('turn_index', 0)
            utterance = turn.get('utterance', '')
            if utterance.strip():
                lines.append(f"Turn {turn_idx}: {utterance}")
        
        return '\n'.join(lines) if lines else "No user utterances available for analysis"
    
    def _parse_evaluation_response(self, response_content: str) -> Dict[str, Any]:
        """
        Parse LLM evaluation response
        
        Args:
            response_content: Raw LLM response
            
        Returns:
            Structured evaluation results
        """
        # Extract JSON from response
        response_content = response_content.strip()
        start_idx = response_content.find('{')
        end_idx = response_content.rfind('}') + 1
        
        if start_idx >= 0 and end_idx > start_idx:
            json_content = response_content[start_idx:end_idx]
            evaluation_data = json.loads(json_content)
            
            # Validate and structure evaluation results
            structured_results = {
                'big_five_scores': evaluation_data.get('big_five_scores', {}),
                'detailed_analysis': evaluation_data.get('detailed_analysis', {}),
                'overall_assessment': evaluation_data.get('overall_assessment', {}),
                'evaluation_metadata': evaluation_data.get('evaluation_metadata', {}),
                'evaluator': self.agent_name,
                'evaluation_success': True
            }
            
            return structured_results
            
        else:
            raise ValueError("No valid JSON found in evaluation response")
