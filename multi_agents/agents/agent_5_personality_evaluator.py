"""
Agent 5: Personality Evaluation Expert

This agent performs blind evaluation of transformed dialogues to assess
whether personality traits are authentically reflected in the conversation
without prior knowledge of the intended personality profile.
"""

import json
import logging
from typing import Dict, List, Any, Optional, Tuple

# Import utilities with proper path handling
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

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
        try:
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
                
        except Exception as e:
            self.logger.error(f"Failed to evaluate personality: {str(e)}")
            return self._create_fallback_evaluation()
    
    def evaluate_transformation_quality(self,
                                       original_personality: Dict[str, Any],
                                       evaluated_personality: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compare intended vs. evaluated personality to assess transformation quality
        
        Args:
            original_personality: Intended personality profile
            evaluated_personality: Blind evaluation results
            
        Returns:
            Transformation quality assessment
        """
        try:
            # Extract comparable personality scores
            original_big5 = original_personality.get('big_five', {})
            evaluated_big5 = evaluated_personality.get('big_five_scores', {})
            
            if not original_big5 or not evaluated_big5:
                return {'error': 'Insufficient personality data for comparison'}
            
            # Calculate accuracy metrics
            quality_metrics = self._calculate_transformation_accuracy(original_big5, evaluated_big5)
            
            # Assess overall transformation quality
            overall_quality = self._assess_overall_quality(quality_metrics)
            
            return {
                'transformation_quality': overall_quality,
                'dimension_accuracy': quality_metrics,
                'evaluation_metadata': {
                    'evaluator': self.agent_name,
                    'comparison_method': 'Big Five correlation analysis'
                }
            }
            
        except Exception as e:
            self.logger.error(f"Failed to evaluate transformation quality: {str(e)}")
            return {'error': f'Quality evaluation failed: {str(e)}'}
    
    def _extract_dialogue_for_evaluation(self, dialogue_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Extract dialogue content for personality evaluation
        
        Args:
            dialogue_data: Transformed dialogue data
            
        Returns:
            List of dialogue turns for evaluation
        """
        evaluation_turns = []
        
        # Handle transformed dialogue structure
        if 'transformed_turns' in dialogue_data:
            for turn in dialogue_data['transformed_turns']:
                if turn.get('speaker') == 'USER':  # Focus on user utterances for personality evaluation
                    evaluation_turns.append({
                        'turn_index': turn.get('turn_index', 0),
                        'utterance': turn.get('transformed_utterance', turn.get('utterance', '')),
                        'speaker': 'USER'
                    })
        
        # Handle direct turns structure
        elif 'turns' in dialogue_data:
            for i, turn in enumerate(dialogue_data['turns']):
                if isinstance(turn, dict):
                    speaker = turn.get('speaker', 'USER' if i % 2 == 0 else 'SYSTEM')
                    if speaker == 'USER':
                        evaluation_turns.append({
                            'turn_index': i,
                            'utterance': turn.get('utterance', turn.get('text', '')),
                            'speaker': 'USER'
                        })
        
        # Handle list of utterances
        elif isinstance(dialogue_data, list):
            for i, turn in enumerate(dialogue_data):
                if i % 2 == 0:  # Assume user turns are even-indexed
                    utterance = turn if isinstance(turn, str) else turn.get('utterance', '')
                    evaluation_turns.append({
                        'turn_index': i,
                        'utterance': utterance,
                        'speaker': 'USER'
                    })
        
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
        "openness": 0.X,
        "conscientiousness": 0.X,
        "extraversion": 0.X,
        "agreeableness": 0.X,
        "neuroticism": 0.X
    }},
    "detailed_analysis": {{
        "openness": {{
            "score": 0.X,
            "evidence": ["Specific examples from dialogue supporting this score"],
            "key_indicators": ["Main linguistic patterns observed"],
            "confidence": "High/Medium/Low assessment confidence"
        }},
        "conscientiousness": {{
            "score": 0.X,
            "evidence": ["Specific examples from dialogue supporting this score"],
            "key_indicators": ["Main linguistic patterns observed"],
            "confidence": "High/Medium/Low assessment confidence"
        }},
        "extraversion": {{
            "score": 0.X,
            "evidence": ["Specific examples from dialogue supporting this score"],
            "key_indicators": ["Main linguistic patterns observed"],
            "confidence": "High/Medium/Low assessment confidence"
        }},
        "agreeableness": {{
            "score": 0.X,
            "evidence": ["Specific examples from dialogue supporting this score"],
            "key_indicators": ["Main linguistic patterns observed"],
            "confidence": "High/Medium/Low assessment confidence"
        }},
        "neuroticism": {{
            "score": 0.X,
            "evidence": ["Specific examples from dialogue supporting this score"],
            "key_indicators": ["Main linguistic patterns observed"],
            "confidence": "High/Medium/Low assessment confidence"
        }}
    }},
    "overall_assessment": {{
        "dominant_traits": ["List 2-3 most prominent personality characteristics"],
        "personality_summary": "Brief description of overall personality profile",
        "authenticity_assessment": "Evaluation of whether personality expression feels natural vs. artificial",
        "communication_style": "Description of overall communication approach and style"
    }},
    "evaluation_metadata": {{
        "utterances_analyzed": X,
        "evaluation_confidence": "Overall confidence in assessment",
        "limitations": ["Any factors that might affect assessment accuracy"]
    }}
}}

Focus on providing accurate, evidence-based assessments that reflect genuine psychological insight rather than superficial pattern matching."""

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
        try:
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
                
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            self.logger.warning(f"Failed to parse evaluation response: {str(e)}")
            return self._extract_evaluation_from_text(response_content)
    
    def _extract_evaluation_from_text(self, text: str) -> Dict[str, Any]:
        """
        Extract evaluation information from free text when JSON parsing fails
        
        Args:
            text: Raw evaluation text
            
        Returns:
            Basic evaluation structure
        """
        # Try to extract personality scores from text
        big_five_scores = {}
        
        # Look for personality dimension mentions and scores
        dimensions = ['openness', 'conscientiousness', 'extraversion', 'agreeableness', 'neuroticism']
        
        for dimension in dimensions:
            # Look for patterns like "openness: 0.7" or "Openness = 0.65"
            import re
            patterns = [
                rf"{dimension}[:\s=]+([0-9]\.[0-9]+)",
                rf"{dimension.capitalize()}[:\s=]+([0-9]\.[0-9]+)"
            ]
            
            for pattern in patterns:
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    try:
                        score = float(match.group(1))
                        if 0.0 <= score <= 1.0:
                            big_five_scores[dimension] = score
                            break
                    except ValueError:
                        continue
        
        # If no scores found, provide moderate defaults
        if not big_five_scores:
            big_five_scores = {dim: 0.5 for dim in dimensions}
        
        return {
            'big_five_scores': big_five_scores,
            'detailed_analysis': {
                'extraction_note': 'Scores extracted from text analysis'
            },
            'overall_assessment': {
                'personality_summary': 'Moderate personality profile - detailed analysis unavailable'
            },
            'evaluation_metadata': {
                'utterances_analyzed': 'Unknown',
                'evaluation_confidence': 'Low - text extraction fallback',
                'limitations': ['JSON parsing failed, used text extraction']
            },
            'evaluator': self.agent_name,
            'evaluation_success': False
        }
    
    def _calculate_transformation_accuracy(self, 
                                         original_big5: Dict[str, float],
                                         evaluated_big5: Dict[str, float]) -> Dict[str, float]:
        """
        Calculate accuracy metrics between intended and evaluated personality
        
        Args:
            original_big5: Intended personality scores
            evaluated_big5: Evaluated personality scores
            
        Returns:
            Accuracy metrics for each dimension
        """
        accuracy_metrics = {}
        
        for dimension in ['openness', 'conscientiousness', 'extraversion', 'agreeableness', 'neuroticism']:
            original_score = original_big5.get(dimension.upper()[0], original_big5.get(dimension, 0.5))
            evaluated_score = evaluated_big5.get(dimension, 0.5)
            
            # Calculate absolute difference (lower is better)
            absolute_difference = abs(original_score - evaluated_score)
            
            # Convert to accuracy score (0-1, higher is better)
            accuracy = 1.0 - absolute_difference
            
            # Calculate correlation direction (positive if both high/low, negative if opposite)
            correlation = 1.0 if (original_score - 0.5) * (evaluated_score - 0.5) >= 0 else -1.0
            
            accuracy_metrics[dimension] = {
                'accuracy_score': max(0.0, accuracy),
                'absolute_difference': absolute_difference,
                'correlation_direction': correlation,
                'original_score': original_score,
                'evaluated_score': evaluated_score
            }
        
        return accuracy_metrics
    
    def _assess_overall_quality(self, accuracy_metrics: Dict[str, Dict[str, float]]) -> Dict[str, Any]:
        """
        Assess overall transformation quality based on accuracy metrics
        
        Args:
            accuracy_metrics: Per-dimension accuracy metrics
            
        Returns:
            Overall quality assessment
        """
        # Calculate average accuracy
        accuracy_scores = [metrics['accuracy_score'] for metrics in accuracy_metrics.values()]
        average_accuracy = sum(accuracy_scores) / len(accuracy_scores)
        
        # Calculate correlation consistency
        correlations = [metrics['correlation_direction'] for metrics in accuracy_metrics.values()]
        positive_correlations = sum(1 for c in correlations if c > 0)
        correlation_consistency = positive_correlations / len(correlations)
        
        # Determine quality level
        if average_accuracy >= 0.8 and correlation_consistency >= 0.8:
            quality_level = 'Excellent'
            quality_description = 'Personality traits accurately reflected in dialogue'
        elif average_accuracy >= 0.6 and correlation_consistency >= 0.6:
            quality_level = 'Good'
            quality_description = 'Personality traits generally well reflected with minor discrepancies'
        elif average_accuracy >= 0.4 and correlation_consistency >= 0.4:
            quality_level = 'Fair'
            quality_description = 'Some personality traits reflected, significant room for improvement'
        else:
            quality_level = 'Poor'
            quality_description = 'Personality traits poorly reflected in dialogue transformation'
        
        return {
            'quality_level': quality_level,
            'quality_description': quality_description,
            'average_accuracy': average_accuracy,
            'correlation_consistency': correlation_consistency,
            'best_dimensions': [dim for dim, metrics in accuracy_metrics.items() if metrics['accuracy_score'] >= 0.7],
            'improvement_needed': [dim for dim, metrics in accuracy_metrics.items() if metrics['accuracy_score'] < 0.5]
        }
    
    def _create_fallback_evaluation(self) -> Dict[str, Any]:
        """Create basic fallback evaluation when LLM call fails"""
        return {
            'big_five_scores': {
                'openness': 0.5,
                'conscientiousness': 0.5,
                'extraversion': 0.5,
                'agreeableness': 0.5,
                'neuroticism': 0.5
            },
            'detailed_analysis': {
                'note': 'Evaluation failed - fallback scores provided'
            },
            'overall_assessment': {
                'personality_summary': 'Unable to assess personality due to evaluation failure',
                'authenticity_assessment': 'Assessment unavailable'
            },
            'evaluation_metadata': {
                'evaluation_confidence': 'None - fallback used',
                'limitations': ['LLM evaluation failed', 'Fallback neutral scores provided']
            },
            'evaluator': self.agent_name,
            'evaluation_success': False
        }
