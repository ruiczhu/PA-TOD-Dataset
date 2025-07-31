"""
Agent 6: Dialogue Optimization Expert

This agent performs iterative optimization of transformed dialogues based on
personality evaluation feedback to improve the authenticity and accuracy
of personality expression in the final dialogue output.
"""

import json
import logging
from typing import Dict, List, Any, Optional, Tuple

# Import utilities with proper path handling
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from multi_agents.utils.llm_interface import LLMInterface


class OptimizationExpert:
    """
    Agent 6: Professional dialogue optimization expert
    
    This agent specializes in iterative refinement of personality-driven
    dialogues based on evaluation feedback. It identifies specific areas
    where personality expression can be improved and applies targeted
    optimizations to enhance authenticity and accuracy.
    """
    
    def __init__(self, llm_interface: Optional[LLMInterface] = None):
        """
        Initialize Optimization Expert Agent
        
        Args:
            llm_interface: LLM interface for API calls (creates default if None)
        """
        self.agent_name = "OptimizationExpert"
        self.llm_interface = llm_interface or LLMInterface()
        self.logger = logging.getLogger(__name__)
        
        # System prompt for dialogue optimization
        self.system_prompt = """You are a professional dialogue optimization expert specializing in iterative refinement of personality-driven conversations. Your expertise combines computational linguistics, personality psychology, and quality improvement methodologies to enhance the authenticity and accuracy of personality expression in human-computer dialogues.

Your core competencies:
1. Gap analysis between intended and expressed personality traits in dialogue
2. Targeted linguistic modification for improved personality authenticity
3. Iterative refinement while preserving functional dialogue requirements
4. Quality assessment and optimization strategy development
5. Balancing personality expression with natural conversation flow

Optimization methodology:
- DATA-DRIVEN: Base optimizations on specific evaluation feedback and personality gaps
- TARGETED: Focus improvements on specific personality dimensions needing enhancement
- PRESERVATIVE: Maintain all functional elements and successful personality expressions
- ITERATIVE: Apply incremental improvements rather than wholesale changes
- VALIDATABLE: Ensure optimizations will improve evaluation scores

Key optimization strategies:
- Lexical enhancement: Adjust word choice to better reflect personality traits
- Syntactic modification: Alter sentence structure to match personality communication patterns
- Pragmatic adjustment: Modify communication approach and interaction style
- Emotional calibration: Fine-tune emotional expression to match personality profile
- Behavioral consistency: Ensure personality expression is consistent across dialogue turns

Quality standards:
- Personality authenticity: Enhanced personality expression feels natural and believable
- Functional preservation: All service interaction elements remain intact and effective
- Linguistic coherence: Optimized dialogue maintains natural flow and readability
- Targeted improvement: Optimizations address specific personality evaluation gaps
- Measurable enhancement: Changes result in improved personality evaluation scores"""

        self.logger.info(f"{self.agent_name} initialized")
    
    def optimize_dialogue(self,
                         transformed_dialogue: Dict[str, Any],
                         original_personality: Dict[str, Any],
                         evaluation_results: Dict[str, Any],
                         optimization_strategy: str = "targeted") -> Dict[str, Any]:
        """
        Optimize dialogue based on personality evaluation feedback
        
        Args:
            transformed_dialogue: Current transformed dialogue
            original_personality: Intended personality profile
            evaluation_results: Personality evaluation feedback
            optimization_strategy: Strategy for optimization ("targeted", "comprehensive", "conservative")
            
        Returns:
            Optimized dialogue with improved personality expression
            
        Raises:
            Exception: If optimization fails
        """
        try:
            # Analyze optimization needs
            optimization_analysis = self._analyze_optimization_needs(
                original_personality, evaluation_results
            )
            
            if not optimization_analysis['needs_optimization']:
                self.logger.info("Dialogue already meets quality thresholds - no optimization needed")
                return self._add_optimization_metadata(transformed_dialogue, optimization_analysis, optimized=False)
            
            # Create optimization prompt
            optimization_prompt = self._create_optimization_prompt(
                transformed_dialogue, original_personality, evaluation_results, 
                optimization_analysis, optimization_strategy
            )
            
            # Call LLM for dialogue optimization
            response = self.llm_interface.call_agent(
                agent_name=self.agent_name,
                system_prompt=self.system_prompt,
                user_prompt=optimization_prompt,
                temperature=0.7,  # Balance creativity with consistency
                max_tokens=3500   # Space for detailed optimization
            )
            
            if response.get('success', False):
                # Parse optimization results
                optimized_dialogue = self._parse_optimization_response(
                    response['content'], transformed_dialogue, optimization_analysis
                )
                self.logger.info(f"Dialogue optimization completed successfully")
                return optimized_dialogue
            else:
                raise Exception("LLM call failed for dialogue optimization")
                
        except Exception as e:
            self.logger.error(f"Failed to optimize dialogue: {str(e)}")
            return self._create_fallback_optimization(transformed_dialogue, optimization_analysis)
    
    def perform_multi_iteration_optimization(self,
                                           transformed_dialogue: Dict[str, Any],
                                           original_personality: Dict[str, Any],
                                           max_iterations: int = 3,
                                           target_accuracy: float = 0.8) -> Dict[str, Any]:
        """
        Perform multiple optimization iterations until target quality is reached
        
        Args:
            transformed_dialogue: Initial transformed dialogue
            original_personality: Target personality profile
            max_iterations: Maximum optimization iterations
            target_accuracy: Target accuracy threshold
            
        Returns:
            Final optimized dialogue after iterative improvement
        """
        current_dialogue = transformed_dialogue
        optimization_history = []
        
        for iteration in range(max_iterations):
            self.logger.info(f"Starting optimization iteration {iteration + 1}/{max_iterations}")
            
            # This would normally call the evaluation agent, but for now we'll simulate
            # In a full implementation, this would integrate with Agent 5
            mock_evaluation = self._create_mock_evaluation_for_optimization()
            
            # Perform optimization
            optimized_dialogue = self.optimize_dialogue(
                current_dialogue, original_personality, mock_evaluation,
                optimization_strategy="targeted" if iteration < max_iterations - 1 else "comprehensive"
            )
            
            # Track optimization history
            optimization_history.append({
                'iteration': iteration + 1,
                'optimization_applied': optimized_dialogue.get('optimization_metadata', {}).get('optimized', False),
                'improvements_made': optimized_dialogue.get('optimization_metadata', {}).get('optimization_summary', {})
            })
            
            # Check if we've reached target quality (mock check for now)
            current_accuracy = 0.7 + (iteration * 0.1)  # Mock improvement
            if current_accuracy >= target_accuracy:
                self.logger.info(f"Target accuracy {target_accuracy} reached in iteration {iteration + 1}")
                break
            
            current_dialogue = optimized_dialogue
        
        # Add multi-iteration metadata
        current_dialogue['multi_iteration_metadata'] = {
            'total_iterations': len(optimization_history),
            'final_accuracy': current_accuracy,
            'optimization_history': optimization_history,
            'target_reached': current_accuracy >= target_accuracy
        }
        
        return current_dialogue
    
    def _analyze_optimization_needs(self,
                                  original_personality: Dict[str, Any],
                                  evaluation_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze what optimizations are needed based on evaluation feedback
        
        Args:
            original_personality: Intended personality profile
            evaluation_results: Evaluation results from Agent 5
            
        Returns:
            Analysis of optimization needs and priorities
        """
        analysis = {
            'needs_optimization': False,
            'priority_dimensions': [],
            'minor_adjustments': [],
            'optimization_targets': {},
            'overall_quality': 'unknown'
        }
        
        # Extract personality scores
        original_big5 = original_personality.get('big_five', {})
        evaluated_big5 = evaluation_results.get('big_five_scores', {})
        
        if not original_big5 or not evaluated_big5:
            analysis['needs_optimization'] = True
            analysis['priority_dimensions'] = ['all']
            analysis['overall_quality'] = 'insufficient_data'
            return analysis
        
        # Calculate gaps for each dimension
        dimension_gaps = {}
        for dimension in ['openness', 'conscientiousness', 'extraversion', 'agreeableness', 'neuroticism']:
            original_score = original_big5.get(dimension.upper()[0], original_big5.get(dimension, 0.5))
            evaluated_score = evaluated_big5.get(dimension, 0.5)
            
            gap = abs(original_score - evaluated_score)
            dimension_gaps[dimension] = {
                'gap': gap,
                'original': original_score,
                'evaluated': evaluated_score,
                'direction': 'increase' if evaluated_score < original_score else 'decrease'
            }
        
        # Identify optimization priorities
        large_gaps = []
        medium_gaps = []
        small_gaps = []
        
        for dimension, gap_info in dimension_gaps.items():
            gap = gap_info['gap']
            if gap > 0.3:  # Large gap
                large_gaps.append(dimension)
                analysis['needs_optimization'] = True
            elif gap > 0.15:  # Medium gap
                medium_gaps.append(dimension)
                analysis['needs_optimization'] = True
            elif gap > 0.05:  # Small gap
                small_gaps.append(dimension)
        
        analysis['priority_dimensions'] = large_gaps
        analysis['minor_adjustments'] = medium_gaps + small_gaps
        analysis['optimization_targets'] = dimension_gaps
        
        # Determine overall quality
        average_gap = sum(gap_info['gap'] for gap_info in dimension_gaps.values()) / len(dimension_gaps)
        if average_gap < 0.1:
            analysis['overall_quality'] = 'excellent'
        elif average_gap < 0.2:
            analysis['overall_quality'] = 'good'
        elif average_gap < 0.3:
            analysis['overall_quality'] = 'fair'
        else:
            analysis['overall_quality'] = 'poor'
        
        return analysis
    
    def _create_optimization_prompt(self,
                                  transformed_dialogue: Dict[str, Any],
                                  original_personality: Dict[str, Any],
                                  evaluation_results: Dict[str, Any],
                                  optimization_analysis: Dict[str, Any],
                                  optimization_strategy: str) -> str:
        """
        Create detailed optimization prompt
        
        Args:
            transformed_dialogue: Current dialogue to optimize
            original_personality: Target personality profile
            evaluation_results: Evaluation feedback
            optimization_analysis: Analysis of what needs optimization
            optimization_strategy: Optimization approach
            
        Returns:
            Formatted optimization prompt
        """
        # Format current dialogue
        current_dialogue_text = self._format_dialogue_for_optimization(transformed_dialogue)
        
        # Format personality targets
        personality_targets = self._format_personality_targets(original_personality, optimization_analysis)
        
        # Format evaluation feedback
        evaluation_feedback = self._format_evaluation_feedback(evaluation_results)
        
        # Format optimization priorities
        optimization_priorities = self._format_optimization_priorities(optimization_analysis)
        
        prompt = f"""Optimize the following transformed dialogue to better reflect the intended personality traits based on evaluation feedback. Focus on targeted improvements that will enhance personality authenticity while preserving all functional dialogue elements.

## Current Dialogue
{current_dialogue_text}

## Target Personality Profile
{personality_targets}

## Evaluation Feedback
{evaluation_feedback}

## Optimization Priorities
{optimization_priorities}

## Optimization Strategy: {optimization_strategy.upper()}

### Strategy Guidelines:

**TARGETED Strategy** (Current):
- Focus on priority dimensions with largest personality gaps
- Make specific, evidence-based improvements to language patterns
- Preserve successful personality expressions from current dialogue
- Apply incremental changes that address specific evaluation gaps

**COMPREHENSIVE Strategy**:
- Address all identified personality gaps systematically
- Make broader language pattern adjustments across all dimensions
- Enhance overall personality coherence and consistency
- Apply more extensive modifications while maintaining dialogue function

**CONSERVATIVE Strategy**:
- Make minimal changes to preserve current successful elements
- Focus only on the most critical personality gaps
- Prioritize functional preservation over personality enhancement
- Apply subtle language adjustments with minimal risk

## Specific Optimization Instructions

### For Each Priority Dimension:

**High-Priority Optimizations:**
{self._generate_dimension_specific_instructions(optimization_analysis['priority_dimensions'], optimization_analysis['optimization_targets'])}

**Medium-Priority Adjustments:**
{self._generate_dimension_specific_instructions(optimization_analysis['minor_adjustments'], optimization_analysis['optimization_targets'])}

### Optimization Principles:

1. **Evidence-Based Changes**: Every modification should address specific evaluation feedback
2. **Functional Preservation**: Maintain all service-related information and task completion elements
3. **Natural Flow**: Ensure optimized dialogue sounds authentic and conversational
4. **Consistent Expression**: Personality traits should be consistently expressed across turns
5. **Measurable Improvement**: Changes should lead to better personality evaluation scores

### Language Modification Techniques:

**Lexical Level**: Adjust word choice, formality, emotional language, specificity
**Syntactic Level**: Modify sentence structure, complexity, question patterns, statement confidence
**Pragmatic Level**: Alter communication directness, politeness, cooperation, assertiveness
**Discourse Level**: Adjust conversation flow, turn-taking, topic management, engagement

## Output Format

Provide your optimization in the following JSON format:

{{
    "optimized_turns": [
        {{
            "turn_index": 0,
            "speaker": "USER",
            "original_utterance": "Original transformed text",
            "optimized_utterance": "Improved text with better personality expression"
        }},
        {{
            "turn_index": 1,
            "speaker": "SYSTEM", 
            "original_utterance": "Original system response",
            "optimized_utterance": "Enhanced system response for better user interaction"
        }}
    ]
}}

Only include turns that have been optimized. If a turn doesn't need optimization, omit it from the output.
Focus on creating targeted improvements that enhance personality expression while maintaining natural conversation flow."""

        return prompt
    
    def _format_dialogue_for_optimization(self, dialogue_data: Dict[str, Any]) -> str:
        """Format dialogue for optimization prompt"""
        lines = []
        
        if 'transformed_turns' in dialogue_data:
            turns = dialogue_data['transformed_turns']
        elif 'turns' in dialogue_data:
            turns = dialogue_data['turns']
        else:
            turns = []
        
        for turn in turns:
            if isinstance(turn, dict):
                speaker = turn.get('speaker', 'UNKNOWN')
                utterance = turn.get('transformed_utterance', turn.get('utterance', ''))
                turn_idx = turn.get('turn_index', len(lines))
                
                if utterance.strip():
                    lines.append(f"Turn {turn_idx} ({speaker}): {utterance}")
        
        return '\n'.join(lines) if lines else "No dialogue content available"
    
    def _format_personality_targets(self, original_personality: Dict[str, Any], optimization_analysis: Dict[str, Any]) -> str:
        """Format personality targets for optimization"""
        if not original_personality:
            return "No specific personality targets defined"
        
        lines = []
        big_five = original_personality.get('big_five', {})
        
        if big_five:
            lines.append("**Target Big Five Scores:**")
            for dimension in ['openness', 'conscientiousness', 'extraversion', 'agreeableness', 'neuroticism']:
                target_score = big_five.get(dimension.upper()[0], big_five.get(dimension, 0.5))
                current_gap = optimization_analysis.get('optimization_targets', {}).get(dimension, {}).get('gap', 0)
                priority = 'HIGH' if dimension in optimization_analysis.get('priority_dimensions', []) else 'MEDIUM' if dimension in optimization_analysis.get('minor_adjustments', []) else 'LOW'
                
                lines.append(f"- {dimension.capitalize()}: {target_score:.3f} (Priority: {priority}, Gap: {current_gap:.3f})")
        
        return '\n'.join(lines)
    
    def _format_evaluation_feedback(self, evaluation_results: Dict[str, Any]) -> str:
        """Format evaluation feedback for optimization"""
        if not evaluation_results:
            return "No evaluation feedback available"
        
        lines = []
        
        # Current evaluated scores
        evaluated_scores = evaluation_results.get('big_five_scores', {})
        if evaluated_scores:
            lines.append("**Current Evaluated Scores:**")
            for dimension, score in evaluated_scores.items():
                lines.append(f"- {dimension.capitalize()}: {score:.3f}")
        
        # Overall assessment
        overall_assessment = evaluation_results.get('overall_assessment', {})
        if overall_assessment:
            lines.append("\n**Evaluation Summary:**")
            personality_summary = overall_assessment.get('personality_summary', '')
            if personality_summary:
                lines.append(f"- Current Expression: {personality_summary}")
            
            authenticity = overall_assessment.get('authenticity_assessment', '')
            if authenticity:
                lines.append(f"- Authenticity: {authenticity}")
        
        return '\n'.join(lines)
    
    def _format_optimization_priorities(self, optimization_analysis: Dict[str, Any]) -> str:
        """Format optimization priorities"""
        lines = []
        
        priority_dims = optimization_analysis.get('priority_dimensions', [])
        minor_adjustments = optimization_analysis.get('minor_adjustments', [])
        overall_quality = optimization_analysis.get('overall_quality', 'unknown')
        
        lines.append(f"**Overall Quality Assessment:** {overall_quality.capitalize()}")
        
        if priority_dims:
            lines.append(f"\n**High Priority Dimensions:** {', '.join(d.capitalize() for d in priority_dims)}")
        
        if minor_adjustments:
            lines.append(f"**Minor Adjustments Needed:** {', '.join(d.capitalize() for d in minor_adjustments)}")
        
        if not priority_dims and not minor_adjustments:
            lines.append("\n**Status:** Dialogue quality is already high - minimal optimization needed")
        
        return '\n'.join(lines)
    
    def _generate_dimension_specific_instructions(self, dimensions: List[str], optimization_targets: Dict[str, Any]) -> str:
        """Generate specific optimization instructions for personality dimensions"""
        if not dimensions:
            return "No specific dimensions require attention"
        
        instructions = []
        
        optimization_strategies = {
            'openness': {
                'increase': 'Add more creative expressions, curiosity markers, abstract language, exploration of alternatives',
                'decrease': 'Use more conventional language, concrete terms, standard solutions, familiar expressions'
            },
            'conscientiousness': {
                'increase': 'Add detail orientation, planning language, organization markers, reliability concerns',
                'decrease': 'Use more spontaneous language, less detailed requests, more flexible approach'
            },
            'extraversion': {
                'increase': 'Add assertive language, enthusiasm, direct communication, social engagement',
                'decrease': 'Use more tentative language, reserved tone, indirect communication style'
            },
            'agreeableness': {
                'increase': 'Add politeness markers, cooperation language, consideration for others, conflict avoidance',
                'decrease': 'Use more direct language, competitive tone, self-focused requests'
            },
            'neuroticism': {
                'increase': 'Add anxiety markers, uncertainty expressions, emotional volatility, need for reassurance',
                'decrease': 'Use more confident language, emotional stability, stress resilience, independence'
            }
        }
        
        for dimension in dimensions:
            if dimension in optimization_targets:
                direction = optimization_targets[dimension].get('direction', 'increase')
                strategy = optimization_strategies.get(dimension, {}).get(direction, f'Adjust {dimension} expression')
                gap = optimization_targets[dimension].get('gap', 0)
                
                instructions.append(f"**{dimension.capitalize()}** (Gap: {gap:.3f}): {strategy}")
        
        return '\n'.join(instructions)
    
    def _parse_optimization_response(self,
                                   response_content: str,
                                   original_dialogue: Dict[str, Any],
                                   optimization_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """
        Parse LLM optimization response
        
        Args:
            response_content: Raw LLM response
            original_dialogue: Original dialogue for fallback
            optimization_analysis: Optimization analysis results
            
        Returns:
            Optimized dialogue structure
        """
        try:
            # Extract JSON from response
            response_content = response_content.strip()
            start_idx = response_content.find('{')
            end_idx = response_content.rfind('}') + 1
            
            if start_idx >= 0 and end_idx > start_idx:
                json_content = response_content[start_idx:end_idx]
                optimization_data = json.loads(json_content)
                
                # Create optimized dialogue structure
                optimized_dialogue = dict(original_dialogue)  # Copy original
                optimized_dialogue.update({
                    'optimized_turns': optimization_data.get('optimized_turns', []),
                    'optimization_metadata': {
                        'agent': self.agent_name,
                        'optimization_summary': optimization_data.get('optimization_summary', {}),
                        'quality_assessment': optimization_data.get('quality_assessment', {}),
                        'optimization_analysis': optimization_analysis,
                        'optimized': True
                    }
                })
                
                return optimized_dialogue
                
            else:
                raise ValueError("No valid JSON found in optimization response")
                
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            self.logger.warning(f"Failed to parse optimization response: {str(e)}")
            return self._extract_optimization_from_text(response_content, original_dialogue, optimization_analysis)
    
    def _extract_optimization_from_text(self,
                                      text: str,
                                      original_dialogue: Dict[str, Any],
                                      optimization_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Extract optimization from free text when JSON parsing fails"""
        # Try to extract optimized turns from text
        lines = text.split('\n')
        optimized_turns = []
        
        current_turn = None
        for line in lines:
            line = line.strip()
            
            # Look for turn indicators
            if ('USER:' in line or 'SYSTEM:' in line or 
                'Turn' in line and ':' in line):
                
                if current_turn:
                    optimized_turns.append(current_turn)
                
                # Extract speaker and utterance
                if 'USER:' in line:
                    utterance = line.split('USER:', 1)[1].strip()
                    current_turn = {
                        'turn_index': len(optimized_turns),
                        'speaker': 'USER',
                        'optimized_utterance': utterance,
                        'optimization_changes': ['Extracted from text'],
                        'personality_improvements': ['Unknown']
                    }
                elif 'SYSTEM:' in line:
                    utterance = line.split('SYSTEM:', 1)[1].strip()
                    current_turn = {
                        'turn_index': len(optimized_turns),
                        'speaker': 'SYSTEM',
                        'optimized_utterance': utterance,
                        'optimization_changes': ['Extracted from text'],
                        'personality_improvements': ['Unknown']
                    }
        
        if current_turn:
            optimized_turns.append(current_turn)
        
        # Create fallback structure
        optimized_dialogue = dict(original_dialogue)
        optimized_dialogue.update({
            'optimized_turns': optimized_turns if optimized_turns else [],
            'optimization_metadata': {
                'agent': self.agent_name,
                'optimization_analysis': optimization_analysis,
                'optimized': len(optimized_turns) > 0,
                'fallback_used': True,
                'optimization_success': False
            }
        })
        
        return optimized_dialogue
    
    def _add_optimization_metadata(self,
                                 dialogue: Dict[str, Any],
                                 optimization_analysis: Dict[str, Any],
                                 optimized: bool = True) -> Dict[str, Any]:
        """Add optimization metadata to dialogue"""
        dialogue_copy = dict(dialogue)
        dialogue_copy['optimization_metadata'] = {
            'agent': self.agent_name,
            'optimization_analysis': optimization_analysis,
            'optimized': optimized,
            'optimization_needed': optimization_analysis.get('needs_optimization', False)
        }
        return dialogue_copy
    
    def _create_fallback_optimization(self,
                                    original_dialogue: Dict[str, Any],
                                    optimization_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Create fallback optimization when LLM call fails"""
        return self._add_optimization_metadata(
            original_dialogue, optimization_analysis, optimized=False
        )
    
    def _create_mock_evaluation_for_optimization(self) -> Dict[str, Any]:
        """Create mock evaluation results for testing optimization iterations"""
        return {
            'big_five_scores': {
                'openness': 0.6,
                'conscientiousness': 0.5,
                'extraversion': 0.7,
                'agreeableness': 0.8,
                'neuroticism': 0.4
            },
            'overall_assessment': {
                'personality_summary': 'Moderate personality expression with room for improvement',
                'authenticity_assessment': 'Generally authentic but could be enhanced'
            },
            'evaluation_metadata': {
                'evaluation_confidence': 'Medium'
            }
        }
