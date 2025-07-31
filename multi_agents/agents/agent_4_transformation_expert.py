"""
Agent 4: Dialogue Transformation Expert

This agent transforms original dialogues by integrating personality traits,
user state, and scenario context to create more realistic and natural
human-like conversations that reflect the user's psychological and
situational characteristics.
"""

import json
import logging
from typing import Dict, List, Any, Optional, Tuple

# Import utilities with proper path handling
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from multi_agents.utils.llm_interface import LLMInterface
from multi_agents.utils.data_structures import EnhancedDialogue


class TransformationExpert:
    """
    Agent 4: Professional dialogue transformation expert
    
    This agent specializes in transforming structured dialogue data into
    natural, personality-driven conversations that authentically reflect
    the user's psychological state, personality traits, and situational
    context while maintaining functional equivalence.
    """
    
    def __init__(self, llm_interface: Optional[LLMInterface] = None):
        """
        Initialize Transformation Expert Agent
        
        Args:
            llm_interface: LLM interface for API calls (creates default if None)
        """
        self.agent_name = "TransformationExpert"
        self.llm_interface = llm_interface or LLMInterface()
        self.logger = logging.getLogger(__name__)
        
        # System prompt for dialogue transformation
        self.system_prompt = """You are a professional dialogue transformation expert specializing in creating natural, psychologically authentic human-computer conversations. Your expertise combines computational linguistics, personality psychology, and human factors engineering to transform structured dialogue data into believable human interactions.

Your core competencies:
1. Personality-driven language adaptation based on Big Five personality traits
2. Emotional state integration into conversational tone and word choice
3. Situational context reflection in communication style and urgency
4. Natural language flow while preserving functional dialogue requirements
5. Cultural and demographic appropriateness in expression patterns

Key transformation principles:
- Maintain all functional elements (intents, slot values, essential information)
- Adapt communication style to reflect personality traits authentically
- Integrate emotional and cognitive state into natural speech patterns
- Reflect situational urgency and environmental context appropriately
- Ensure dialogue feels genuinely human while accomplishing task objectives

Quality standards:
- Psychological authenticity: Language patterns consistent with personality profile
- Functional preservation: All original service interaction elements maintained
- Natural flow: Conversations sound like real human speech, not scripted responses
- Contextual appropriateness: Tone and style match the scenario and user state
- Believability: Transformed dialogue could pass as authentic human communication"""

        self.logger.info(f"{self.agent_name} initialized")
    
    def transform_dialogue(self,
                          original_dialogue: Dict[str, Any],
                          personality_data: Dict[str, Any],
                          user_state: Dict[str, Any],
                          scenario_info: Dict[str, Any],
                          user_profile: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Transform dialogue with personality and state integration
        
        Args:
            original_dialogue: Original dialogue data structure
            personality_data: Personality trait information
            user_state: Current user state simulation
            scenario_info: Scenario context information
            user_profile: Optional user profile information
            
        Returns:
            Transformed dialogue with personality integration
            
        Raises:
            Exception: If transformation fails
        """
        try:
            # Extract dialogue turns and metadata
            dialogue_turns = self._extract_dialogue_turns(original_dialogue)
            
            if not dialogue_turns:
                raise ValueError("No dialogue turns found in original dialogue")
            
            # Create transformation prompt with all context
            transformation_prompt = self._create_transformation_prompt(
                dialogue_turns, personality_data, user_state, scenario_info, user_profile
            )
            
            # Call LLM to transform dialogue
            response = self.llm_interface.call_agent(
                agent_name=self.agent_name,
                system_prompt=self.system_prompt,
                user_prompt=transformation_prompt,
                temperature=0.9,  # Higher creativity for natural language variation
                max_tokens=3000   # More space for complete dialogue transformation
            )
            
            if response.get('success', False):
                # Parse transformed dialogue
                transformed_dialogue = self._parse_transformation_response(
                    response['content'], original_dialogue
                )
                self.logger.info(f"Dialogue transformed successfully with {len(transformed_dialogue.get('turns', []))} turns")
                return transformed_dialogue
            else:
                raise Exception("LLM call failed for dialogue transformation")
                
        except Exception as e:
            self.logger.error(f"Failed to transform dialogue: {str(e)}")
            # Return original dialogue with minimal enhancement as fallback
            return self._create_fallback_transformation(original_dialogue, personality_data, user_state)
    
    def _extract_dialogue_turns(self, dialogue_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Extract dialogue turns from various possible data structures
        
        Args:
            dialogue_data: Original dialogue in various formats
            
        Returns:
            List of dialogue turns with speaker and utterance information
        """
        turns = []
        
        # Handle EnhancedDialogue objects
        if hasattr(dialogue_data, 'turns'):
            return dialogue_data.turns
        
        # Handle dictionary with 'turns' key
        if isinstance(dialogue_data, dict) and 'turns' in dialogue_data:
            raw_turns = dialogue_data['turns']
        elif isinstance(dialogue_data, dict) and 'dialogue' in dialogue_data:
            raw_turns = dialogue_data['dialogue']
        elif isinstance(dialogue_data, list):
            raw_turns = dialogue_data
        else:
            # Try to find turns in any nested structure
            raw_turns = self._find_turns_in_structure(dialogue_data)
        
        # Process raw turns into standard format
        for i, turn in enumerate(raw_turns):
            if isinstance(turn, dict):
                # Extract speaker and utterance
                speaker = turn.get('speaker', turn.get('role', 'USER' if i % 2 == 0 else 'SYSTEM'))
                utterance = turn.get('utterance', turn.get('text', turn.get('content', '')))
                
                # Extract additional metadata
                turn_data = {
                    'speaker': speaker,
                    'utterance': utterance,
                    'turn_index': i,
                    'original_data': turn
                }
                
                # Include service-related information if available
                if 'frames' in turn:
                    turn_data['frames'] = turn['frames']
                if 'actions' in turn:
                    turn_data['actions'] = turn['actions']
                
                turns.append(turn_data)
            elif isinstance(turn, str):
                # Simple string turns - alternate speakers
                turns.append({
                    'speaker': 'USER' if i % 2 == 0 else 'SYSTEM',
                    'utterance': turn,
                    'turn_index': i,
                    'original_data': turn
                })
        
        return turns
    
    def _find_turns_in_structure(self, data: Any) -> List[Any]:
        """
        Recursively search for dialogue turns in complex structures
        
        Args:
            data: Complex data structure to search
            
        Returns:
            List of found dialogue turns
        """
        if isinstance(data, list):
            return data
        elif isinstance(data, dict):
            # Search for common dialogue keys
            for key in ['turns', 'dialogue', 'conversation', 'messages']:
                if key in data and isinstance(data[key], list):
                    return data[key]
            
            # Search for any list in the dictionary
            for value in data.values():
                if isinstance(value, list) and value:
                    # Check if list contains dialogue-like structures
                    first_item = value[0]
                    if isinstance(first_item, dict):
                        dialogue_keys = ['utterance', 'text', 'content', 'speaker', 'role']
                        if any(key in first_item for key in dialogue_keys):
                            return value
        
        return []
    
    def _create_transformation_prompt(self,
                                    dialogue_turns: List[Dict[str, Any]],
                                    personality_data: Dict[str, Any],
                                    user_state: Dict[str, Any],
                                    scenario_info: Dict[str, Any],
                                    user_profile: Dict[str, Any] = None) -> str:
        """
        Create comprehensive transformation prompt
        
        Args:
            dialogue_turns: Original dialogue turns
            personality_data: Personality trait information
            user_state: User state simulation
            scenario_info: Scenario context
            user_profile: Optional user profile
            
        Returns:
            Formatted transformation prompt
        """
        # Format original dialogue
        original_dialogue_text = self._format_original_dialogue(dialogue_turns)
        
        # Format personality context
        personality_context = self._format_personality_for_transformation(personality_data)
        
        # Format user state context
        state_context = self._format_user_state_for_transformation(user_state)
        
        # Format scenario context
        scenario_context = self._format_scenario_for_transformation(scenario_info)
        
        # Format user profile if available
        profile_context = ""
        if user_profile:
            profile_context = f"\n### User Profile Context\n{self._format_profile_for_transformation(user_profile)}"
        
        prompt = f"""Transform the following structured dialogue into a natural, psychologically authentic conversation that reflects the user's personality traits, current emotional/cognitive state, and situational context. The transformation should maintain all functional elements while making the conversation sound genuinely human.

## Original Dialogue
{original_dialogue_text}

## User Psychological Profile

### Personality Traits
{personality_context}

### Current User State
{state_context}

### Situational Context
{scenario_context}
{profile_context}

## Transformation Instructions

### Primary Objectives:
1. **Personality Integration**: Adapt language patterns, word choice, and communication style to authentically reflect the Big Five personality profile
2. **State Reflection**: Integrate current emotional, cognitive, and physical state into natural speech patterns
3. **Contextual Adaptation**: Ensure dialogue tone and urgency match the scenario and environmental factors
4. **Functional Preservation**: Maintain all service-related information, intents, and task completion elements
5. **Natural Flow**: Create conversations that sound like authentic human speech, not scripted interactions

### Specific Transformation Guidelines:

**For USER turns:**
- Adapt communication style based on personality traits (e.g., high openness = more exploratory language; high conscientiousness = more structured requests)
- Reflect current emotional state in tone and word choice (e.g., anxiety = more hesitant language; confidence = more direct requests)
- Integrate physical/cognitive state (e.g., time pressure = more urgent language; high cognitive load = simpler expression)
- Match scenario context (e.g., formal setting = more polite language; urgent situation = more direct communication)

**For SYSTEM turns:**
- Maintain professional service quality
- Adapt response style to user's communication patterns for better rapport
- Include appropriate acknowledgment of user's state when relevant

### Language Pattern Examples by Personality:

**High Extraversion**: More expressive, direct, enthusiastic language
- "I'd love to book a table!" vs "I need a reservation"

**High Agreeableness**: More polite, cooperative, apologetic language
- "I'm so sorry to bother you, but..." vs "I need help with..."

**High Conscientiousness**: More structured, detailed, organized requests
- "I'd like to schedule an appointment for 2 PM on Tuesday, if that works with your availability" vs "Can I get an appointment?"

**High Neuroticism**: More uncertain, anxious, seeking reassurance
- "I'm not sure if I'm doing this right, but..." vs "I want to..."

**High Openness**: More exploratory, curious, creative language
- "What options do you have available?" vs "Give me the standard option"

### State Integration Examples:

**High Stress**: Shorter sentences, more direct, occasionally repetitive
**Low Confidence**: More questioning, seeking confirmation
**Time Pressure**: More urgent, focused language
**Physical Discomfort**: Shorter interactions, more efficient communication

## Output Format

Provide the transformed dialogue in the following JSON format:

{{
    "transformed_turns": [
        {{
            "turn_index": 0,
            "speaker": "USER",
            "original_utterance": "Original text",
            "transformed_utterance": "Transformed text reflecting personality and state",
            "transformation_notes": "Brief explanation of key changes made"
        }},
        {{
            "turn_index": 1,
            "speaker": "SYSTEM",
            "original_utterance": "Original system response",
            "transformed_utterance": "Adapted system response for better user rapport",
            "transformation_notes": "Adaptation rationale"
        }}
    ],
    "transformation_summary": {{
        "personality_adaptations": ["List of personality-based changes made"],
        "state_integrations": ["List of user state reflections added"],
        "contextual_adjustments": ["List of scenario-based adaptations"],
        "preserved_functions": ["List of maintained functional elements"]
    }}
}}

Ensure every transformed utterance feels naturally human while accomplishing the same functional goals as the original dialogue."""

        return prompt
    
    def _format_original_dialogue(self, dialogue_turns: List[Dict[str, Any]]) -> str:
        """Format original dialogue for prompt"""
        lines = []
        for turn in dialogue_turns:
            speaker = turn.get('speaker', 'UNKNOWN')
            utterance = turn.get('utterance', '')
            turn_idx = turn.get('turn_index', 0)
            
            lines.append(f"Turn {turn_idx} ({speaker}): {utterance}")
            
            # Add any service action information
            if 'actions' in turn.get('original_data', {}):
                actions = turn['original_data']['actions']
                if actions:
                    lines.append(f"    [Service Actions: {actions}]")
        
        return '\n'.join(lines)
    
    def _format_personality_for_transformation(self, personality_data: Dict[str, Any]) -> str:
        """Format personality data for transformation prompt"""
        if not personality_data:
            return "No specific personality profile - use moderate, professional communication style"
        
        lines = []
        
        # Big Five traits with communication implications
        big_five = personality_data.get('big_five', {})
        if big_five:
            trait_implications = {
                'O': ('Openness', 'curiosity, creative expression, openness to new ideas'),
                'C': ('Conscientiousness', 'structured communication, detail orientation, reliability'),
                'E': ('Extraversion', 'assertive expression, social engagement, enthusiasm'),
                'A': ('Agreeableness', 'polite language, cooperation, consideration for others'),
                'N': ('Neuroticism', 'emotional expression, uncertainty, need for reassurance')
            }
            
            if isinstance(big_five, dict):
                for dim, score in big_five.items():
                    name, implications = trait_implications.get(dim, (dim, 'general trait'))
                    level = self._score_to_communication_level(score)
                    lines.append(f"**{name}** ({level}): {implications}")
        
        # Key facets that affect communication
        facets = personality_data.get('facets', {})
        if facets and isinstance(facets, dict):
            communication_facets = []
            for dimension, traits in facets.items():
                if isinstance(traits, dict):
                    for trait_code, score in traits.items():
                        try:
                            score_float = float(score)
                            if trait_code in ['N1', 'E3', 'A4', 'C5', 'O5'] and (score_float > 0.7 or score_float < 0.3):
                                facet_names = {
                                    'N1': 'Anxiety level',
                                    'E3': 'Assertiveness',
                                    'A4': 'Compliance/Cooperation',
                                    'C5': 'Self-discipline in communication',
                                    'O5': 'Intellectual engagement'
                                }
                                trait_name = facet_names.get(trait_code, trait_code)
                                level = 'Very High' if score_float > 0.7 else 'Very Low'
                                communication_facets.append(f"- {trait_name}: {level}")
                        except (ValueError, TypeError):
                            continue
            
            if communication_facets:
                lines.append("\n**Communication-Relevant Traits:**")
                lines.extend(communication_facets)
        
        return '\n'.join(lines) if lines else "Moderate personality profile"
    
    def _format_user_state_for_transformation(self, user_state: Dict[str, Any]) -> str:
        """Format user state for transformation prompt"""
        if not user_state:
            return "Standard user state - calm and focused"
        
        state_dict = user_state.__dict__ if hasattr(user_state, '__dict__') else user_state
        
        lines = [
            f"**Emotional State:** {state_dict.get('emotional_state', 'Neutral')}",
            f"**Stress Level:** {state_dict.get('stress_level', 'Moderate')}",
            f"**Patience Level:** {state_dict.get('patience_level', 'Moderate')}",
            f"**Confidence Level:** {state_dict.get('confidence_level', 'Moderate')}",
            f"**Current Mood:** {state_dict.get('current_mood', 'Professional')}",
            f"**Physical State:** {state_dict.get('physical_state', 'Alert')}",
            f"**Cognitive Load:** {state_dict.get('cognitive_load', 'Medium')}"
        ]
        
        situational_factors = state_dict.get('situational_factors', {})
        if situational_factors and isinstance(situational_factors, dict):
            lines.append("\n**Situational Factors:**")
            for factor, value in situational_factors.items():
                if factor != 'fallback':  # Skip technical flags
                    lines.append(f"- {factor.replace('_', ' ').title()}: {value}")
        
        return '\n'.join(lines)
    
    def _format_scenario_for_transformation(self, scenario_info: Dict[str, Any]) -> str:
        """Format scenario information for transformation prompt"""
        if not scenario_info:
            return "Standard service interaction scenario"
        
        scenario_dict = scenario_info.__dict__ if hasattr(scenario_info, '__dict__') else scenario_info
        
        lines = [
            f"**Service Environment:** {scenario_dict.get('location', 'Service location')}",
            f"**Timing:** {scenario_dict.get('time_of_day', 'Business hours')}",
            f"**Activity Level:** {scenario_dict.get('crowd_level', 'Moderate')}",
            f"**Urgency:** {scenario_dict.get('urgency_level', 'Moderate')}",
            f"**Complexity:** {scenario_dict.get('service_complexity', 'Medium')}"
        ]
        
        weather = scenario_dict.get('weather')
        if weather and weather != 'Moderate weather conditions':
            lines.append(f"**Environmental:** {weather}")
        
        social_context = scenario_dict.get('social_context')
        if social_context:
            lines.append(f"**Social Context:** {social_context}")
        
        return '\n'.join(lines)
    
    def _format_profile_for_transformation(self, user_profile: Dict[str, Any]) -> str:
        """Format user profile for transformation prompt"""
        profile_dict = user_profile.__dict__ if hasattr(user_profile, '__dict__') else user_profile
        
        lines = [
            f"**Demographics:** {profile_dict.get('age_range', 'Adult')}",
            f"**Background:** {profile_dict.get('occupation', 'Professional')}",
            f"**Education:** {profile_dict.get('education_level', 'College')}",
            f"**Tech Comfort:** {profile_dict.get('tech_savviness', 'Moderate')}",
            f"**Communication Preference:** {profile_dict.get('communication_style', 'Direct')}"
        ]
        
        return '\n'.join(lines)
    
    def _score_to_communication_level(self, score) -> str:
        """Convert personality score to communication impact level"""
        try:
            score_float = float(score)
            if score_float >= 0.8:
                return "Very High"
            elif score_float >= 0.6:
                return "High"
            elif score_float >= 0.4:
                return "Moderate"
            elif score_float >= 0.2:
                return "Low"
            else:
                return "Very Low"
        except (ValueError, TypeError):
            return "Moderate"  # Default fallback
    
    def _parse_transformation_response(self, response_content: str, original_dialogue: Dict[str, Any]) -> Dict[str, Any]:
        """
        Parse LLM transformation response
        
        Args:
            response_content: Raw LLM response
            original_dialogue: Original dialogue for fallback
            
        Returns:
            Structured transformed dialogue
        """
        try:
            # Extract JSON from response
            response_content = response_content.strip()
            start_idx = response_content.find('{')
            end_idx = response_content.rfind('}') + 1
            
            if start_idx >= 0 and end_idx > start_idx:
                json_content = response_content[start_idx:end_idx]
                transformation_data = json.loads(json_content)
                
                # Extract transformed turns
                transformed_turns = transformation_data.get('transformed_turns', [])
                
                # Create enhanced dialogue structure
                enhanced_dialogue = {
                    'dialogue_id': original_dialogue.get('dialogue_id', 'transformed_dialogue'),
                    'original_dialogue': original_dialogue,
                    'transformed_turns': transformed_turns,
                    'transformation_metadata': {
                        'transformation_summary': transformation_data.get('transformation_summary', {}),
                        'agent': self.agent_name,
                        'transformation_success': True
                    }
                }
                
                return enhanced_dialogue
                
            else:
                raise ValueError("No valid JSON found in transformation response")
                
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            self.logger.warning(f"Failed to parse transformation response: {str(e)}")
            return self._extract_transformation_from_text(response_content, original_dialogue)
    
    def _extract_transformation_from_text(self, text: str, original_dialogue: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract transformation from free text when JSON parsing fails
        
        Args:
            text: Raw transformation text
            original_dialogue: Original dialogue for fallback
            
        Returns:
            Basic transformed dialogue structure
        """
        # Try to extract dialogue lines from text
        lines = text.split('\n')
        transformed_turns = []
        
        current_turn = None
        for line in lines:
            line = line.strip()
            
            # Look for turn indicators
            if ('USER:' in line or 'SYSTEM:' in line or 
                'Turn' in line and ':' in line):
                
                if current_turn:
                    transformed_turns.append(current_turn)
                
                # Extract speaker and utterance
                if 'USER:' in line:
                    utterance = line.split('USER:', 1)[1].strip()
                    current_turn = {
                        'turn_index': len(transformed_turns),
                        'speaker': 'USER',
                        'transformed_utterance': utterance,
                        'transformation_notes': 'Extracted from text'
                    }
                elif 'SYSTEM:' in line:
                    utterance = line.split('SYSTEM:', 1)[1].strip()
                    current_turn = {
                        'turn_index': len(transformed_turns),
                        'speaker': 'SYSTEM',
                        'transformed_utterance': utterance,
                        'transformation_notes': 'Extracted from text'
                    }
        
        if current_turn:
            transformed_turns.append(current_turn)
        
        # If no turns found, create fallback
        if not transformed_turns:
            transformed_turns = self._create_minimal_transformation(original_dialogue)
        
        return {
            'dialogue_id': original_dialogue.get('dialogue_id', 'transformed_dialogue'),
            'original_dialogue': original_dialogue,
            'transformed_turns': transformed_turns,
            'transformation_metadata': {
                'agent': self.agent_name,
                'transformation_success': False,
                'fallback_used': True
            }
        }
    
    def _create_minimal_transformation(self, original_dialogue: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Create minimal transformation as fallback"""
        original_turns = self._extract_dialogue_turns(original_dialogue)
        transformed_turns = []
        
        for turn in original_turns:
            transformed_turns.append({
                'turn_index': turn.get('turn_index', len(transformed_turns)),
                'speaker': turn.get('speaker', 'USER'),
                'original_utterance': turn.get('utterance', ''),
                'transformed_utterance': turn.get('utterance', ''),  # No transformation
                'transformation_notes': 'Minimal fallback transformation'
            })
        
        return transformed_turns
    
    def _create_fallback_transformation(self, 
                                      original_dialogue: Dict[str, Any],
                                      personality_data: Dict[str, Any],
                                      user_state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create fallback transformation when LLM call fails
        
        Args:
            original_dialogue: Original dialogue
            personality_data: Personality information
            user_state: User state information
            
        Returns:
            Basic transformed dialogue with minimal enhancement
        """
        original_turns = self._extract_dialogue_turns(original_dialogue)
        transformed_turns = []
        
        # Apply simple personality-based modifications
        for turn in original_turns:
            utterance = turn.get('utterance', '')
            speaker = turn.get('speaker', 'USER')
            
            if speaker == 'USER' and utterance:
                # Apply basic personality-based modifications
                transformed_utterance = self._apply_basic_personality_transformation(
                    utterance, personality_data, user_state
                )
            else:
                transformed_utterance = utterance
            
            transformed_turns.append({
                'turn_index': turn.get('turn_index', len(transformed_turns)),
                'speaker': speaker,
                'original_utterance': utterance,
                'transformed_utterance': transformed_utterance,
                'transformation_notes': 'Basic fallback transformation applied'
            })
        
        return {
            'dialogue_id': original_dialogue.get('dialogue_id', 'fallback_transformed'),
            'original_dialogue': original_dialogue,
            'transformed_turns': transformed_turns,
            'transformation_metadata': {
                'agent': self.agent_name,
                'transformation_success': False,
                'fallback_transformation': True
            }
        }
    
    def _apply_basic_personality_transformation(self,
                                              utterance: str,
                                              personality_data: Dict[str, Any],
                                              user_state: Dict[str, Any]) -> str:
        """Apply basic personality-based text modifications"""
        if not utterance:
            return utterance
        
        modified_utterance = utterance
        
        # Basic personality modifications
        big_five = personality_data.get('big_five', {}) if personality_data else {}
        
        # High agreeableness - add politeness
        if big_five.get('A', 0.5) > 0.7:
            if not any(polite_word in modified_utterance.lower() for polite_word in ['please', 'thank', 'sorry']):
                modified_utterance = f"Please {modified_utterance.lower()}"
        
        # High neuroticism - add uncertainty
        if big_five.get('N', 0.5) > 0.7:
            if '?' not in modified_utterance:
                modified_utterance = f"I think {modified_utterance.lower()}"
        
        # Low extraversion - make more tentative
        if big_five.get('E', 0.5) < 0.3:
            modified_utterance = modified_utterance.replace('I want', 'I would like')
            modified_utterance = modified_utterance.replace('I need', 'I would need')
        
        return modified_utterance
