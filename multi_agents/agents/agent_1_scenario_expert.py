"""
Agent 1: Scenario Context Expert

This agent analyzes dialogue context and generates detailed scenario information
including location, environment, situational factors, and contextual details
that influence how the dialogue takes place.
"""

import json
import logging
from typing import Dict, List, Any, Optional

# Import utilities with proper path handling
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from multi_agents.utils.llm_interface import LLMInterface
from multi_agents.utils.data_structures import ScenarioInfo, EnhancedDialogue


class ScenarioExpert:
    """
    Agent 1: Professional scenario and context setting expert
    
    This agent analyzes dialogue content and generates comprehensive scenario
    information including environmental factors, situational context, and 
    circumstances that would influence user behavior and dialogue style.
    """
    
    def __init__(self, llm_interface: Optional[LLMInterface] = None):
        """
        Initialize Scenario Expert Agent
        
        Args:
            llm_interface: LLM interface for API calls (creates default if None)
        """
        self.agent_name = "ScenarioExpert"
        self.llm_interface = llm_interface or LLMInterface()
        self.logger = logging.getLogger(__name__)
        
        # System prompt for scenario generation
        self.system_prompt = """You are a professional scenario and context setting expert specializing in task-oriented dialogue analysis. Your expertise lies in analyzing dialogue content and generating comprehensive, realistic scenario information that provides rich context for understanding user behavior and communication patterns.

Your responsibilities:
1. Analyze dialogue content to infer environmental and situational context
2. Generate detailed scenario information including location, time, environmental factors
3. Assess situational urgency, complexity, and environmental pressures
4. Provide structured, consistent scenario details in English
5. Ensure scenario information is realistic and relevant to the dialogue's service domain

Key expertise areas:
- Environmental context analysis (location, weather, crowd levels, etc.)
- Temporal context (time of day, business hours, scheduling constraints)
- Service context (complexity, urgency, accessibility)
- Situational factors that influence user communication style"""

        self.logger.info(f"{self.agent_name} initialized")
    
    def generate_scenario(self, dialogue: Dict[str, Any]) -> ScenarioInfo:
        """
        Generate detailed scenario information for a dialogue
        
        Args:
            dialogue: Original dialogue data
            
        Returns:
            ScenarioInfo object with comprehensive scenario details
            
        Raises:
            Exception: If scenario generation fails
        """
        try:
            # Extract dialogue context
            dialogue_context = self._extract_dialogue_context(dialogue)
            
            # Create user prompt for scenario generation
            user_prompt = self._create_scenario_prompt(dialogue_context)
            
            # Call LLM to generate scenario
            response = self.llm_interface.call_agent(
                agent_name=self.agent_name,
                system_prompt=self.system_prompt,
                user_prompt=user_prompt,
                temperature=0.7,
                max_tokens=1500
            )
            
            if response.get('success', False):
                # Parse and structure the scenario information
                scenario_info = self._parse_scenario_response(response['content'])
                self.logger.info(f"Scenario generated successfully for dialogue {dialogue.get('dialogue_id', 'unknown')}")
                return scenario_info
            else:
                raise Exception("LLM call failed for scenario generation")
                
        except Exception as e:
            self.logger.error(f"Failed to generate scenario for dialogue {dialogue.get('dialogue_id', 'unknown')}: {str(e)}")
            # Return basic scenario as fallback
            return self._create_fallback_scenario(dialogue)
    
    def _extract_dialogue_context(self, dialogue: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract relevant context from dialogue for scenario generation
        Uses structured SGD data including intents, slots, and actions
        
        Args:
            dialogue: Original dialogue data
            
        Returns:
            Dictionary with extracted context information
        """
        context = {
            'dialogue_id': dialogue.get('dialogue_id', ''),
            'services': dialogue.get('services', []),
            'turn_count': len(dialogue.get('turns', [])),
            'user_utterances': [],
            'system_utterances': [],
            'structured_entities': {},
            'intents': set(),
            'actions': set(),
            'slot_values': {},
            'dialogue_flow': []
        }
        
        # Extract utterances and structured information
        for i, turn in enumerate(dialogue.get('turns', [])):
            speaker = turn.get('speaker', '')
            utterance = turn.get('utterance', '')
            frames = turn.get('frames', [])
            
            # Store utterances by speaker
            if speaker == 'USER':
                context['user_utterances'].append(utterance)
            elif speaker == 'SYSTEM':
                context['system_utterances'].append(utterance)
            
            # Track dialogue flow
            context['dialogue_flow'].append({
                'turn': i,
                'speaker': speaker,
                'utterance': utterance[:100] + '...' if len(utterance) > 100 else utterance
            })
            
            # Extract structured information from frames
            for frame in frames:
                service = frame.get('service', '')
                
                # Extract intents
                if 'state' in frame and 'active_intent' in frame['state']:
                    intent = frame['state']['active_intent']
                    if intent:
                        context['intents'].add(intent)
                
                # Extract slot values
                if 'state' in frame and 'slot_values' in frame['state']:
                    for slot, values in frame['state']['slot_values'].items():
                        if slot not in context['slot_values']:
                            context['slot_values'][slot] = []
                        context['slot_values'][slot].extend(values)
                
                # Extract actions
                for action in frame.get('actions', []):
                    act = action.get('act', '')
                    slot = action.get('slot', '')
                    values = action.get('values', [])
                    canonical_values = action.get('canonical_values', [])
                    
                    if act:
                        context['actions'].add(act)
                    
                    # Store structured entities based on actions
                    if slot and (values or canonical_values):
                        entity_values = values or canonical_values
                        if slot not in context['structured_entities']:
                            context['structured_entities'][slot] = []
                        context['structured_entities'][slot].extend(entity_values)
        
        # Convert sets to lists for JSON serialization
        context['intents'] = list(context['intents'])
        context['actions'] = list(context['actions'])
        
        # Remove duplicates from structured entities
        for slot, values in context['structured_entities'].items():
            context['structured_entities'][slot] = list(set(values))
        
        return context
    
    def _create_scenario_prompt(self, context: Dict[str, Any]) -> str:
        """
        Create detailed prompt for scenario generation
        
        Args:
            context: Extracted dialogue context
            
        Returns:
            Formatted prompt string for LLM
        """
        # Format dialogue flow for context
        dialogue_flow_text = "\n".join([
            f"Turn {item['turn']} [{item['speaker']}]: {item['utterance']}"
            for item in context['dialogue_flow'][:8]  # Limit to first 8 turns to manage token count
        ])
        
        prompt = f"""Based on the following task-oriented dialogue, generate a comprehensive scenario context that would realistically frame this interaction.

## Dialogue Information
**Dialogue ID:** {context['dialogue_id']}
**Services:** {', '.join(context['services'])}
**Total Turns:** {context['turn_count']}

## Structured Information
**User Intents:** {', '.join(context['intents'])}
**System Actions:** {', '.join(context['actions'])}
**Key Entities:** {', '.join([f"{slot}: {', '.join(values)}" for slot, values in context['structured_entities'].items()])}

## Dialogue Flow (First 8 turns)
{dialogue_flow_text}

## Scenario Generation Task
Analyze this dialogue and generate detailed scenario information that would provide realistic context for this interaction. Consider:

### Environmental Context
- Specific location type and characteristics
- Geographic context (country, city type, area)
- Physical environment details
- Weather conditions if relevant

### Temporal Context  
- Time of day and its implications
- Day of week if relevant
- Business hours and time constraints
- Seasonal factors if applicable

### Situational Factors
- Crowd levels and busyness
- Service complexity and difficulty
- User urgency level and time pressure
- Environmental factors affecting communication

### Contextual Details
- Any special circumstances
- Accessibility considerations
- Technology/service availability
- Cultural or regional factors

Please provide your analysis in the following JSON format:

{{
    "location": "Specific location description (e.g., 'Busy downtown restaurant district')",
    "country": "Country/region (e.g., 'United States')",
    "weather": "Weather conditions if relevant (e.g., 'Rainy evening')",
    "crowd_level": "Crowd/busyness level (e.g., 'High - peak dinner time')",
    "time_of_day": "Specific time context (e.g., 'Evening, around 7 PM')",
    "business_hours": "Business operational context (e.g., 'Peak operating hours')",
    "urgency_level": "Assessed urgency (e.g., 'Moderate - dinner reservation needed')",
    "service_complexity": "Service difficulty level (e.g., 'Medium - multiple preferences to accommodate')",
    "additional_context": {{
        "environmental_factors": "Any environmental pressures or factors",
        "technological_context": "Technology availability and usage context", 
        "social_context": "Social or cultural factors influencing interaction",
        "accessibility_notes": "Any accessibility or convenience factors"
    }}
}}

Focus on creating a realistic, coherent scenario that would naturally lead to this type of dialogue interaction."""

        return prompt
    
    def _parse_scenario_response(self, response_content: str) -> ScenarioInfo:
        """
        Parse LLM response and create ScenarioInfo object
        
        Args:
            response_content: Raw LLM response content
            
        Returns:
            ScenarioInfo object with parsed data
        """
        try:
            # Try to extract JSON from response
            response_content = response_content.strip()
            
            # Find JSON content (handle cases where LLM adds extra text)
            start_idx = response_content.find('{')
            end_idx = response_content.rfind('}') + 1
            
            if start_idx >= 0 and end_idx > start_idx:
                json_content = response_content[start_idx:end_idx]
                scenario_data = json.loads(json_content)
                
                # Create ScenarioInfo object
                scenario_info = ScenarioInfo(
                    location=scenario_data.get('location', ''),
                    country=scenario_data.get('country', ''),
                    weather=scenario_data.get('weather', ''),
                    crowd_level=scenario_data.get('crowd_level', ''),
                    time_of_day=scenario_data.get('time_of_day', ''),
                    business_hours=scenario_data.get('business_hours', ''),
                    urgency_level=scenario_data.get('urgency_level', ''),
                    service_complexity=scenario_data.get('service_complexity', ''),
                    additional_context=scenario_data.get('additional_context', {})
                )
                
                return scenario_info
                
            else:
                raise ValueError("No valid JSON found in response")
                
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            self.logger.warning(f"Failed to parse scenario response: {str(e)}")
            # Create basic scenario from response text
            return self._extract_scenario_from_text(response_content)
    
    def _extract_scenario_from_text(self, text: str) -> ScenarioInfo:
        """
        Extract scenario information from free text when JSON parsing fails
        
        Args:
            text: Raw text response
            
        Returns:
            ScenarioInfo object with extracted information
        """
        # Basic text parsing as fallback
        scenario_info = ScenarioInfo()
        
        text_lower = text.lower()
        
        # Simple keyword-based extraction
        if 'restaurant' in text_lower:
            scenario_info.location = 'Restaurant area'
        elif 'hotel' in text_lower:
            scenario_info.location = 'Hotel/accommodation area'
        elif 'travel' in text_lower or 'flight' in text_lower:
            scenario_info.location = 'Travel/transportation hub'
        else:
            scenario_info.location = 'Urban service area'
        
        if 'evening' in text_lower or 'night' in text_lower:
            scenario_info.time_of_day = 'Evening'
        elif 'morning' in text_lower:
            scenario_info.time_of_day = 'Morning'
        else:
            scenario_info.time_of_day = 'Business hours'
        
        if 'urgent' in text_lower or 'emergency' in text_lower:
            scenario_info.urgency_level = 'High'
        else:
            scenario_info.urgency_level = 'Moderate'
        
        scenario_info.country = 'United States'  # Default
        scenario_info.crowd_level = 'Moderate'
        scenario_info.business_hours = 'Regular business hours'
        scenario_info.service_complexity = 'Medium'
        
        return scenario_info
    
    def _create_fallback_scenario(self, dialogue: Dict[str, Any]) -> ScenarioInfo:
        """
        Create basic fallback scenario when generation fails
        
        Args:
            dialogue: Original dialogue data
            
        Returns:
            Basic ScenarioInfo object
        """
        services = dialogue.get('services', [])
        service_type = services[0] if services else 'general'
        
        return ScenarioInfo(
            location=f"{service_type.title()} service location",
            country="United States",
            weather="Moderate weather conditions",
            crowd_level="Moderate",
            time_of_day="Business hours",
            business_hours="Regular operating hours",
            urgency_level="Moderate",
            service_complexity="Medium",
            additional_context={
                "fallback": True,
                "reason": "Scenario generation failed, using default values"
            }
        )
