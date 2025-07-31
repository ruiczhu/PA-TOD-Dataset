"""
Agent 3: User State Simulation Expert

This agent simulates comprehensive user state information based on personality traits,
user profile, and scenario context to predict the user's emotional, cognitive, and
physical state during the dialogue interaction.
"""

import json
import logging
from typing import Dict, List, Any, Optional

from multi_agents.utils.llm_interface import LLMInterface
from multi_agents.utils.data_structures import UserState, EnhancedDialogue


class SimulationExpert:
    """
    Agent 3: Professional user state simulation expert
    
    This agent analyzes personality traits, user profile, and scenario context
    to simulate comprehensive user state including emotional, cognitive, and
    physical aspects that influence communication behavior during dialogue.
    """
    
    def __init__(self, llm_interface: Optional[LLMInterface] = None):
        """
        Initialize Simulation Expert Agent
        
        Args:
            llm_interface: LLM interface for API calls (creates default if None)
        """
        self.agent_name = "SimulationExpert"
        self.llm_interface = llm_interface or LLMInterface()
        self.logger = logging.getLogger(__name__)
        
        # System prompt for user state simulation
        self.system_prompt = """You are a professional user state simulation expert specializing in psychological and behavioral modeling for human-computer interactions. Your expertise lies in analyzing personality traits, user profiles, and contextual scenarios to predict comprehensive user states that influence communication patterns and service interaction behaviors.

Your responsibilities:
1. Analyze personality traits and their impact on emotional and cognitive states
2. Consider scenario context and environmental factors affecting user state
3. Integrate user profile characteristics with situational pressures
4. Simulate realistic emotional, cognitive, and physical states
5. Predict how these states influence communication style and service expectations

Key expertise areas:
- Psychological state modeling based on personality traits
- Stress and emotional response prediction under various scenarios
- Cognitive load assessment in service interaction contexts
- Physical state considerations affecting communication preferences
- Behavioral manifestation of internal states in dialogue contexts"""

        self.logger.info(f"{self.agent_name} initialized")
    
    def simulate_user_state(self, 
                           personality_data: Dict[str, Any],
                           user_profile: Dict[str, Any],
                           scenario_info: Dict[str, Any],
                           dialogue_context: Dict[str, Any] = None) -> UserState:
        """
        Simulate comprehensive user state based on all available context
        
        Args:
            personality_data: Personality information (Big Five + facets)
            user_profile: User profile information
            scenario_info: Scenario context information
            dialogue_context: Optional dialogue context for additional insights
            
        Returns:
            UserState object with comprehensive state simulation
            
        Raises:
            Exception: If state simulation fails
        """
        try:
            # Create user prompt for state simulation
            user_prompt = self._create_simulation_prompt(
                personality_data, user_profile, scenario_info, dialogue_context
            )
            
            # Call LLM to simulate user state
            response = self.llm_interface.call_agent(
                agent_name=self.agent_name,
                system_prompt=self.system_prompt,
                user_prompt=user_prompt,
                temperature=0.8,  # Slightly higher for more nuanced state modeling
                max_tokens=2000
            )
            
            if response.get('success', False):
                # Parse and structure the state information
                user_state = self._parse_state_response(response['content'])
                self.logger.info(f"User state simulated successfully")
                return user_state
            else:
                raise Exception("LLM call failed for state simulation")
                
        except Exception as e:
            self.logger.error(f"Failed to simulate user state: {str(e)}")
            raise
    
    def _create_simulation_prompt(self, 
                                personality_data: Dict[str, Any],
                                user_profile: Dict[str, Any],
                                scenario_info: Dict[str, Any],
                                dialogue_context: Dict[str, Any] = None) -> str:
        """
        Create detailed prompt for user state simulation
        
        Args:
            personality_data: Personality trait information
            user_profile: User profile information
            scenario_info: Scenario context information
            dialogue_context: Optional dialogue context
            
        Returns:
            Formatted prompt string for LLM
        """
        # Format personality information
        personality_summary = self._format_personality_for_simulation(personality_data)
        
        # Format user profile
        profile_summary = self._format_profile_for_simulation(user_profile)
        
        # Format scenario information
        scenario_summary = self._format_scenario_for_simulation(scenario_info)
        
        # Format dialogue context if available
        context_info = ""
        if dialogue_context:
            services = dialogue_context.get('services', [])
            intents = dialogue_context.get('intents', [])
            context_info = f"""
**Service Interaction Context:**
- Services involved: {', '.join(services)}
- User intents: {', '.join(intents)}
- Interaction complexity: {'High' if len(services) > 2 else 'Medium' if len(services) > 1 else 'Low'}"""
        
        prompt = f"""Based on the comprehensive user information provided below, simulate a detailed and realistic user state that would naturally emerge from the combination of personality traits, personal circumstances, and situational context.

## User Information

### Personality Profile
{personality_summary}

### User Background
{profile_summary}

### Current Scenario
{scenario_summary}
{context_info}

## User State Simulation Task

Analyze how this specific combination of personality, background, and situational factors would realistically manifest in the user's current state across multiple dimensions:

### Emotional State Analysis
- Primary emotional state based on personality tendencies and scenario pressure
- Emotional intensity level considering neuroticism and situational stress
- Mood fluctuations and emotional stability in this context
- Confidence level in handling the current service interaction

### Cognitive State Assessment
- Mental clarity and focus level given scenario complexity
- Cognitive load from environmental and task demands
- Decision-making capacity under current conditions
- Information processing preferences based on personality and stress

### Physical State Considerations
- Energy level considering time, scenario, and personality factors
- Physical comfort level in the current environment
- Any physical constraints affecting communication preferences
- Urgency-related physical manifestations (if applicable)

### Behavioral Manifestation Prediction
- How these internal states would influence communication style
- Service interaction preferences under current state conditions
- Patience level and tolerance for complications or delays
- Adaptation strategies the user might employ

Please provide your simulation in the following JSON format:

{{
    "emotional_state": "Primary emotional state with intensity (e.g., 'Moderately anxious but optimistic')",
    "stress_level": "Current stress level assessment (e.g., 'Low to moderate - manageable daily stress')",
    "patience_level": "Patience capacity assessment (e.g., 'Moderate - willing to work through issues')",
    "confidence_level": "Confidence in handling situation (e.g., 'High - comfortable with technology and services')",
    "current_mood": "Overall mood description (e.g., 'Professional but friendly, focused on efficiency')",
    "physical_state": "Physical condition and energy (e.g., 'Alert and energetic, no physical constraints')",
    "cognitive_load": "Mental processing capacity (e.g., 'Medium - can handle moderate complexity')",
    "situational_factors": {{
        "urgency_perception": "How urgent the situation feels to the user",
        "environmental_comfort": "Comfort level with current environment and context",
        "social_pressure": "Any social or external pressure factors",
        "task_confidence": "Confidence in successfully completing the service task",
        "adaptation_strategies": "How the user is likely to adapt their communication style"
    }}
}}

Ensure the simulation is psychologically realistic and internally consistent, reflecting how someone with this specific personality profile would actually experience and respond to this particular situation."""

        return prompt
    
    def _format_personality_for_simulation(self, personality_data: Dict[str, Any]) -> str:
        """Format personality data for simulation context"""
        if not personality_data:
            return "No specific personality profile available - use general behavioral patterns"
        
        lines = []
        
        # Big Five dimensions with simulation-relevant descriptions
        big_five = personality_data.get('big_five', {})
        if big_five and isinstance(big_five, dict):
            lines.append("**Core Personality Traits:**")
            trait_descriptions = {
                'O': ('Openness', 'novelty-seeking, adaptability, intellectual curiosity'),
                'C': ('Conscientiousness', 'organization, self-discipline, goal orientation'),
                'E': ('Extraversion', 'social energy, assertiveness, positive emotions'),
                'A': ('Agreeableness', 'cooperation, trust, empathy toward others'),
                'N': ('Neuroticism', 'emotional reactivity, stress sensitivity, anxiety proneness')
            }
            
            for dim, score in big_five.items():
                name, description = trait_descriptions.get(dim, (dim, 'general trait'))
                level = self._score_to_level(score)
                lines.append(f"- {name} ({level}): {description}")
        
        # High-impact facets for state simulation
        facets = personality_data.get('facets', {})
        if facets and isinstance(facets, dict):
            high_impact_facets = []
            for dimension, traits in facets.items():
                if isinstance(traits, dict):
                    for trait_code, score in traits.items():
                        try:
                            score_float = float(score)
                            if score_float > 0.7:  # Very high traits
                                high_impact_facets.append((trait_code, score_float, 'Very High'))
                            elif score_float < 0.3:  # Very low traits
                                high_impact_facets.append((trait_code, score_float, 'Very Low'))
                        except (ValueError, TypeError):
                            continue
            
            if high_impact_facets:
                lines.append("\n**Notable Trait Extremes:**")
                facet_names = {
                    'N1': 'Anxiety', 'N6': 'Vulnerability', 'E3': 'Assertiveness',
                    'C5': 'Self-Discipline', 'A4': 'Compliance', 'O5': 'Intellectual Curiosity'
                }
                for trait_code, score, level in high_impact_facets[:4]:  # Top 4 extremes
                    trait_name = facet_names.get(trait_code, trait_code)
                    lines.append(f"- {trait_name}: {level} ({score:.3f})")
        
        return '\n'.join(lines) if lines else "Moderate personality profile across all dimensions"
    
    def _format_profile_for_simulation(self, user_profile: Dict[str, Any]) -> str:
        """Format user profile for simulation context"""
        if not user_profile:
            return "General adult user profile"
        
        profile_dict = user_profile.__dict__ if hasattr(user_profile, '__dict__') else user_profile
        
        lines = [
            f"**Demographics:** {profile_dict.get('age_range', 'Adult')} - {profile_dict.get('occupation', 'Working professional')}",
            f"**Education:** {profile_dict.get('education_level', 'College educated')}",
            f"**Tech Comfort:** {profile_dict.get('tech_savviness', 'Moderate technology user')}",
            f"**Communication Style:** {profile_dict.get('communication_style', 'Professional and direct')}",
            f"**Background:** {profile_dict.get('background', 'Standard professional background')}"
        ]
        
        motivations = profile_dict.get('motivations', [])
        if motivations:
            lines.append(f"**Key Motivations:** {', '.join(motivations[:3])}")
        
        return '\n'.join(lines)
    
    def _format_scenario_for_simulation(self, scenario_info: Dict[str, Any]) -> str:
        """Format scenario information for simulation context"""
        if not scenario_info:
            return "Standard service interaction environment"
        
        scenario_dict = scenario_info.__dict__ if hasattr(scenario_info, '__dict__') else scenario_info
        
        lines = [
            f"**Location:** {scenario_dict.get('location', 'Service location')}",
            f"**Timing:** {scenario_dict.get('time_of_day', 'Business hours')}",
            f"**Environment:** {scenario_dict.get('crowd_level', 'Moderate activity')} activity level",
            f"**Service Urgency:** {scenario_dict.get('urgency_level', 'Moderate')}",
            f"**Task Complexity:** {scenario_dict.get('service_complexity', 'Medium')} complexity"
        ]
        
        weather = scenario_dict.get('weather')
        if weather and weather != 'Moderate weather conditions':
            lines.append(f"**Weather:** {weather}")
        
        return '\n'.join(lines)
    
    def _score_to_level(self, score) -> str:
        """Convert numeric score to descriptive level"""
        try:
            score_float = float(score)
            if score_float >= 0.7:
                return "High"
            elif score_float >= 0.4:
                return "Moderate"
            else:
                return "Low"
        except (ValueError, TypeError):
            return "Moderate"  # Default fallback
    
    def _parse_state_response(self, response_content: str) -> UserState:
        """
        Parse LLM response and create UserState object
        
        Args:
            response_content: Raw LLM response content
            
        Returns:
            UserState object with parsed data
        """
        # Try to extract JSON from response
        response_content = response_content.strip()
        
        # Find JSON content
        start_idx = response_content.find('{')
        end_idx = response_content.rfind('}') + 1
        
        if start_idx >= 0 and end_idx > start_idx:
            json_content = response_content[start_idx:end_idx]
            state_data = json.loads(json_content)
            
            # Create UserState object
            user_state = UserState(
                emotional_state=state_data.get('emotional_state', ''),
                stress_level=state_data.get('stress_level', ''),
                patience_level=state_data.get('patience_level', ''),
                confidence_level=state_data.get('confidence_level', ''),
                current_mood=state_data.get('current_mood', ''),
                physical_state=state_data.get('physical_state', ''),
                cognitive_load=state_data.get('cognitive_load', ''),
                situational_factors=state_data.get('situational_factors', {})
            )
            
            return user_state
        else:
            raise ValueError("No valid JSON found in response")
