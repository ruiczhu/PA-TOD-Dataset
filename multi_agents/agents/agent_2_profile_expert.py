"""
Agent 2: User Profile Expert

This agent generates comprehensive user profiles and persona information
based on personality traits and dialogue context, creating detailed user
backgrounds that influence communication style and behavior.
"""

import json
import logging
from typing import Dict, List, Any, Optional

from multi_agents.utils.llm_interface import LLMInterface
from multi_agents.utils.data_structures import UserProfile, EnhancedDialogue

# Import existing personality framework
from MPEAF.personality_framework import PersonalityFramework


class ProfileExpert:
    """
    Agent 2: Professional user profile and persona generation expert
    
    This agent analyzes personality traits and dialogue context to generate
    comprehensive user profiles including demographics, background, motivations,
    and behavioral characteristics that influence communication patterns.
    """
    
    def __init__(self, llm_interface: Optional[LLMInterface] = None):
        """
        Initialize Profile Expert Agent
        
        Args:
            llm_interface: LLM interface for API calls (creates default if None)
        """
        self.agent_name = "ProfileExpert"
        self.llm_interface = llm_interface or LLMInterface()
        self.logger = logging.getLogger(__name__)
        
        # Initialize personality framework
        self.personality_framework = PersonalityFramework()
        
        # System prompt for user profile generation
        self.system_prompt = """You are a professional user profile and persona generation expert specializing in creating comprehensive user backgrounds based on personality traits and behavioral patterns. Your expertise lies in psychological profiling, demographic analysis, and understanding how personality traits manifest in user behavior and communication styles.

Your responsibilities:
1. Analyze Big Five personality traits and their implications for user characteristics
2. Generate realistic user demographics and background information
3. Create coherent user motivations, preferences, and behavioral patterns
4. Ensure profile consistency with observed personality traits
5. Provide detailed, psychologically-grounded user personas in English

Key expertise areas:
- Personality psychology and trait manifestation
- Demographic profiling and behavioral correlation
- User motivation and preference analysis
- Communication style prediction based on personality
- Cultural and social factors influencing user behavior"""

        self.logger.info(f"{self.agent_name} initialized")
    
    def generate_complete_profile(self, 
                                dialogue: Dict[str, Any], 
                                scenario_info: Any = None) -> Dict[str, Any]:
        """
        Generate complete user profile from dialogue and scenario context
        
        Args:
            dialogue: Dialogue data to analyze
            scenario_info: Scenario context information
            
        Returns:
            Dictionary containing both personality_data and user_profile
        """
        try:
            # Generate personality data using LLM and PersonalityFramework structure
            personality_data = self._generate_personality_from_dialogue(dialogue)
            
            # Convert scenario_info to dict format if it's an object
            scenario_dict = {}
            if scenario_info:
                if hasattr(scenario_info, '__dict__'):
                    scenario_dict = scenario_info.__dict__
                elif isinstance(scenario_info, dict):
                    scenario_dict = scenario_info
            
            # Generate user profile using personality data and scenario context
            dialogue_context = {
                'dialogue': dialogue,
                'scenario': scenario_dict
            }
            
            user_profile = self.generate_profile(personality_data, dialogue_context)
            
            return {
                'personality_data': personality_data,
                'user_profile': user_profile
            }
            
        except Exception as e:
            self.logger.error(f"Failed to generate complete profile: {str(e)}")
            raise
    
    def generate_profile(self, personality_data: Dict[str, Any], dialogue_context: Dict[str, Any] = None) -> UserProfile:
        """
        Generate comprehensive user profile based on personality traits
        
        Args:
            personality_data: Personality information (Big Five + facets)
            dialogue_context: Optional dialogue context for additional insights
            
        Returns:
            UserProfile object with comprehensive user information
            
        Raises:
            Exception: If profile generation fails
        """
        try:
            # Create user prompt for profile generation
            user_prompt = self._create_profile_prompt(personality_data, dialogue_context)
            
            # Call LLM to generate user profile
            response = self.llm_interface.call_agent(
                agent_name=self.agent_name,
                system_prompt=self.system_prompt,
                user_prompt=user_prompt,
                temperature=0.8,  # Slightly higher for more creative profiling
                max_tokens=2000
            )
            
            if response.get('success', False):
                # Parse and structure the profile information
                user_profile = self._parse_profile_response(response['content'])
                self.logger.info(f"User profile generated successfully")
                return user_profile
            else:
                raise Exception("LLM call failed for profile generation")
                
        except Exception as e:
            self.logger.error(f"Failed to generate user profile: {str(e)}")
            raise
    
    def _create_profile_prompt(self, personality_data: Dict[str, Any], dialogue_context: Dict[str, Any] = None) -> str:
        """
        Create detailed prompt for user profile generation
        
        Args:
            personality_data: Personality trait information
            dialogue_context: Optional dialogue context
            
        Returns:
            Formatted prompt string for LLM
        """
        # Extract Big Five scores
        big_five = personality_data.get('big_five', {})
        facets = personality_data.get('facets', {})
        
        # Format personality information
        personality_summary = self._format_personality_summary(big_five, facets)
        
        # Format dialogue context if available
        context_info = ""
        if dialogue_context:
            services = dialogue_context.get('services', [])
            context_info = f"\n**Service Context:** {', '.join(services)}"
            
            if 'scenario_info' in dialogue_context:
                scenario = dialogue_context['scenario_info']
                context_info += f"\n**Scenario:** {scenario.location} - {scenario.time_of_day}"
        
        prompt = f"""Based on the following personality profile, generate a comprehensive and realistic user persona that would naturally exhibit these personality traits in their communication and behavior patterns.

## Personality Profile
{personality_summary}
{context_info}

## User Profile Generation Task

Create a detailed user profile that psychologically aligns with these personality traits. Consider how each trait influences:

### Demographic Characteristics
- Age range that fits personality maturity and behavioral patterns
- Occupation that aligns with personality strengths and preferences
- Education level reflecting cognitive openness and achievement orientation
- Technology adoption patterns based on openness and conscientiousness

### Background & Lifestyle
- Professional background and career trajectory
- Personal interests and hobbies that reflect personality traits
- Social connections and relationship patterns
- Life experiences that shaped current personality expression

### Communication & Behavior Patterns
- Preferred communication style and tone
- Decision-making approach and information processing
- Stress response and coping mechanisms
- Social interaction preferences and boundaries

### Motivations & Preferences
- Core values and priorities based on personality structure
- Service preferences and expectations
- Quality vs. efficiency trade-offs
- Risk tolerance and adventure seeking

Please provide your analysis in the following JSON format:

{{
    "age_range": "Specific age range (e.g., '28-35 years old')",
    "occupation": "Specific job title and field (e.g., 'Marketing Manager in tech industry')",
    "education_level": "Educational background (e.g., 'Bachelor's degree in Business Administration')",
    "tech_savviness": "Technology comfort level (e.g., 'High - early adopter of new apps and services')",
    "communication_style": "Preferred communication approach (e.g., 'Direct and efficient, prefers clear information')",
    "background": "2-3 sentence personal background story that explains personality development",
    "motivations": [
        "Primary motivation 1",
        "Primary motivation 2", 
        "Primary motivation 3"
    ],
    "preferences": {{
        "service_style": "How they prefer to be served (e.g., 'Quick and professional')",
        "information_detail": "Level of detail preferred (e.g., 'Comprehensive but organized')",
        "social_interaction": "Social preference (e.g., 'Friendly but task-focused')",
        "decision_making": "How they make decisions (e.g., 'Analytical with quick execution')",
        "quality_vs_speed": "Trade-off preference (e.g., 'Values quality but appreciates efficiency')"
    }}
}}

Ensure the profile is internally consistent and realistically reflects how someone with this personality profile would actually behave and communicate in service interactions."""

        return prompt
    
    def _format_personality_summary(self, big_five: Dict[str, float], facets: Dict[str, Dict[str, float]]) -> str:
        """
        Format personality information for prompt inclusion using existing framework
        
        Args:
            big_five: Big Five dimension scores
            facets: Facet scores by dimension
            
        Returns:
            Formatted personality summary string
        """
        summary_lines = []
        
        # Get dimension information from framework
        dimensions_info = self.personality_framework.get_big_five_dimensions()
        
        # Format Big Five dimensions
        if big_five and isinstance(big_five, dict):
            summary_lines.append("**Big Five Dimensions:**")
            for dimension, score in big_five.items():
                level = self._score_to_level(score)
                dim_name = dimensions_info.get(dimension, {}).get('name', dimension)
                summary_lines.append(f"- {dim_name}: {score:.3f} ({level})")
        
        # Format key facets (high scoring ones) using framework
        if facets and isinstance(facets, dict):
            summary_lines.append("\n**Notable Personality Facets:**")
            all_traits = self.personality_framework.get_all_traits_with_markers()
            all_facets = []
            
            for dimension, traits in facets.items():
                if isinstance(traits, dict):
                    for trait_code, score in traits.items():
                        # Ensure score is numeric before comparison
                        try:
                            score_float = float(score)
                            if score_float > 0.6:  # High scoring traits
                                all_facets.append((trait_code, score_float))
                        except (ValueError, TypeError):
                            continue
            
            # Sort by score and take top 6
            all_facets.sort(key=lambda x: x[1], reverse=True)
            for trait_code, score in all_facets[:6]:
                level = self._score_to_level(score)
                trait_info = all_traits.get(trait_code, {})
                trait_name = trait_info.get('name', trait_code)
                summary_lines.append(f"- {trait_name}: {score:.3f} ({level})")
        
        return '\n'.join(summary_lines)
    
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
    
    def _parse_profile_response(self, response_content: str) -> UserProfile:
        """
        Parse LLM response and create UserProfile object
        
        Args:
            response_content: Raw LLM response content
            
        Returns:
            UserProfile object with parsed data
        """
        try:
            # Try to extract JSON from response
            response_content = response_content.strip()
            
            # Find JSON content
            start_idx = response_content.find('{')
            end_idx = response_content.rfind('}') + 1
            
            if start_idx >= 0 and end_idx > start_idx:
                json_content = response_content[start_idx:end_idx]
                profile_data = json.loads(json_content)
                
                # Create UserProfile object
                user_profile = UserProfile(
                    age_range=profile_data.get('age_range', ''),
                    occupation=profile_data.get('occupation', ''),
                    education_level=profile_data.get('education_level', ''),
                    tech_savviness=profile_data.get('tech_savviness', ''),
                    communication_style=profile_data.get('communication_style', ''),
                    background=profile_data.get('background', ''),
                    motivations=profile_data.get('motivations', []),
                    preferences=profile_data.get('preferences', {})
                )
                
                return user_profile
                
            else:
                raise ValueError("No valid JSON found in response")
                
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            self.logger.warning(f"Failed to parse profile response: {str(e)}")
            raise
    
    def _generate_personality_from_dialogue(self, dialogue: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate personality data from dialogue using LLM and PersonalityFramework structure
        
        Args:
            dialogue: Input dialogue data
            
        Returns:
            Dictionary containing personality dimensions and facets
        """
        try:
            # Extract dialogue text for analysis
            dialogue_text = []
            if 'turns' in dialogue:
                for turn in dialogue['turns']:
                    if turn.get('speaker') == 'USER':
                        dialogue_text.append(turn.get('utterance', ''))
            
            dialogue_content = ' '.join(dialogue_text)
            
            # Use LLM to analyze personality based on Big Five model
            prompt = f"""
Analyze the following user dialogue and extract Big Five personality traits.
Provide scores from 0.0 to 1.0 for each dimension based on the language patterns and content.

Dialogue: {dialogue_content}

Please provide:
1. Big Five scores (Openness, Conscientiousness, Extraversion, Agreeableness, Neuroticism)
2. Key facets for each dimension
3. Brief explanation for each score

Format as JSON with this structure:
{{
    "big_five": {{
        "O": <score>,
        "C": <score>, 
        "E": <score>,
        "A": <score>,
        "N": <score>
    }},
    "facets": {{
        "dominant_facets": ["list of prominent facet codes"],
        "explanations": {{
            "O": "reason for openness score",
            "C": "reason for conscientiousness score",
            "E": "reason for extraversion score", 
            "A": "reason for agreeableness score",
            "N": "reason for neuroticism score"
        }}
    }}
}}
"""

            response = self.llm_interface.call_agent(
                agent_name="ProfileExpert",
                system_prompt="You are a personality analysis expert. Analyze dialogue patterns to extract Big Five personality traits. Always respond with valid JSON.",
                user_prompt=prompt,
                temperature=0.3,
                max_tokens=800
            )
            
            if response and response.get('success') and response.get('content'):
                content = response['content'].strip()
                
                # Try to extract JSON from response
                import json
                import re
                
                try:
                    # First try direct parsing
                    personality_data = json.loads(content)
                    return personality_data
                except json.JSONDecodeError:
                    # Try to find JSON in the response
                    json_match = re.search(r'\{.*\}', content, re.DOTALL)
                    if json_match:
                        try:
                            personality_data = json.loads(json_match.group())
                            return personality_data
                        except json.JSONDecodeError:
                            pass
                    
                    self.logger.warning(f"Failed to parse LLM personality response as JSON: {content[:200]}...")
            
            # No fallback - raise error to surface parsing issues
            raise ValueError("Failed to parse personality data from LLM response")
            
        except Exception as e:
            self.logger.error(f"Error generating personality from dialogue: {str(e)}")
            raise
