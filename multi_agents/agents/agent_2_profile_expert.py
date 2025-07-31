"""
Agent 2: User Profile Expert

This agent generates comprehensive user profiles and persona information
based on personality traits and dialogue context, creating detailed user
backgrounds that influence communication style and behavior.
"""

import json
import logging
from typing import Dict, List, Any, Optional

# Import utilities with proper path handling
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

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
            # Return fallback data
            fallback_personality = {
                'big_five': {'O': 0.5, 'C': 0.5, 'E': 0.5, 'A': 0.5, 'N': 0.5},
                'facets': {}
            }
            fallback_profile = self._create_fallback_profile(fallback_personality)
            return {
                'personality_data': fallback_personality,
                'user_profile': fallback_profile
            }
    
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
            # Return basic profile as fallback
            return self._create_fallback_profile(personality_data)
    
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
            # Create basic profile from response text
            return self._extract_profile_from_text(response_content)
    
    def _extract_profile_from_text(self, text: str) -> UserProfile:
        """
        Extract profile information from free text when JSON parsing fails
        
        Args:
            text: Raw text response
            
        Returns:
            UserProfile object with extracted information
        """
        # Basic text parsing as fallback
        user_profile = UserProfile()
        
        text_lower = text.lower()
        
        # Extract age if mentioned
        if '20s' in text_lower or 'twenties' in text_lower:
            user_profile.age_range = '25-29 years old'
        elif '30s' in text_lower or 'thirties' in text_lower:
            user_profile.age_range = '30-35 years old'
        else:
            user_profile.age_range = '25-40 years old'
        
        # Extract occupation hints
        if 'manager' in text_lower:
            user_profile.occupation = 'Manager in service industry'
        elif 'professional' in text_lower:
            user_profile.occupation = 'Professional worker'
        else:
            user_profile.occupation = 'Service industry professional'
        
        # Set default values
        user_profile.education_level = 'College educated'
        user_profile.tech_savviness = 'Moderate technology user'
        user_profile.communication_style = 'Clear and direct communication'
        user_profile.background = 'Working professional with service interaction experience'
        user_profile.motivations = ['Efficient service', 'Quality results', 'Professional interaction']
        user_profile.preferences = {
            'service_style': 'Professional and efficient',
            'information_detail': 'Clear and sufficient detail',
            'social_interaction': 'Polite and task-focused'
        }
        
        return user_profile
    
    def _create_fallback_profile(self, personality_data: Dict[str, Any]) -> UserProfile:
        """
        Create basic fallback profile when generation fails
        
        Args:
            personality_data: Personality trait information
            
        Returns:
            Basic UserProfile object based on personality
        """
        # Use personality data to make educated guesses
        big_five = personality_data.get('big_five', {})
        
        # Determine characteristics based on personality scores
        openness = big_five.get('O', 0.5)
        conscientiousness = big_five.get('C', 0.5)
        extraversion = big_five.get('E', 0.5)
        
        # Age based on conscientiousness and openness
        if conscientiousness > 0.6:
            age_range = '30-40 years old'  # More mature, established
        elif openness > 0.6:
            age_range = '25-32 years old'  # Younger, more exploratory
        else:
            age_range = '25-35 years old'
        
        # Occupation based on traits
        if conscientiousness > 0.6 and extraversion > 0.6:
            occupation = 'Management or leadership role'
        elif openness > 0.6:
            occupation = 'Creative or analytical professional'
        else:
            occupation = 'Service industry professional'
        
        # Communication style based on extraversion
        if extraversion > 0.6:
            communication_style = 'Outgoing and expressive'
        elif extraversion < 0.4:
            communication_style = 'Reserved and thoughtful'
        else:
            communication_style = 'Balanced and adaptive'
        
        return UserProfile(
            age_range=age_range,
            occupation=occupation,
            education_level='College educated',
            tech_savviness='Moderate technology user',
            communication_style=communication_style,
            background='Working professional with diverse service interaction experience',
            motivations=['Effective service delivery', 'Professional interaction', 'Quality outcomes'],
            preferences={
                'service_style': 'Professional and courteous',
                'information_detail': 'Adequate detail for decision making',
                'social_interaction': 'Respectful and efficient',
                'fallback': True
            }
        )
    
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
            
            # Fallback to default personality
            self.logger.info("Using default personality data due to parsing failure")
            return self._get_default_personality()
            
        except Exception as e:
            self.logger.error(f"Error generating personality from dialogue: {str(e)}")
            return self._get_default_personality()
    
    def _get_default_personality(self) -> Dict[str, Any]:
        """
        Generate default/neutral personality data
        
        Returns:
            Default personality data structure
        """
        return {
            'big_five': {
                'O': 0.5,  # Neutral openness
                'C': 0.6,  # Slightly organized
                'E': 0.5,  # Neutral extraversion
                'A': 0.7,  # Somewhat agreeable
                'N': 0.4   # Low neuroticism
            },
            'facets': {
                'dominant_facets': ['C2_Order', 'A3_Altruism'],
                'explanations': {
                    'O': 'Moderate openness based on standard dialogue patterns',
                    'C': 'Organized approach to service requests',
                    'E': 'Balanced social interaction style',
                    'A': 'Cooperative and polite communication',
                    'N': 'Calm and stable emotional tone'
                }
            }
        }
