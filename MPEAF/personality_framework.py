from collections import OrderedDict


class PersonalityFramework:
    def __init__(self):
        """Initialize multi-level personality framework"""
        
        # NEO-PIR 大五人格完整框架，包含子特质定义和语言标记
        self.NEO_PIR_FACETS = OrderedDict([
            ('O', {  # Openness 开放性
                'name': 'Openness',
                'chinese_name': '开放性',
                'facets': [
                    'O1_Fantasy',           # 幻想
                    'O2_Aesthetics',        # 审美
                    'O3_Feelings',          # 情感
                    'O4_Actions',           # 行动
                    'O5_Ideas',             # 思辨
                    'O6_Values'             # 价值观
                ],
                'linguistic_markers': {
                    'O1': [
                        "Using imaginative descriptions and hypothetical scenarios",
                        "Discussing creative activities or dreams",
                        "Mentioning fictional worlds or scenes"
                    ],
                    'O2': [
                        "Discussing art, aesthetics, or natural beauty",
                        "Using rich descriptive vocabulary",
                        "Expressing appreciation for beauty"
                    ],
                    'O3': [
                        "Expressing emotional depth and intensity",
                        "Sharing inner feelings",
                        "Reflecting on emotional and mood states"
                    ],
                    'O4': [
                        "Discussing new experiences or attempts",
                        "Expressing preference for diversity",
                        "Mentioning unconventional activities"
                    ],
                    'O5': [
                        "Asking deep questions or philosophical thoughts",
                        "Exploring abstract concepts",
                        "Analyzing and reasoning through complex problems"
                    ],
                    'O6': [
                        "Discussing moral or ethical issues",
                        "Questioning traditional ideas or norms",
                        "Expressing non-traditional values"
                    ]
                }
            }),
            ('C', {  # Conscientiousness 尽责性
                'name': 'Conscientiousness',
                'chinese_name': '尽责性',
                'facets': [
                    'C1_Competence',        # 胜任
                    'C2_Order',             # 条理
                    'C3_Dutifulness',       # 尽职
                    'C4_Achievement_Striving', # 成就追求
                    'C5_Self_Discipline',   # 自律
                    'C6_Deliberation'       # 审慎
                ],
                'linguistic_markers': {
                    'C1': [
                        "Expressing confidence and capability",
                        "Discussing personal achievements",
                        "Providing detailed, organized explanations"
                    ],
                    'C2': [
                        "Using ordered structure and lists",
                        "Focusing on details and precision",
                        "Emphasizing planning and organization"
                    ],
                    'C3': [
                        "Emphasizing sense of duty and obligation",
                        "Expressing importance of keeping promises",
                        "Discussing moral and ethical guidelines"
                    ],
                    'C4': [
                        "Setting clear goals",
                        "Expressing ambition and drive",
                        "Discussing effort and success"
                    ],
                    'C5': [
                        "Emphasizing self-control and perseverance",
                        "Discussing ability to complete tasks",
                        "Avoiding distractions or procrastination"
                    ],
                    'C6': [
                        "Demonstrating thoughtful decision-making processes",
                        "Considering pros and cons of multiple options",
                        "Avoiding impulsive behavior"
                    ]
                }
            }),
            ('E', {  # Extraversion 外向性
                'name': 'Extraversion',
                'chinese_name': '外向性',
                'facets': [
                    'E1_Warmth',            # 热情
                    'E2_Gregariousness',    # 群居性
                    'E3_Assertiveness',     # 独断性
                    'E4_Activity',          # 活跃性
                    'E5_Excitement_Seeking', # 寻求刺激
                    'E6_Positive_Emotions'  # 积极情绪
                ],
                'linguistic_markers': {
                    'E1': [
                        "Expressing friendliness and enthusiasm",
                        "Focusing on connections with others",
                        "Using warm, affectionate language"
                    ],
                    'E2': [
                        "Mentioning social activities and groups",
                        "Expressing enjoyment of social occasions",
                        "Avoiding solitude"
                    ],
                    'E3': [
                        "Using direct, assertive language",
                        "Expressing opinions without hesitation",
                        "Guiding conversation or making suggestions"
                    ],
                    'E4': [
                        "Discussing physical activities and high-energy pursuits",
                        "Portraying busy and active lifestyle",
                        "Using dynamic vocabulary"
                    ],
                    'E5': [
                        "Seeking stimulation and adventure",
                        "Expressing interest in novel experiences",
                        "Discussing exciting activities"
                    ],
                    'E6': [
                        "Expressing optimism and joy",
                        "Using positive emotional vocabulary",
                        "Sharing happy things and laughter"
                    ]
                }
            }),
            ('A', {  # Agreeableness 宜人性
                'name': 'Agreeableness',
                'chinese_name': '宜人性',
                'facets': [
                    'A1_Trust',             # 信任
                    'A2_Straightforwardness', # 直率
                    'A3_Altruism',          # 利他
                    'A4_Compliance',        # 依从
                    'A5_Modesty',           # 谦虚
                    'A6_Tender_Mindedness'  # 同情
                ],
                'linguistic_markers': {
                    'A1': [
                        "Expressing trust in others",
                        "Assuming others have good intentions",
                        "Willingness to share personal information"
                    ],
                    'A2': [
                        "Direct, honest expression",
                        "Avoiding manipulation or concealment",
                        "Expressing genuine thoughts"
                    ],
                    'A3': [
                        "Offering help or support",
                        "Expressing concern for others' well-being",
                        "Engaging in selfless acts"
                    ],
                    'A4': [
                        "Avoiding conflict and arguments",
                        "Compromising or yielding",
                        "Displaying mild manner"
                    ],
                    'A5': [
                        "Avoiding boasting or excessive confidence",
                        "Downplaying personal achievements",
                        "Acknowledging personal limitations"
                    ],
                    'A6': [
                        "Expressing sympathy and understanding",
                        "Focusing on others' emotional needs",
                        "Using gentle, supportive language"
                    ]
                }
            }),
            ('N', {  # Neuroticism 神经质
                'name': 'Neuroticism',
                'chinese_name': '神经质',
                'facets': [
                    'N1_Anxiety',           # 焦虑
                    'N2_Angry_Hostility',   # 敌对性
                    'N3_Depression',        # 抑郁
                    'N4_Self_Consciousness', # 自我意识
                    'N5_Impulsiveness',     # 冲动性
                    'N6_Vulnerability'      # 脆弱性
                ],
                'linguistic_markers': {
                    'N1': [
                        "Expressing worry and unease",
                        "Imagining negative outcomes",
                        "Seeking reassurance"
                    ],
                    'N2': [
                        "Expressing anger or dissatisfaction",
                        "Using intense or aggressive language",
                        "Negative evaluation of others or situations"
                    ],
                    'N3': [
                        "Expressing pessimism and hopelessness",
                        "Focusing on negative aspects",
                        "Using negative emotional vocabulary"
                    ],
                    'N4': [
                        "Displaying social anxiety or awkwardness",
                        "Worrying about others' evaluations",
                        "Self-criticism"
                    ],
                    'N5': [
                        "Displaying impulsive decision making",
                        "Difficulty resisting temptation",
                        "Expressing desire for immediate gratification"
                    ],
                    'N6': [
                        "Displaying difficulty coping under pressure",
                        "Seeking help with problems",
                        "Showing overwhelm in face of challenges"
                    ]
                }
            })
        ])
    
    def get_dimension_traits(self, dimension):
        """Get all sub-traits for a specific dimension"""
        if dimension in self.NEO_PIR_FACETS:
            return self.NEO_PIR_FACETS[dimension]['facets']
        return []
    
    def get_trait_markers(self, trait_code):
        """Get linguistic markers for a specific sub-trait"""
        # Extract dimension from trait_code (e.g., 'O1' -> 'O')
        dimension = trait_code[0] if trait_code else ''
        if dimension in self.NEO_PIR_FACETS:
            return self.NEO_PIR_FACETS[dimension]['linguistic_markers'].get(trait_code, [])
        return []
    
    def get_all_traits_with_markers(self):
        """Get all sub-traits with their linguistic markers"""
        result = {}
        for domain, info in self.NEO_PIR_FACETS.items():
            for facet in info['facets']:
                # Extract facet code (e.g., 'O1_Fantasy' -> 'O1')
                facet_code = facet.split('_')[0]
                facet_name = facet.split('_', 1)[1].replace('_', ' ') if '_' in facet else facet
                
                result[facet_code] = {
                    "name": facet_name,
                    "full_name": facet,
                    "dimension": domain,
                    "dimension_name": info['name'],
                    "dimension_chinese_name": info['chinese_name'],
                    "markers": self.get_trait_markers(facet_code)
                }
        return result
    
    def get_big_five_dimensions(self):
        """Get Big Five dimensions information"""
        dimensions = {}
        for domain, info in self.NEO_PIR_FACETS.items():
            dimensions[domain] = {
                'name': info['name'],
                'chinese_name': info['chinese_name']
            }
        return dimensions