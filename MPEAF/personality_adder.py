import random
from typing import Dict, Any, Optional
import numpy as np

try:
    from .personality_framework import PersonalityFramework
except ImportError:
    from personality_framework import PersonalityFramework

class PersonalityAdder:
    """
    基于大五人格OCEAN模型的性格标签添加器
    
    使用NEO-PIR模型为对话数据添加30个子特质的性格标签：
    - O (Openness): 开放性 - 6个子特质
    - C (Conscientiousness): 尽责性 - 6个子特质  
    - E (Extraversion): 外向性 - 6个子特质
    - A (Agreeableness): 宜人性 - 6个子特质
    - N (Neuroticism): 神经质 - 6个子特质
    
    每个子特质取值范围为0-1，大五人格主特质值为其6个子特质的平均值
    """
    
    def __init__(self, random_seed: Optional[int] = None):
        """
        初始化性格添加器
        
        Args:
            random_seed: 随机种子，用于随机生成性格时确保可重现性
        """
        self.random_seed = random_seed
        if random_seed is not None:
            random.seed(random_seed)
            np.random.seed(random_seed)
        
        # 初始化性格框架
        self.personality_framework = PersonalityFramework()
        self.NEO_PIR_FACETS = self.personality_framework.NEO_PIR_FACETS
        
        # 构建所有30个子特质的列表
        self.all_facets = []
        self.facet_to_domain = {}  # 子特质到主维度的映射
        
        for domain, info in self.NEO_PIR_FACETS.items():
            for facet in info['facets']:
                self.all_facets.append(facet)
                self.facet_to_domain[facet] = domain
        
        print(f"PersonalityAdder 初始化完成")
        print(f"支持 {len(self.NEO_PIR_FACETS)} 个主维度，{len(self.all_facets)} 个子特质")
    
    def _generate_random_personality(self) -> Dict[str, float]:
        """
        随机生成30个子特质的性格值
        
        Returns:
            包含30个子特质值的字典，每个值在0-1范围内
        """
        personality = {}
        
        for facet in self.all_facets:
            # 使用正态分布生成，均值0.5，标准差0.2，然后裁剪到0-1范围
            value = np.random.normal(0.5, 0.2)
            value = max(0.0, min(1.0, value))  # 裁剪到0-1范围
            personality[facet] = round(value, 3)
        
        return personality
    
    def _calculate_big_five_scores(self, facet_scores: Dict[str, float]) -> Dict[str, float]:
        """
        根据30个子特质分数计算大五人格主维度分数
        
        Args:
            facet_scores: 包含30个子特质分数的字典
            
        Returns:
            包含5个主维度分数的字典
        """
        big_five_scores = {}
        
        for domain, info in self.NEO_PIR_FACETS.items():
            facet_values = []
            for facet in info['facets']:
                if facet in facet_scores:
                    facet_values.append(facet_scores[facet])
            
            # 计算该维度下所有子特质的平均值
            if facet_values:
                avg_score = sum(facet_values) / len(facet_values)
                big_five_scores[info['name']] = round(avg_score, 3)
            else:
                big_five_scores[info['name']] = 0.5  # 默认值
        
        return big_five_scores
    
    def _validate_personality_values(self, personality_values: Dict[str, float]) -> bool:
        """
        验证性格值的有效性
        
        Args:
            personality_values: 性格值字典
            
        Returns:
            是否有效
        """
        # 检查是否包含所有必需的子特质
        missing_facets = set(self.all_facets) - set(personality_values.keys())
        if missing_facets:
            print(f"警告: 缺少以下子特质: {missing_facets}")
            return False
        
        # 检查值是否在有效范围内
        for facet, value in personality_values.items():
            if not isinstance(value, (int, float)):
                print(f"警告: {facet} 的值不是数字: {value}")
                return False
            
            if not (0.0 <= value <= 1.0):
                print(f"警告: {facet} 的值超出范围 [0, 1]: {value}")
                return False
        
        return True
    
    def _format_personality_dict(self, facet_scores: Dict[str, float], 
                                big_five_scores: Dict[str, float]) -> Dict[str, Any]:
        """
        将性格分数格式化为结构化字典
        
        Args:
            facet_scores: 30个子特质分数
            big_five_scores: 5个主维度分数
            
        Returns:
            结构化的性格字典
        """
        personality_dict = {
            "big_five": {},
            "facets": {}
        }
        
        # 添加大五人格主维度分数和对应的子特质
        for domain, info in self.NEO_PIR_FACETS.items():
            domain_name = info['name']
            domain_score = big_five_scores.get(domain_name, 0.5)
            
            # 构建该维度的子特质字典
            facet_dict = {}
            for facet in info['facets']:
                facet_key = facet.split('_', 1)[0]  # 例如 O1_Fantasy -> O1
                facet_score = facet_scores.get(facet, 0.5)
                facet_dict[facet_key] = facet_score
            
            # 将主维度分数和子特质添加到结构中
            personality_dict["big_five"][domain] = domain_score
            personality_dict["facets"][domain] = facet_dict
        
        return personality_dict
    
    def add_random_personality(self, dialogue: Dict[str, Any]) -> Dict[str, Any]:
        """
        为单个对话添加随机生成的性格标签
        
        Args:
            dialogue: 对话数据字典
            
        Returns:
            添加了性格标签的对话数据
        """
        # 生成随机性格
        facet_scores = self._generate_random_personality()
        big_five_scores = self._calculate_big_five_scores(facet_scores)
        
        # 格式化性格字典
        personality_dict = self._format_personality_dict(facet_scores, big_five_scores)
        
        # 添加到对话数据
        dialogue_copy = dialogue.copy()
        dialogue_copy['personality'] = personality_dict
        
        return dialogue_copy
    
    def add_specified_personality(self, dialogue: Dict[str, Any], 
                                personality_values: Dict[str, float]) -> Dict[str, Any]:
        """
        为单个对话添加指定的性格标签
        
        Args:
            dialogue: 对话数据字典
            personality_values: 包含30个子特质分数的字典
            
        Returns:
            添加了性格标签的对话数据
        """
        # 验证输入的性格值
        if not self._validate_personality_values(personality_values):
            raise ValueError("提供的性格值无效")
        
        # 计算大五人格主维度分数
        big_five_scores = self._calculate_big_five_scores(personality_values)
        
        # 格式化性格字典
        personality_dict = self._format_personality_dict(personality_values, big_five_scores)
        
        # 添加到对话数据
        dialogue_copy = dialogue.copy()
        dialogue_copy['personality'] = personality_dict
        
        return dialogue_copy
    