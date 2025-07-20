import json
import random
from typing import Dict, List, Any, Optional, Tuple, Union
import numpy as np
from collections import OrderedDict
from .personality_framework import PersonalityFramework

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
    
    def get_linguistic_markers(self, facet_code: str) -> List[str]:
        """
        获取特定子特质的语言标记
        
        Args:
            facet_code: 子特质代码，如 'O1', 'C2' 等
            
        Returns:
            该子特质的语言标记列表
        """
        return self.personality_framework.get_trait_markers(facet_code)
    
    def get_all_facets_with_markers(self) -> Dict[str, Any]:
        """
        获取所有子特质及其语言标记
        
        Returns:
            包含所有子特质和语言标记的字典
        """
        return self.personality_framework.get_all_traits_with_markers()
    
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
    
    def batch_add_random_personality(self, dialogues: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        为多个对话批量添加随机性格标签
        
        Args:
            dialogues: 对话数据列表
            
        Returns:
            添加了性格标签的对话数据列表
        """
        processed_dialogues = []
        
        for i, dialogue in enumerate(dialogues):
            try:
                processed_dialogue = self.add_random_personality(dialogue)
                processed_dialogues.append(processed_dialogue)
                
                # 每处理100个对话打印一次进度
                if (i + 1) % 100 == 0:
                    print(f"已处理 {i + 1}/{len(dialogues)} 个对话")
                    
            except Exception as e:
                print(f"处理对话 {dialogue.get('dialogue_id', 'unknown')} 时出错: {str(e)}")
                # 即使出错也要保留原始对话
                processed_dialogues.append(dialogue)
        
        print(f"批量处理完成，共处理 {len(processed_dialogues)} 个对话")
        return processed_dialogues
    
    def batch_add_specified_personality(self, dialogues: List[Dict[str, Any]], 
                                      personality_values: Dict[str, float]) -> List[Dict[str, Any]]:
        """
        为多个对话批量添加相同的指定性格标签
        
        Args:
            dialogues: 对话数据列表
            personality_values: 包含30个子特质分数的字典
            
        Returns:
            添加了性格标签的对话数据列表
        """
        # 验证性格值
        if not self._validate_personality_values(personality_values):
            raise ValueError("提供的性格值无效")
        
        processed_dialogues = []
        
        for i, dialogue in enumerate(dialogues):
            try:
                processed_dialogue = self.add_specified_personality(dialogue, personality_values)
                processed_dialogues.append(processed_dialogue)
                
                # 每处理100个对话打印一次进度
                if (i + 1) % 100 == 0:
                    print(f"已处理 {i + 1}/{len(dialogues)} 个对话")
                    
            except Exception as e:
                print(f"处理对话 {dialogue.get('dialogue_id', 'unknown')} 时出错: {str(e)}")
                # 即使出错也要保留原始对话
                processed_dialogues.append(dialogue)
        
        print(f"批量处理完成，共处理 {len(processed_dialogues)} 个对话")
        return processed_dialogues
    
    def parse_personality_string(self, personality_string: str) -> Tuple[Dict[str, float], Dict[str, float]]:
        """
        解析性格字符串，提取大五人格和子特质分数
        
        Args:
            personality_string: 格式化的性格字符串
            
        Returns:
            (大五人格分数字典, 子特质分数字典)
        """
        if not personality_string:
            return {}, {}
        
        big_five_scores = {}
        facet_scores = {}
        
        parts = personality_string.split("|")
        current_section = None
        
        for part in parts:
            if part == "BIG_FIVE:":
                current_section = "big_five"
                continue
            elif part == "FACETS:":
                current_section = "facets"
                continue
            
            if ":" in part:
                key, value = part.split(":", 1)
                try:
                    score = float(value)
                    if current_section == "big_five":
                        big_five_scores[key] = score
                    elif current_section == "facets":
                        facet_scores[key] = score
                except ValueError:
                    print(f"警告: 无法解析分数 {part}")
        
        return big_five_scores, facet_scores
    
    def parse_personality_dict(self, personality_data: Union[Dict[str, Any], str]) -> Tuple[Dict[str, float], Dict[str, float]]:
        """
        解析性格数据（支持字典格式和字符串格式），提取大五人格和子特质分数
        
        Args:
            personality_data: 结构化的性格字典或格式化的性格字符串
            
        Returns:
            (大五人格分数字典, 子特质分数字典)
        """
        if not personality_data:
            return {}, {}
        
        # 如果是字符串格式，调用原有的字符串解析方法
        if isinstance(personality_data, str):
            return self.parse_personality_string(personality_data)
        
        # 如果是字典格式，直接解析结构
        if isinstance(personality_data, dict):
            big_five_scores = {}
            facet_scores = {}
            
            # 解析大五人格主维度分数
            if "big_five" in personality_data:
                for domain, score in personality_data["big_five"].items():
                    if isinstance(score, (int, float)):
                        big_five_scores[domain] = score
            
            # 解析子特质分数
            if "facets" in personality_data:
                for domain, facets in personality_data["facets"].items():
                    if isinstance(facets, dict):
                        for facet_key, score in facets.items():
                            if isinstance(score, (int, float)):
                                # 重构完整的子特质名称
                                full_facet_name = self._get_full_facet_name(domain, facet_key)
                                if full_facet_name:
                                    facet_scores[full_facet_name] = score
            
            return big_five_scores, facet_scores
        
        return {}, {}
    
    def _get_full_facet_name(self, domain: str, facet_key: str) -> Optional[str]:
        """
        根据维度和子特质键获取完整的子特质名称
        
        Args:
            domain: 大五人格维度 (O, C, E, A, N)
            facet_key: 子特质键 (O1, O2, etc.)
            
        Returns:
            完整的子特质名称，如果未找到则返回None
        """
        if domain in self.NEO_PIR_FACETS:
            for facet in self.NEO_PIR_FACETS[domain]['facets']:
                if facet.startswith(f"{facet_key}_"):
                    return facet
        return None
    
    def get_personality_summary(self, personality_data: Union[Dict[str, Any], str]) -> Dict[str, Any]:
        """
        获取性格的摘要信息
        
        Args:
            personality_data: 结构化的性格字典或格式化的性格字符串
            
        Returns:
            包含性格摘要的字典
        """
        big_five_scores, facet_scores = self.parse_personality_dict(personality_data)
        
        if not big_five_scores or not facet_scores:
            return {"error": "无效的性格数据"}
        
        summary = {
            "big_five_scores": big_five_scores,
            "facet_scores": facet_scores,
            "dominant_traits": [],
            "recessive_traits": [],
            "personality_description": []
        }
        
        # 找出主导特质和隐性特质
        for trait, score in big_five_scores.items():
            if score >= 0.7:
                summary["dominant_traits"].append(f"高{trait}")
            elif score <= 0.3:
                summary["recessive_traits"].append(f"低{trait}")
        
        # 生成性格描述
        trait_descriptions = {
            "Openness": ("富有想象力，喜欢新奇", "保守，偏好熟悉的事物"),
            "Conscientiousness": ("有组织，负责任", "随性，灵活"),
            "Extraversion": ("外向，活跃", "内向，安静"),
            "Agreeableness": ("友善，信任他人", "竞争性，怀疑"),
            "Neuroticism": ("情绪不稳定，容易焦虑", "情绪稳定，平静")
        }
        
        for trait, score in big_five_scores.items():
            if trait in trait_descriptions:
                if score >= 0.6:
                    summary["personality_description"].append(trait_descriptions[trait][0])
                elif score <= 0.4:
                    summary["personality_description"].append(trait_descriptions[trait][1])
        
        return summary
    
    def create_personality_template(self) -> Dict[str, float]:
        """
        创建一个性格模板，包含所有30个子特质，值设为0.5
        
        Returns:
            包含所有子特质的模板字典
        """
        template = {}
        for facet in self.all_facets:
            template[facet] = 0.5
        return template
    
    def print_personality_info(self):
        """打印NEO-PIR性格模型的详细信息"""
        print("=" * 80)
        print("NEO-PIR 大五人格模型详细信息")
        print("=" * 80)
        
        for domain, info in self.NEO_PIR_FACETS.items():
            print(f"\n{domain}. {info['name']} ({info['chinese_name']})")
            print("-" * 40)
            for i, facet in enumerate(info['facets'], 1):
                facet_name = facet.split('_', 1)[1].replace('_', ' ')
                print(f"  {facet}: {facet_name}")
        
        print(f"\n总计: {len(self.NEO_PIR_FACETS)} 个主维度，{len(self.all_facets)} 个子特质")
        print("每个特质取值范围: 0.0 - 1.0")
        print("主维度分数 = 该维度下6个子特质的平均值")
        print("=" * 80)


# 使用示例和测试功能
def main():
    """演示PersonalityAdder的使用方法"""
    
    print("PersonalityAdder 使用演示")
    print("=" * 60)
    
    # 初始化性格添加器
    personality_adder = PersonalityAdder(random_seed=42)
    
    # 打印性格模型信息
    personality_adder.print_personality_info()
    
    # 创建示例对话数据
    sample_dialogue = {
        'dialogue_id': 'test_001',
        'services': ['Restaurants_1'],
        'personality': '',
        'turns': [
            {'speaker': 'USER', 'utterance': 'Hello', 'transformed_utterance': ''},
            {'speaker': 'SYSTEM', 'utterance': 'Hi there!', 'transformed_utterance': ''}
        ]
    }
    
    print("\n" + "=" * 60)
    print("功能测试")
    print("=" * 60)
    
    # 测试1: 添加随机性格
    print("\n1. 添加随机性格:")
    dialogue_with_random = personality_adder.add_random_personality(sample_dialogue)
    personality_string = dialogue_with_random['personality']
    print(f"生成的性格字符串长度: {len(personality_string)} 字符")
    
    # 解析并显示性格摘要
    summary = personality_adder.get_personality_summary(personality_string)
    print("\n性格摘要:")
    print(f"  大五人格分数: {summary['big_five_scores']}")
    print(f"  主导特质: {summary.get('dominant_traits', [])}")
    print(f"  隐性特质: {summary.get('recessive_traits', [])}")
    print(f"  性格描述: {', '.join(summary.get('personality_description', []))}")
    
    # 测试2: 添加指定性格
    print("\n2. 添加指定性格:")
    
    # 创建一个高外向性、高宜人性的性格
    custom_personality = personality_adder.create_personality_template()
    
    # 设置外向性相关特质为高分
    for facet in personality_adder.NEO_PIR_FACETS['E']['facets']:
        custom_personality[facet] = 0.8
    
    # 设置宜人性相关特质为高分
    for facet in personality_adder.NEO_PIR_FACETS['A']['facets']:
        custom_personality[facet] = 0.8
    
    # 设置神经质相关特质为低分
    for facet in personality_adder.NEO_PIR_FACETS['N']['facets']:
        custom_personality[facet] = 0.2
    
    dialogue_with_custom = personality_adder.add_specified_personality(sample_dialogue, custom_personality)
    custom_summary = personality_adder.get_personality_summary(dialogue_with_custom['personality'])
    
    print("自定义性格摘要:")
    print(f"  大五人格分数: {custom_summary['big_five_scores']}")
    print(f"  性格描述: {', '.join(custom_summary.get('personality_description', []))}")
    
    # 测试3: 批量处理
    print("\n3. 批量处理测试:")
    
    # 创建多个示例对话
    sample_dialogues = []
    for i in range(5):
        dialogue = sample_dialogue.copy()
        dialogue['dialogue_id'] = f'test_{i:03d}'
        sample_dialogues.append(dialogue)
    
    # 批量添加随机性格
    processed_dialogues = personality_adder.batch_add_random_personality(sample_dialogues)
    
    print(f"批量处理结果: {len(processed_dialogues)} 个对话")
    for dialogue in processed_dialogues[:3]:  # 只显示前3个
        summary = personality_adder.get_personality_summary(dialogue['personality'])
        print(f"  {dialogue['dialogue_id']}: {summary['big_five_scores']}")
        print(dialogue)
    
    print("\n✨ PersonalityAdder 演示完成!")


if __name__ == "__main__":
    main()
