#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
对话转换流水线

实现完整的对话数据处理流程：
1. 使用DialogueLoader加载SGD对话数据
2. 使用PersonalityAdder为对话添加性格标签
3. 导出处理后的数据为JSON格式

支持多种数据获取方式和处理配置。
"""

import json
import os
import sys
from typing import Dict, List, Any, Optional, Union
from datetime import datetime
import argparse

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from MPEAF.dialogue_loader import DialogueLoader
from MPEAF.personality_adder import PersonalityAdder


class TransformationPipeline:
    """
    对话转换流水线类
    
    负责协调整个数据处理流程：
    1. 数据加载
    2. 性格标签添加
    3. 数据导出
    """
    
    def __init__(self, 
                 data_dir: str = "datasets/sgd_processed",
                 output_dir: str = "output",
                 random_seed: Optional[int] = None,
                 load_all_data: bool = True):
        """
        初始化转换流水线
        
        Args:
            data_dir: SGD数据目录路径
            output_dir: 输出目录路径
            random_seed: 随机种子，用于确保结果可重现
            load_all_data: 是否在初始化时加载所有数据
        """
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.random_seed = random_seed
        
        # 确保输出目录存在
        os.makedirs(output_dir, exist_ok=True)
        
        print("🏗️  初始化转换流水线")
        print("=" * 60)
        print(f"数据目录: {data_dir}")
        print(f"输出目录: {output_dir}")
        print(f"随机种子: {random_seed if random_seed is not None else '未设置'}")
        
        # 初始化组件
        try:
            print("\n📚 初始化对话加载器...")
            self.dialogue_loader = DialogueLoader(
                data_dir=data_dir, 
                load_all=load_all_data
            )
            
            print("\n🎯 初始化性格添加器...")
            self.personality_adder = PersonalityAdder(random_seed=random_seed)
            
            print("\n✅ 流水线初始化完成!")
            
        except Exception as e:
            print(f"❌ 初始化失败: {str(e)}")
            raise
    
    def load_dialogues_by_id(self, dialogue_ids: List[str]) -> List[Dict[str, Any]]:
        """
        根据对话ID列表加载对话数据
        
        Args:
            dialogue_ids: 对话ID列表
            
        Returns:
            对话数据列表
        """
        print(f"\n📋 根据ID加载对话数据 (共{len(dialogue_ids)}个)")
        
        dialogues = []
        for dialogue_id in dialogue_ids:
            dialogue = self.dialogue_loader.get_dialogue_by_id(dialogue_id)
            if dialogue:
                dialogues.append(dialogue)
            else:
                print(f"⚠️  未找到对话ID: {dialogue_id}")
        
        print(f"✅ 成功加载 {len(dialogues)}/{len(dialogue_ids)} 个对话")
        return dialogues
    
    def load_dialogues_by_service(self, 
                                 service: str, 
                                 count: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        根据服务类型加载对话数据
        
        Args:
            service: 服务类型
            count: 加载数量限制
            
        Returns:
            对话数据列表
        """
        print(f"\n📋 根据服务类型加载对话数据")
        print(f"服务类型: {service}")
        print(f"数量限制: {count if count is not None else '无限制'}")
        
        dialogues = self.dialogue_loader.get_dialogues_by_service(
            service=service,
            count=count,
            random_seed=self.random_seed
        )
        
        return dialogues
    
    def load_random_dialogues(self, count: int) -> List[Dict[str, Any]]:
        """
        随机加载指定数量的对话数据
        
        Args:
            count: 要加载的对话数量
            
        Returns:
            对话数据列表
        """
        print(f"\n📋 随机加载对话数据")
        print(f"数量: {count}")
        
        dialogues = self.dialogue_loader.get_random_dialogues(
            count=count,
            random_seed=self.random_seed
        )
        
        return dialogues
    
    def add_personality_labels(self, 
                             dialogues: List[Dict[str, Any]],
                             personality_type: str = "random",
                             personality_config: Optional[Dict[str, float]] = None) -> List[Dict[str, Any]]:
        """
        为对话数据添加性格标签
        
        Args:
            dialogues: 对话数据列表
            personality_type: 性格类型 ("random" 或 "specified")
            personality_config: 指定性格配置（当personality_type为"specified"时使用）
            
        Returns:
            添加了性格标签的对话数据列表
        """
        print(f"\n🎯 添加性格标签")
        print(f"性格类型: {personality_type}")
        print(f"对话数量: {len(dialogues)}")
        
        if personality_type == "random":
            # 批量添加随机性格
            processed_dialogues = self.personality_adder.batch_add_random_personality(dialogues)
        
        elif personality_type == "specified":
            if personality_config is None:
                print("⚠️  指定性格模式但未提供性格配置，使用默认配置")
                # 创建一个示例性格配置（高外向性、高宜人性）
                personality_config = self.personality_adder.create_personality_template()
                # 设置外向性和宜人性为高分
                for facet in self.personality_adder.NEO_PIR_FACETS['E']['facets']:
                    personality_config[facet] = 0.8
                for facet in self.personality_adder.NEO_PIR_FACETS['A']['facets']:
                    personality_config[facet] = 0.75
            
            processed_dialogues = self.personality_adder.batch_add_specified_personality(
                dialogues, personality_config
            )
        
        else:
            raise ValueError(f"不支持的性格类型: {personality_type}")
        
        return processed_dialogues
    
    def export_to_json(self, 
                      dialogues: List[Dict[str, Any]], 
                      filename: Optional[str] = None,
                      include_metadata: bool = True) -> str:
        """
        将处理后的对话数据导出为JSON文件
        
        Args:
            dialogues: 处理后的对话数据
            filename: 输出文件名（可选）
            include_metadata: 是否包含元数据
            
        Returns:
            导出文件的完整路径
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"transformed_dialogues_{timestamp}.json"
        
        # 确保文件名以.json结尾
        if not filename.endswith('.json'):
            filename += '.json'
        
        output_path = os.path.join(self.output_dir, filename)
        
        print(f"\n💾 导出数据到JSON文件")
        print(f"文件路径: {output_path}")
        print(f"对话数量: {len(dialogues)}")
        
        # 准备导出数据
        export_data = {
            "dialogues": dialogues
        }
        
        # 添加元数据
        if include_metadata:
            export_data["metadata"] = {
                "total_dialogues": len(dialogues),
                "export_timestamp": datetime.now().isoformat(),
                "random_seed": self.random_seed,
                "data_source": self.data_dir,
                "pipeline_version": "1.0.0"
            }
            
            # 添加性格统计
            personality_stats = self._analyze_personality_distribution(dialogues)
            export_data["metadata"]["personality_statistics"] = personality_stats
        
        # 写入文件
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, ensure_ascii=False, indent=2)
            
            file_size = os.path.getsize(output_path) / (1024 * 1024)  # MB
            print(f"✅ 导出成功!")
            print(f"文件大小: {file_size:.2f} MB")
            
            return output_path
            
        except Exception as e:
            print(f"❌ 导出失败: {str(e)}")
            raise
    
    def _analyze_personality_distribution(self, dialogues: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        分析对话数据中的性格分布统计
        
        Args:
            dialogues: 包含性格标签的对话数据
            
        Returns:
            性格分布统计信息
        """
        if not dialogues:
            return {}
        
        # 统计大五人格维度分数
        dimension_stats = {
            'O': [], 'C': [], 'E': [], 'A': [], 'N': []
        }
        
        valid_dialogues = 0
        
        for dialogue in dialogues:
            personality = dialogue.get('personality', {})
            if isinstance(personality, dict) and 'big_five' in personality:
                valid_dialogues += 1
                for dimension, score in personality['big_five'].items():
                    if dimension in dimension_stats:
                        dimension_stats[dimension].append(score)
        
        # 计算统计信息
        stats = {
            'valid_dialogues': valid_dialogues,
            'dimension_statistics': {}
        }
        
        for dimension, scores in dimension_stats.items():
            if scores:
                dimension_name = self.personality_adder.NEO_PIR_FACETS[dimension]['chinese_name']
                stats['dimension_statistics'][dimension] = {
                    'name': self.personality_adder.NEO_PIR_FACETS[dimension]['name'],
                    'chinese_name': dimension_name,
                    'mean': round(sum(scores) / len(scores), 3),
                    'min': round(min(scores), 3),
                    'max': round(max(scores), 3),
                    'count': len(scores)
                }
        
        return stats
    
    def process_dialogues(self,
                         load_method: str,
                         load_params: Dict[str, Any],
                         personality_type: str = "random",
                         personality_config: Optional[Dict[str, float]] = None,
                         output_filename: Optional[str] = None) -> str:
        """
        执行完整的对话处理流程
        
        Args:
            load_method: 数据加载方法 ("by_id", "by_service", "random")
            load_params: 加载参数
            personality_type: 性格类型 ("random" 或 "specified")
            personality_config: 指定性格配置
            output_filename: 输出文件名
            
        Returns:
            导出文件路径
        """
        print("🚀 开始执行对话处理流程")
        print("=" * 60)
        
        # 1. 加载对话数据
        print(f"步骤 1/3: 加载对话数据")
        print(f"加载方法: {load_method}")
        
        if load_method == "by_id":
            dialogue_ids = load_params.get("dialogue_ids", [])
            dialogues = self.load_dialogues_by_id(dialogue_ids)
            
        elif load_method == "by_service":
            service = load_params.get("service")
            count = load_params.get("count")
            if not service:
                raise ValueError("使用by_service方法时必须提供service参数")
            dialogues = self.load_dialogues_by_service(service, count)
            
        elif load_method == "random":
            count = load_params.get("count", 10)
            dialogues = self.load_random_dialogues(count)
            
        else:
            raise ValueError(f"不支持的加载方法: {load_method}")
        
        if not dialogues:
            raise ValueError("没有加载到任何对话数据")
        
        # 2. 添加性格标签
        print(f"\n步骤 2/3: 添加性格标签")
        processed_dialogues = self.add_personality_labels(
            dialogues, personality_type, personality_config
        )
        
        # 3. 导出数据
        print(f"\n步骤 3/3: 导出数据")
        output_path = self.export_to_json(processed_dialogues, output_filename)
        
        print("\n" + "=" * 60)
        print("🎉 处理流程完成!")
        print(f"输出文件: {output_path}")
        
        return output_path
    
    def get_available_services(self) -> List[str]:
        """
        获取可用的服务类型列表
        
        Returns:
            服务类型列表
        """
        return self.dialogue_loader.get_all_services()
    
    def print_data_summary(self):
        """打印数据集概览信息"""
        self.dialogue_loader.print_summary()


def create_example_pipeline():
    """创建示例流水线并演示基本功能"""
    print("📋 TransformationPipeline 示例演示")
    print("=" * 80)
    
    # 创建流水线实例
    pipeline = TransformationPipeline(
        data_dir="datasets/sgd_processed_train",
        output_dir="output/pipeline_demo",
        random_seed=42
    )
    
    # 打印数据概览
    print("\n📊 数据集概览:")
    pipeline.print_data_summary()
    
    # 获取可用服务
    services = pipeline.get_available_services()
    print(f"\n🔧 可用服务类型 (共{len(services)}种):")
    for i, service in enumerate(services[:10], 1):  # 只显示前10个
        print(f"  {i}. {service}")
    if len(services) > 10:
        print(f"  ... 还有 {len(services) - 10} 个服务类型")
    
    return pipeline


def main():
    """主函数 - 命令行接口"""
    parser = argparse.ArgumentParser(description="对话转换流水线")
    
    # 基础参数
    parser.add_argument("--data-dir", default="datasets/sgd_processed_train",
                       help="SGD数据目录路径")
    parser.add_argument("--output-dir", default="output",
                       help="输出目录路径")
    parser.add_argument("--random-seed", type=int, default=42,
                       help="随机种子")
    
    # 数据加载参数
    parser.add_argument("--load-method", choices=["by_id", "by_service", "random"],
                       default="random", help="数据加载方法")
    parser.add_argument("--dialogue-ids", nargs="+",
                       help="对话ID列表 (当load-method为by_id时使用)")
    parser.add_argument("--service", help="服务类型 (当load-method为by_service时使用)")
    parser.add_argument("--count", type=int, default=10,
                       help="加载数量")
    
    # 性格处理参数
    parser.add_argument("--personality-type", choices=["random", "specified"],
                       default="random", help="性格类型")
    
    # 输出参数
    parser.add_argument("--output-filename", help="输出文件名")
    parser.add_argument("--demo", action="store_true", help="运行示例演示")
    
    args = parser.parse_args()
    
    if args.demo:
        # 运行演示
        pipeline = create_example_pipeline()
        
        # 演示不同的处理方式
        demo_configs = [
            {
                "name": "随机10个对话",
                "load_method": "random",
                "load_params": {"count": 10},
                "output_filename": "demo_random_10.json"
            }
        ]
        
        if pipeline.get_available_services():
            demo_configs.append({
                "name": f"服务类型: {pipeline.get_available_services()[0]}",
                "load_method": "by_service", 
                "load_params": {"service": pipeline.get_available_services()[0], "count": 5},
                "output_filename": "demo_service_5.json"
            })
        
        for config in demo_configs:
            print(f"\n🎯 演示: {config['name']}")
            print("-" * 40)
            
            try:
                output_path = pipeline.process_dialogues(
                    load_method=config["load_method"],
                    load_params=config["load_params"],
                    output_filename=config["output_filename"]
                )
                print(f"✅ 成功: {output_path}")
            except Exception as e:
                print(f"❌ 失败: {str(e)}")
    
    else:
        # 执行指定的处理任务
        pipeline = TransformationPipeline(
            data_dir=args.data_dir,
            output_dir=args.output_dir, 
            random_seed=args.random_seed
        )
        
        # 准备加载参数
        load_params = {"count": args.count}
        
        if args.load_method == "by_id":
            if not args.dialogue_ids:
                parser.error("使用by_id方法时必须提供--dialogue-ids参数")
            load_params["dialogue_ids"] = args.dialogue_ids
            
        elif args.load_method == "by_service":
            if not args.service:
                parser.error("使用by_service方法时必须提供--service参数")
            load_params["service"] = args.service
        
        # 执行处理
        try:
            output_path = pipeline.process_dialogues(
                load_method=args.load_method,
                load_params=load_params,
                personality_type=args.personality_type,
                output_filename=args.output_filename
            )
            print(f"\n✅ 处理完成: {output_path}")
            
        except Exception as e:
            print(f"\n❌ 处理失败: {str(e)}")
            sys.exit(1)


if __name__ == "__main__":
    main()
