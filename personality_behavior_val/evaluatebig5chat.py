#!/usr/bin/env python3
"""
简化的Big5人格特征评估脚本
对每个特征为True的数据中，从high和low各采样500条，总共5000条数据
使用BERT模型评分并绘制分布图
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import warnings

# 设置中文字体和忽略警告
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
warnings.filterwarnings('ignore')

# 添加项目根目录到路径
project_root = "/Users/zhuruichen/PycharmProjects/PA-TOD-Dataset"
if project_root not in sys.path:
    sys.path.append(project_root)

from evaluator.bert_personality import PersonalityDetector


class SimpleBig5Evaluator:
    """简化的Big5评估器"""
    
    def __init__(self):
        # 数据路径
        self.data_path = "/Users/zhuruichen/PycharmProjects/PA-TOD-Dataset/personality_behavior_val/processed_data_encoded.csv"
        self.output_dir = "/Users/zhuruichen/PycharmProjects/PA-TOD-Dataset/personality_behavior_val"
        
        # 初始化BERT模型
        model_path = "/Users/zhuruichen/PycharmProjects/PA-TOD-Dataset/Model/Bert_personality"
        self.detector = PersonalityDetector(model_path=model_path)
        
        # 特征定义
        self.traits = {
            'trait_agreeableness': ('bert_A', '宜人性'),
            'trait_conscientiousness': ('bert_C', '尽责性'),
            'trait_extraversion': ('bert_E', '外向性'),
            'trait_neuroticism': ('bert_N', '神经质'),
            'trait_openness': ('bert_O', '开放性')
        }
        
        print("✅ SimpleBig5Evaluator 初始化完成")
    
    def load_and_sample_data(self):
        """加载数据并执行采样"""
        print("📊 加载数据集...")
        df = pd.read_csv(self.data_path)
        print(f"原始数据集大小: {len(df)} 条")
        
        sampled_data = []
        
        print("\n🎯 开始采样...")
        for trait_col, (bert_col, trait_name) in self.traits.items():
            print(f"\n处理 {trait_name} ({trait_col}):")
            
            # 获取该特征为True的数据
            trait_true = df[df[trait_col] == True]
            print(f"  {trait_name} 为True的总数: {len(trait_true)}")
            
            # 分别获取high和low的数据
            trait_high = trait_true[trait_true['level_high'] == True]
            trait_low = trait_true[trait_true['level_low'] == True]
            
            print(f"  High level: {len(trait_high)} 条")
            print(f"  Low level: {len(trait_low)} 条")
            
            # 各采样500条
            sample_size = 500
            high_sample = trait_high.sample(n=min(sample_size, len(trait_high)), random_state=42)
            low_sample = trait_low.sample(n=min(sample_size, len(trait_low)), random_state=42)
            
            print(f"  采样结果: High={len(high_sample)}, Low={len(low_sample)}")
            
            # 合并该特征的采样数据
            trait_sample = pd.concat([high_sample, low_sample], ignore_index=True)
            trait_sample['target_trait'] = trait_col  # 标记目标特征
            sampled_data.append(trait_sample)
        
        # 合并所有采样数据
        self.sampled_df = pd.concat(sampled_data, ignore_index=True)
        print(f"\n📊 总采样数据: {len(self.sampled_df)} 条")
        
        return True
    
    def evaluate_bert_scores(self):
        """使用BERT模型评估人格得分"""
        print("\n🤖 开始BERT人格评估...")
        
        bert_scores = []
        total = len(self.sampled_df)
        
        for idx, row in self.sampled_df.iterrows():
            if idx % 100 == 0:
                print(f"进度: {idx+1}/{total} ({(idx+1)/total*100:.1f}%)")
            
            text = str(row['train_output']).strip()
            
            try:
                # 获取BERT人格得分
                scores = self.detector.personality_detection(text)
                bert_scores.append({
                    'idx': idx,
                    'bert_O': scores['Openness'],
                    'bert_C': scores['Conscientiousness'],
                    'bert_E': scores['Extroversion'],
                    'bert_A': scores['Agreeableness'],
                    'bert_N': scores['Neuroticism']
                })
            except Exception as e:
                print(f"⚠️ 处理索引 {idx} 时出错: {str(e)}")
                # 使用默认值
                bert_scores.append({
                    'idx': idx,
                    'bert_O': 0.5, 'bert_C': 0.5, 'bert_E': 0.5, 
                    'bert_A': 0.5, 'bert_N': 0.5
                })
        
        # 将BERT得分添加到数据中
        bert_df = pd.DataFrame(bert_scores)
        self.sampled_df = pd.concat([self.sampled_df.reset_index(drop=True), 
                                   bert_df[['bert_O', 'bert_C', 'bert_E', 'bert_A', 'bert_N']]], axis=1)
        
        print("✅ BERT评估完成！")
        return True
    
    def create_distribution_plots(self):
        """为每个特征创建分布图"""
        print("\n📊 生成分布图...")
        
        for trait_col, (bert_col, trait_name) in self.traits.items():
            print(f"生成 {trait_name} 分布图...")
            
            # 获取该特征的数据（只有True标签的1000条数据）
            trait_data = self.sampled_df[self.sampled_df['target_trait'] == trait_col]
            
            if len(trait_data) == 0:
                print(f"⚠️ {trait_name} 没有数据，跳过")
                continue
            
            # 按level分组
            high_data = trait_data[trait_data['level_high'] == True][bert_col]
            low_data = trait_data[trait_data['level_low'] == True][bert_col]
            all_data = trait_data[bert_col]
            
            # 创建三个子图
            fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
            
            # 子图1: High Level分布
            if len(high_data) > 0:
                ax1.hist(high_data, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
                ax1.axvline(high_data.mean(), color='red', linestyle='--', linewidth=2)
            ax1.set_title(f'{trait_name} - High Level (n={len(high_data)})', fontsize=14, fontweight='bold')
            ax1.set_xlabel('BERT预测得分')
            ax1.set_ylabel('频次')
            ax1.grid(True, alpha=0.3)
            if len(high_data) > 0:
                ax1.text(0.05, 0.95, f'均值: {high_data.mean():.3f}\n标准差: {high_data.std():.3f}', 
                        transform=ax1.transAxes, verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            # 子图2: Low Level分布
            if len(low_data) > 0:
                ax2.hist(low_data, bins=20, alpha=0.7, color='lightcoral', edgecolor='black')
                ax2.axvline(low_data.mean(), color='red', linestyle='--', linewidth=2)
            ax2.set_title(f'{trait_name} - Low Level (n={len(low_data)})', fontsize=14, fontweight='bold')
            ax2.set_xlabel('BERT预测得分')
            ax2.set_ylabel('频次')
            ax2.grid(True, alpha=0.3)
            if len(low_data) > 0:
                ax2.text(0.05, 0.95, f'均值: {low_data.mean():.3f}\n标准差: {low_data.std():.3f}', 
                        transform=ax2.transAxes, verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            # 子图3: 整体分布
            ax3.hist(all_data, bins=20, alpha=0.7, color='lightgreen', edgecolor='black')
            ax3.axvline(all_data.mean(), color='red', linestyle='--', linewidth=2)
            ax3.set_title(f'{trait_name} - 整体分布 (n={len(all_data)})', fontsize=14, fontweight='bold')
            ax3.set_xlabel('BERT预测得分')
            ax3.set_ylabel('频次')
            ax3.grid(True, alpha=0.3)
            ax3.text(0.05, 0.95, f'均值: {all_data.mean():.3f}\n标准差: {all_data.std():.3f}', 
                    transform=ax3.transAxes, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            # 计算并显示差异
            if len(high_data) > 0 and len(low_data) > 0:
                diff = high_data.mean() - low_data.mean()
                fig.suptitle(f'{trait_name} BERT得分分布 (High-Low差异: {diff:.3f})', 
                           fontsize=16, fontweight='bold')
            else:
                fig.suptitle(f'{trait_name} BERT得分分布', fontsize=16, fontweight='bold')
            
            plt.tight_layout()
            
            # 保存图片
            save_path = os.path.join(self.output_dir, f'{trait_name}_simple_distribution.png')
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.show()
            
            print(f"✅ {trait_name} 分布图已保存: {save_path}")
    
    def save_results(self):
        """保存结果数据"""
        output_path = os.path.join(self.output_dir, 'simple_big5_results.csv')
        self.sampled_df.to_csv(output_path, index=False, encoding='utf-8')
        print(f"✅ 结果数据已保存: {output_path}")
    
    def run_evaluation(self):
        """运行完整评估流程"""
        print("🚀 开始简化Big5评估流程")
        print("="*50)
        
        # 1. 加载和采样数据
        if not self.load_and_sample_data():
            return False
        
        # 2. BERT评估
        if not self.evaluate_bert_scores():
            return False
        
        # 3. 生成分布图
        self.create_distribution_plots()
        
        # 4. 保存结果
        self.save_results()
        
        print("\n🎉 评估完成！")
        return True


def main():
    """主函数"""
    print("简化Big5人格特征评估工具")
    print("每个特征True标签中，High和Low各采样500条")
    print("="*50)
    
    evaluator = SimpleBig5Evaluator()
    success = evaluator.run_evaluation()
    
    if success:
        print("\n✨ 评估和可视化完成！")
    else:
        print("\n💥 评估失败！")


if __name__ == "__main__":
    main()
