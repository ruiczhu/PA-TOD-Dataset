import json
import pandas as pd
import numpy as np
import os
import sys
from typing import Dict, List, Tuple

# 添加项目根目录到路径
project_root = "/Users/zhuruichen/PycharmProjects/PA-TOD-Dataset"
if project_root not in sys.path:
    sys.path.append(project_root)

from evaluator.bert_personality import PersonalityDetector


class SimplifiedPersonalityComparator:
    """
    简化的人格对比评估器
    对比转换后对话的模型评分与原始对话、转换目标值、LLM评分值
    """
    def __init__(self, json_file_path=None, model_path=None, output_dir=None):
        self.json_file_path = json_file_path
        self.output_dir = output_dir or "."
        
        # 使用相对路径或尝试多个可能的路径
        if model_path is None:
            # 尝试几个可能的模型路径
            possible_paths = [
                "../Model/Bert_personality",  # 相对于当前项目的Model目录
                "./Model/Bert_personality",   # 当前目录下的Model目录
                "Model/Bert_personality",     # 直接在项目根目录下的Model目录
                "/Users/zhuruichen/PycharmProjects/PA-TOD-Dataset/Model/Bert_personality",
                "bert-base-uncased"  # 如果本地模型不存在，使用预训练模型
            ]
            
            for path in possible_paths:
                try:
                    self.personality_detector = PersonalityDetector(model_path=path)
                    print(f"Successfully loaded model from: {path}")
                    break
                except Exception as e:
                    print(f"Failed to load model from {path}: {str(e)}")
                    continue
            else:
                # 如果所有路径都失败，使用默认的BERT模型
                print("Warning: Using default BERT model. Personality detection may not be accurate.")
                self.personality_detector = PersonalityDetector(model_path="bert-base-uncased")
        else:
            self.personality_detector = PersonalityDetector(model_path=model_path)

    def load_json_dialogues(self, file_path):
        """加载JSON格式的对话文件"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                dialogues = json.load(f)
            return dialogues
        except Exception as e:
            print(f"Error loading JSON file: {str(e)}")
            return None

    def extract_dialogue_texts_and_scores(self, dialogues):
        """从JSON对话数据中提取文本和已有的人格评分"""
        results = []
        
        for dialogue in dialogues:
            dialogue_id = dialogue.get('dialogue_id', 'unknown')
            
            # 提取对话文本
            original_texts = []
            transformed_texts = []
            
            for turn in dialogue.get('turns', []):
                utterance = turn.get('utterance', '').strip()
                transformed_utterance = turn.get('transformed_utterance', '').strip()
                
                if utterance:
                    original_texts.append(utterance)
                if transformed_utterance:
                    transformed_texts.append(transformed_utterance)
            
            # 合并为完整的对话文本
            combined_original = " ".join(original_texts)
            combined_transformed = " ".join(transformed_texts)
            
            # 提取已有的人格评分
            transformation_quality = dialogue.get('transformation_quality', {})
            
            # 1. 转换目标值 (personality_data)
            target_scores = transformation_quality.get('personality_data', {})
            
            # 2. LLM评分值 (transformed_big_five)
            llm_scores = transformation_quality.get('transformed_big_five', {})
            
            results.append({
                'dialogue_id': dialogue_id,
                'original_text': combined_original,
                'transformed_text': combined_transformed,
                'turn_count': len(dialogue.get('turns', [])),
                'target_O': float(target_scores.get('O', 0)),
                'target_C': float(target_scores.get('C', 0)),
                'target_E': float(target_scores.get('E', 0)),
                'target_A': float(target_scores.get('A', 0)),
                'target_N': float(target_scores.get('N', 0)),
                'llm_O': float(llm_scores.get('O', 0)),
                'llm_C': float(llm_scores.get('C', 0)),
                'llm_E': float(llm_scores.get('E', 0)),
                'llm_A': float(llm_scores.get('A', 0)),
                'llm_N': float(llm_scores.get('N', 0))
            })
        
        return results

    def _compute_personality_scores(self, text):
        """计算文本的人格得分"""
        if not text or len(text.strip()) == 0:
            return {
                'Extroversion': 0.0,
                'Neuroticism': 0.0,
                'Agreeableness': 0.0,
                'Conscientiousness': 0.0,
                'Openness': 0.0
            }
        return self.personality_detector.personality_detection(text)

    def _compute_absolute_difference(self, score1, score2):
        """计算两个分数的绝对差值"""
        return abs(score1 - score2)

    def compare_personality_scores(self, file_path):
        """
        比较转换后对话的模型评分与三个参考值
        """
        # 加载对话数据
        dialogues = self.load_json_dialogues(file_path)
        if not dialogues:
            return None
        
        # 提取对话文本和已有评分
        dialogue_data = self.extract_dialogue_texts_and_scores(dialogues)
        
        results = []
        
        for data in dialogue_data:
            dialogue_id = data['dialogue_id']
            original_text = data['original_text']
            transformed_text = data['transformed_text']
            turn_count = data['turn_count']
            
            # 计算模型评分
            original_model_scores = self._compute_personality_scores(original_text)
            transformed_model_scores = self._compute_personality_scores(transformed_text)
            
            # 获取目标值和LLM评分
            target_scores = {
                'Openness': data['target_O'],
                'Conscientiousness': data['target_C'],
                'Extroversion': data['target_E'],
                'Agreeableness': data['target_A'],
                'Neuroticism': data['target_N']
            }
            
            llm_scores = {
                'Openness': data['llm_O'],
                'Conscientiousness': data['llm_C'],
                'Extroversion': data['llm_E'],
                'Agreeableness': data['llm_A'],
                'Neuroticism': data['llm_N']
            }
            
            # 计算差异 (转换后模型评分作为基准)
            # 1. 转换后 vs 原始对话
            diff_vs_original = {}
            # 2. 转换后 vs 转换目标值
            diff_vs_target = {}
            # 3. 转换后 vs LLM评分值
            diff_vs_llm = {}
            
            for dimension in ['Openness', 'Conscientiousness', 'Extroversion', 'Agreeableness', 'Neuroticism']:
                transformed_score = transformed_model_scores[dimension]
                original_score = original_model_scores[dimension]
                target_score = target_scores[dimension]
                llm_score = llm_scores[dimension]
                
                diff_vs_original[f'{dimension}_vs_original'] = self._compute_absolute_difference(transformed_score, original_score)
                diff_vs_target[f'{dimension}_vs_target'] = self._compute_absolute_difference(transformed_score, target_score)
                diff_vs_llm[f'{dimension}_vs_llm'] = self._compute_absolute_difference(transformed_score, llm_score)
            
            # 计算平均差异
            avg_diff_vs_original = np.mean(list(diff_vs_original.values()))
            avg_diff_vs_target = np.mean(list(diff_vs_target.values()))
            avg_diff_vs_llm = np.mean(list(diff_vs_llm.values()))
            
            result = {
                'dialogue_id': dialogue_id,
                'turn_count': turn_count,
                
                # 原始对话模型评分
                'original_model_O': original_model_scores['Openness'],
                'original_model_C': original_model_scores['Conscientiousness'],
                'original_model_E': original_model_scores['Extroversion'],
                'original_model_A': original_model_scores['Agreeableness'],
                'original_model_N': original_model_scores['Neuroticism'],
                
                # 转换后对话模型评分 (基准)
                'transformed_model_O': transformed_model_scores['Openness'],
                'transformed_model_C': transformed_model_scores['Conscientiousness'],
                'transformed_model_E': transformed_model_scores['Extroversion'],
                'transformed_model_A': transformed_model_scores['Agreeableness'],
                'transformed_model_N': transformed_model_scores['Neuroticism'],
                
                # 转换目标值
                'target_O': target_scores['Openness'],
                'target_C': target_scores['Conscientiousness'],
                'target_E': target_scores['Extroversion'],
                'target_A': target_scores['Agreeableness'],
                'target_N': target_scores['Neuroticism'],
                
                # LLM评分值
                'llm_O': llm_scores['Openness'],
                'llm_C': llm_scores['Conscientiousness'],
                'llm_E': llm_scores['Extroversion'],
                'llm_A': llm_scores['Agreeableness'],
                'llm_N': llm_scores['Neuroticism'],
                
                # 差异1: 转换后 vs 原始对话
                'O_vs_original': diff_vs_original['Openness_vs_original'],
                'C_vs_original': diff_vs_original['Conscientiousness_vs_original'],
                'E_vs_original': diff_vs_original['Extroversion_vs_original'],
                'A_vs_original': diff_vs_original['Agreeableness_vs_original'],
                'N_vs_original': diff_vs_original['Neuroticism_vs_original'],
                'avg_diff_vs_original': avg_diff_vs_original,
                
                # 差异2: 转换后 vs 转换目标值
                'O_vs_target': diff_vs_target['Openness_vs_target'],
                'C_vs_target': diff_vs_target['Conscientiousness_vs_target'],
                'E_vs_target': diff_vs_target['Extroversion_vs_target'],
                'A_vs_target': diff_vs_target['Agreeableness_vs_target'],
                'N_vs_target': diff_vs_target['Neuroticism_vs_target'],
                'avg_diff_vs_target': avg_diff_vs_target,
                
                # 差异3: 转换后 vs LLM评分值
                'O_vs_llm': diff_vs_llm['Openness_vs_llm'],
                'C_vs_llm': diff_vs_llm['Conscientiousness_vs_llm'],
                'E_vs_llm': diff_vs_llm['Extroversion_vs_llm'],
                'A_vs_llm': diff_vs_llm['Agreeableness_vs_llm'],
                'N_vs_llm': diff_vs_llm['Neuroticism_vs_llm'],
                'avg_diff_vs_llm': avg_diff_vs_llm
            }
            
            results.append(result)
        
        return pd.DataFrame(results)

    def generate_comparison_summary(self, results_df):
        """生成对比汇总统计"""
        if results_df is None or results_df.empty:
            return None
        
        summary = {
            'total_dialogues': len(results_df),
            'avg_turn_count': results_df['turn_count'].mean(),
            'comparison_results': {
                'vs_original': {
                    'avg_difference': results_df['avg_diff_vs_original'].mean(),
                    'median_difference': results_df['avg_diff_vs_original'].median(),
                    'std_difference': results_df['avg_diff_vs_original'].std(),
                    'min_difference': results_df['avg_diff_vs_original'].min(),
                    'max_difference': results_df['avg_diff_vs_original'].max()
                },
                'vs_target': {
                    'avg_difference': results_df['avg_diff_vs_target'].mean(),
                    'median_difference': results_df['avg_diff_vs_target'].median(),
                    'std_difference': results_df['avg_diff_vs_target'].std(),
                    'min_difference': results_df['avg_diff_vs_target'].min(),
                    'max_difference': results_df['avg_diff_vs_target'].max()
                },
                'vs_llm': {
                    'avg_difference': results_df['avg_diff_vs_llm'].mean(),
                    'median_difference': results_df['avg_diff_vs_llm'].median(),
                    'std_difference': results_df['avg_diff_vs_llm'].std(),
                    'min_difference': results_df['avg_diff_vs_llm'].min(),
                    'max_difference': results_df['avg_diff_vs_llm'].max()
                }
            }
        }
        
        # 各维度详细统计
        dimensions = ['O', 'C', 'E', 'A', 'N']
        dimension_names = ['开放性', '尽责性', '外向性', '宜人性', '神经质']
        
        summary['dimension_details'] = {}
        for dim, name in zip(dimensions, dimension_names):
            summary['dimension_details'][name] = {
                'vs_original': {
                    'mean': results_df[f'{dim}_vs_original'].mean(),
                    'std': results_df[f'{dim}_vs_original'].std()
                },
                'vs_target': {
                    'mean': results_df[f'{dim}_vs_target'].mean(),
                    'std': results_df[f'{dim}_vs_target'].std()
                },
                'vs_llm': {
                    'mean': results_df[f'{dim}_vs_llm'].mean(),
                    'std': results_df[f'{dim}_vs_llm'].std()
                }
            }
        
        return summary

    def save_results(self, results_df, output_path):
        """保存对比结果到CSV文件"""
        if results_df is None or results_df.empty:
            print("No results to save.")
            return False
        
        try:
            results_df.to_csv(output_path, index=False, encoding='utf-8')
            print(f"Results saved to: {output_path}")
            return True
        except Exception as e:
            print(f"Error saving results: {str(e)}")
            return False
    
    def run_comparison(self):
        """运行完整的对比分析流程"""
        if not self.json_file_path:
            print("Error: No JSON file path provided")
            return False
        
        if not os.path.exists(self.json_file_path):
            print(f"Error: JSON file not found: {self.json_file_path}")
            return False
        
        print("开始人格评分对比分析...")
        results_df = self.compare_personality_scores(self.json_file_path)
        
        if results_df is not None:
            print(f"对比分析完成！共处理了 {len(results_df)} 个对话。")
            
            # 生成汇总统计
            summary = self.generate_comparison_summary(results_df)
            
            print("\n=== 对比分析汇总 ===")
            print(f"总对话数: {summary['total_dialogues']}")
            print(f"平均轮次数: {summary['avg_turn_count']:.2f}")
            
            print("\n整体对比结果:")
            print(f"转换后 vs 原始对话: {summary['comparison_results']['vs_original']['avg_difference']:.4f} (±{summary['comparison_results']['vs_original']['std_difference']:.4f})")
            print(f"转换后 vs 转换目标值: {summary['comparison_results']['vs_target']['avg_difference']:.4f} (±{summary['comparison_results']['vs_target']['std_difference']:.4f})")
            print(f"转换后 vs LLM评分值: {summary['comparison_results']['vs_llm']['avg_difference']:.4f} (±{summary['comparison_results']['vs_llm']['std_difference']:.4f})")
            
            print("\n各维度详细对比:")
            for dim_name, stats in summary['dimension_details'].items():
                print(f"{dim_name}:")
                print(f"  vs原始: {stats['vs_original']['mean']:.4f} (±{stats['vs_original']['std']:.4f})")
                print(f"  vs目标: {stats['vs_target']['mean']:.4f} (±{stats['vs_target']['std']:.4f})")
                print(f"  vsLLM: {stats['vs_llm']['mean']:.4f} (±{stats['vs_llm']['std']:.4f})")
                print()
            
            # 保存结果
            output_path = os.path.join(self.output_dir, "personality_comparison_results.csv")
            success = self.save_results(results_df, output_path)
            return success
        else:
            print("对比分析失败。")
            return False


def main():
    """示例用法"""
    # 初始化对比器
    comparator = SimplifiedPersonalityComparator()
    
    # 评估JSON对话文件
    file_path = "/Users/zhuruichen/PycharmProjects/PA-TOD-Dataset/output/mAPipelineOutput/rs_42_d_100/complete_transformed_dialogues.json"
    
    print("开始人格评分对比分析...")
    results_df = comparator.compare_personality_scores(file_path)
    
    if results_df is not None:
        print(f"对比分析完成！共处理了 {len(results_df)} 个对话。")
        
        # 生成汇总统计
        summary = comparator.generate_comparison_summary(results_df)
        if summary:
            print("\n=== 对比分析汇总 ===")
            print(f"总对话数: {summary['total_dialogues']}")
            print(f"平均轮次数: {summary['avg_turn_count']:.2f}")
            
            print("\n三种对比的平均差异:")
            print(f"转换后 vs 原始对话: {summary['comparison_results']['vs_original']['avg_difference']:.4f}")
            print(f"转换后 vs 转换目标值: {summary['comparison_results']['vs_target']['avg_difference']:.4f}")
            print(f"转换后 vs LLM评分值: {summary['comparison_results']['vs_llm']['avg_difference']:.4f}")
            
            print("\n各维度详细对比:")
            for dim_name, stats in summary['dimension_details'].items():
                print(f"{dim_name}:")
                print(f"  vs原始: {stats['vs_original']['mean']:.4f} (±{stats['vs_original']['std']:.4f})")
                print(f"  vs目标: {stats['vs_target']['mean']:.4f} (±{stats['vs_target']['std']:.4f})")
                print(f"  vsLLM: {stats['vs_llm']['mean']:.4f} (±{stats['vs_llm']['std']:.4f})")
                print()
        
        # 保存结果
        output_path = "/Users/zhuruichen/PycharmProjects/PA-TOD-Dataset/output/mAPipelineOutput/rs_42_d_100/personality_comparison_results.csv"
        comparator.save_results(results_df, output_path)
    else:
        print("对比分析失败。")


if __name__ == "__main__":
    main()
