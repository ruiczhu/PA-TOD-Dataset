#!/usr/bin/env python3
"""
简化人格对比结果可视化脚本
用于可视化转换后对话与三个参考值的对比结果
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class ComparisonResultsVisualizer:
    def __init__(self, results_path):
        """
        初始化可视化器
        
        Args:
            results_path: 对比结果CSV文件路径
        """
        self.results_path = results_path
        self.df = None
        self.output_dir = os.path.dirname(results_path)
        
    def load_results(self):
        """加载对比结果"""
        try:
            self.df = pd.read_csv(self.results_path)
            print(f"成功加载 {len(self.df)} 个对话的对比结果")
            return True
        except Exception as e:
            print(f"加载结果文件失败: {str(e)}")
            return False
    
    def plot_average_differences_comparison(self):
        """绘制三种对比的平均差异对比图"""
        if self.df is None:
            print("请先加载数据")
            return
        
        # 准备数据
        comparison_types = ['vs_original', 'vs_target', 'vs_llm']
        comparison_labels = ['vs 原始对话', 'vs 转换目标值', 'vs LLM评分值']
        avg_columns = ['avg_diff_vs_original', 'avg_diff_vs_target', 'avg_diff_vs_llm']
        
        avg_diffs = [self.df[col].mean() for col in avg_columns]
        std_diffs = [self.df[col].std() for col in avg_columns]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # 绘制柱状图
        bars = ax.bar(comparison_labels, avg_diffs, yerr=std_diffs, 
                     capsize=5, alpha=0.8, color=['skyblue', 'lightgreen', 'lightcoral'])
        
        # 添加数值标签
        for bar, avg_diff in zip(bars, avg_diffs):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                   f'{avg_diff:.4f}', ha='center', va='bottom', fontweight='bold')
        
        ax.set_title('转换后对话模型评分的三种对比平均差异', fontsize=16, fontweight='bold')
        ax.set_ylabel('平均绝对差异', fontsize=12)
        ax.set_xlabel('对比类型', fontsize=12)
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'average_differences_comparison.png'), 
                   dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_dimension_comparison_heatmap(self):
        """绘制各维度对比差异的热力图"""
        if self.df is None:
            print("请先加载数据")
            return
        
        # 准备数据
        dimensions = ['O', 'C', 'E', 'A', 'N']
        dimension_names = ['开放性', '尽责性', '外向性', '宜人性', '神经质']
        comparison_types = ['vs_original', 'vs_target', 'vs_llm']
        comparison_labels = ['vs 原始对话', 'vs 转换目标值', 'vs LLM评分值']
        
        # 构建热力图数据
        heatmap_data = []
        for comp_type in comparison_types:
            row = []
            for dim in dimensions:
                col_name = f'{dim}_{comp_type}'
                row.append(self.df[col_name].mean())
            heatmap_data.append(row)
        
        heatmap_df = pd.DataFrame(heatmap_data, 
                                 columns=dimension_names, 
                                 index=comparison_labels)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # 绘制热力图
        sns.heatmap(heatmap_df, annot=True, fmt='.4f', cmap='YlOrRd', 
                   cbar_kws={'label': '平均绝对差异'}, ax=ax)
        
        ax.set_title('各人格维度对比差异热力图', fontsize=16, fontweight='bold')
        ax.set_xlabel('人格维度', fontsize=12)
        ax.set_ylabel('对比类型', fontsize=12)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'dimension_comparison_heatmap.png'), 
                   dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_difference_distributions(self):
        """绘制三种对比差异的分布图"""
        if self.df is None:
            print("请先加载数据")
            return
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        comparison_data = [
            ('avg_diff_vs_original', 'vs 原始对话', 'skyblue'),
            ('avg_diff_vs_target', 'vs 转换目标值', 'lightgreen'),
            ('avg_diff_vs_llm', 'vs LLM评分值', 'lightcoral')
        ]
        
        for i, (col, label, color) in enumerate(comparison_data):
            axes[i].hist(self.df[col], bins=20, alpha=0.7, color=color, edgecolor='black')
            axes[i].set_title(f'{label}差异分布', fontsize=14)
            axes[i].set_xlabel('平均绝对差异', fontsize=12)
            axes[i].set_ylabel('频次', fontsize=12)
            axes[i].grid(True, alpha=0.3)
            
            # 添加统计线
            mean_val = self.df[col].mean()
            median_val = self.df[col].median()
            axes[i].axvline(mean_val, color='red', linestyle='--', label=f'均值: {mean_val:.4f}')
            axes[i].axvline(median_val, color='green', linestyle='--', label=f'中位数: {median_val:.4f}')
            axes[i].legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'difference_distributions.png'), 
                   dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_score_comparison_radar(self):
        """绘制人格得分对比雷达图（选择一个对话作为示例）"""
        if self.df is None:
            print("请先加载数据")
            return
        
        # 选择平均差异最大的对话作为示例
        max_diff_idx = self.df['avg_diff_vs_target'].idxmax()
        sample_dialogue = self.df.iloc[max_diff_idx]
        
        # 准备雷达图数据
        dimensions = ['O', 'C', 'E', 'A', 'N']
        dimension_names = ['开放性', '尽责性', '外向性', '宜人性', '神经质']
        
        # 四种得分
        original_scores = [sample_dialogue[f'original_model_{dim}'] for dim in dimensions]
        transformed_scores = [sample_dialogue[f'transformed_model_{dim}'] for dim in dimensions]
        target_scores = [sample_dialogue[f'target_{dim}'] for dim in dimensions]
        llm_scores = [sample_dialogue[f'llm_{dim}'] for dim in dimensions]
        
        # 设置雷达图
        angles = np.linspace(0, 2 * np.pi, len(dimensions), endpoint=False).tolist()
        angles += angles[:1]  # 闭合图形
        
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
        
        # 绘制四条线
        def close_data(data):
            return data + data[:1]
        
        ax.plot(angles, close_data(original_scores), 'o-', linewidth=2, label='原始对话模型评分', color='blue')
        ax.plot(angles, close_data(transformed_scores), 'o-', linewidth=2, label='转换后对话模型评分', color='red')
        ax.plot(angles, close_data(target_scores), 'o-', linewidth=2, label='转换目标值', color='green')
        ax.plot(angles, close_data(llm_scores), 'o-', linewidth=2, label='LLM评分值', color='orange')
        
        # 填充转换后对话区域
        ax.fill(angles, close_data(transformed_scores), alpha=0.25, color='red')
        
        # 设置标签
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(dimension_names)
        ax.set_ylim(0, 1)
        ax.set_title(f'人格得分对比雷达图\n(对话ID: {sample_dialogue["dialogue_id"]})', 
                    fontsize=16, fontweight='bold', pad=20)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
        ax.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'score_comparison_radar.png'), 
                   dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_correlation_analysis(self):
        """绘制不同评分之间的相关性分析"""
        if self.df is None:
            print("请先加载数据")
            return
        
        dimensions = ['O', 'C', 'E', 'A', 'N']
        dimension_names = ['开放性', '尽责性', '外向性', '宜人性', '神经质']
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        for i, (dim, name) in enumerate(zip(dimensions, dimension_names)):
            ax = axes[i]
            
            # 绘制转换后模型评分与其他三种评分的相关性
            transformed_col = f'transformed_model_{dim}'
            original_col = f'original_model_{dim}'
            target_col = f'target_{dim}'
            llm_col = f'llm_{dim}'
            
            # 散点图
            ax.scatter(self.df[target_col], self.df[transformed_col], 
                      alpha=0.6, label='vs 转换目标值', color='green', s=30)
            ax.scatter(self.df[llm_col], self.df[transformed_col], 
                      alpha=0.6, label='vs LLM评分值', color='orange', s=30)
            ax.scatter(self.df[original_col], self.df[transformed_col], 
                      alpha=0.6, label='vs 原始对话', color='blue', s=30)
            
            # 添加对角线
            min_val = min(self.df[[transformed_col, original_col, target_col, llm_col]].min())
            max_val = max(self.df[[transformed_col, original_col, target_col, llm_col]].max())
            ax.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8, linewidth=1)
            
            ax.set_xlabel('参考评分')
            ax.set_ylabel('转换后模型评分')
            ax.set_title(f'{name}相关性分析')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # 移除多余的子图
        axes[-1].remove()
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'correlation_analysis.png'), 
                   dpi=300, bbox_inches='tight')
        plt.show()
    
    def generate_summary_report(self):
        """生成汇总报告"""
        if self.df is None:
            print("请先加载数据")
            return
        
        print("\n" + "="*80)
        print("人格评分对比分析汇总报告")
        print("="*80)
        
        print(f"📊 总对话数: {len(self.df)}")
        print(f"🔢 平均轮次数: {self.df['turn_count'].mean():.2f}")
        
        # 三种对比的整体统计
        print("\n三种对比的平均差异统计:")
        print("-" * 60)
        comparisons = [
            ('avg_diff_vs_original', '转换后 vs 原始对话'),
            ('avg_diff_vs_target', '转换后 vs 转换目标值'),
            ('avg_diff_vs_llm', '转换后 vs LLM评分值')
        ]
        
        for col, name in comparisons:
            print(f"{name}:")
            print(f"  平均差异: {self.df[col].mean():.4f}")
            print(f"  中位数差异: {self.df[col].median():.4f}")
            print(f"  标准差: {self.df[col].std():.4f}")
            print(f"  最小差异: {self.df[col].min():.4f}")
            print(f"  最大差异: {self.df[col].max():.4f}")
            print()
        
        # 找出表现最好和最差的对话
        best_vs_target_idx = self.df['avg_diff_vs_target'].idxmin()
        worst_vs_target_idx = self.df['avg_diff_vs_target'].idxmax()
        
        print("与转换目标值对比结果:")
        print(f"最接近目标的对话: {self.df.loc[best_vs_target_idx, 'dialogue_id']} "
              f"(差异: {self.df.loc[best_vs_target_idx, 'avg_diff_vs_target']:.4f})")
        print(f"偏离目标最大的对话: {self.df.loc[worst_vs_target_idx, 'dialogue_id']} "
              f"(差异: {self.df.loc[worst_vs_target_idx, 'avg_diff_vs_target']:.4f})")
    
    def create_all_visualizations(self):
        """创建所有可视化图表"""
        if not self.load_results():
            return
        
        print("正在生成对比分析可视化图表...")
        
        # 生成汇总报告
        self.generate_summary_report()
        
        # 生成所有图表
        self.plot_average_differences_comparison()
        self.plot_dimension_comparison_heatmap()
        self.plot_difference_distributions()
        self.plot_score_comparison_radar()
        self.plot_correlation_analysis()
        
        print(f"\n所有图表已保存到: {self.output_dir}")


def main():
    """主函数"""
    results_path = "/Users/zhuruichen/PycharmProjects/PA-TOD-Dataset/output/mAPipelineOutput/rs_42_d_100/personality_comparison_results.csv"
    
    if not os.path.exists(results_path):
        print(f"结果文件不存在: {results_path}")
        print("请先运行 simplified_personality_comparator.py 生成对比结果")
        return
    
    print("人格评分对比结果可视化工具")
    print("=" * 50)
    
    visualizer = ComparisonResultsVisualizer(results_path)
    visualizer.create_all_visualizations()


if __name__ == "__main__":
    main()
