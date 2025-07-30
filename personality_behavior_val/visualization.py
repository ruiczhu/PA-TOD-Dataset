import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from matplotlib.patches import Rectangle
import warnings
warnings.filterwarnings('ignore')

class PersonalityVisualization:
    """
    性格特质行为分析可视化类
    """
    
    def __init__(self, analyzer):
        """
        初始化可视化器
        
        参数:
            analyzer: PersonalityComparisonAnalyzer实例
        """
        self.analyzer = analyzer
        self.results = analyzer.results
        self.traits = analyzer.traits
        
        # 设置绘图风格
        plt.style.use('default')
        sns.set_palette("husl")
        
        # 更新指标名称映射（移除已删除的指标）
        self.metric_labels = {
            'ttr': 'Type-Token Ratio',
            'mtld': 'MTLD',
            'lexical_density': 'Lexical Density',
            'vocabulary_size': 'Vocabulary Size',
            'word_frequency_variance': 'Word Frequency Variance',
            'vader_compound': 'VADER Compound',
            'vader_positive': 'VADER Positive',
            'vader_negative': 'VADER Negative',
            'achievement_words_count': 'Achievement Words',
            'avg_utterance_length': 'Avg Utterance Length',
            'questions_count': 'Questions Count',
            'confirmation_words_count': 'Confirmation Words',
            'social_words_count': 'Social Words',
            'aggressive_words_count': 'Aggressive Words',
            'politeness_words_count': 'Politeness Words',
            'uncertainty_words_count': 'Uncertainty Words'
        }
        
        # 定义指标分类
        self.metric_categories = {
            'Lexical Complexity': ['ttr', 'mtld', 'lexical_density', 'vocabulary_size', 
                                  'word_frequency_variance', 'avg_utterance_length'],
            'Sentiment & Emotion': ['vader_compound', 'vader_positive', 'vader_negative'],
            'Social & Communication': ['social_words_count', 'politeness_words_count', 
                                     'questions_count', 'confirmation_words_count'],
            'Psychological': ['uncertainty_words_count', 'achievement_words_count', 'aggressive_words_count']
        }
    
    def create_trait_comparison_plot(self, trait, save_path=None):
        """
        为单个性格特质创建高低对比图
        
        参数:
            trait: str, 性格特质名称
            save_path: str, 保存路径（可选）
        """
        if trait not in self.results:
            print(f"No data available for {trait}")
            return
        
        # 准备数据
        high_stats = self.results[trait].get('high', {}).get('summary_stats', {})
        low_stats = self.results[trait].get('low', {}).get('summary_stats', {})
        
        if not high_stats or not low_stats:
            print(f"Insufficient data for {trait}")
            return
        
        # 提取均值数据
        metrics = []
        high_values = []
        low_values = []
        
        for key in high_stats.keys():
            if key.endswith('_mean'):
                metric = key[:-5]
                if metric in self.metric_labels:
                    metrics.append(self.metric_labels[metric])
                    high_values.append(high_stats[key])
                    low_values.append(low_stats[key])
        
        if not metrics:
            print(f"No valid metrics found for {trait}")
            return
        
        # 创建图表
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 10))
        
        # 左图：并排柱状图
        x = np.arange(len(metrics))
        width = 0.35
        
        bars1 = ax1.bar(x - width/2, high_values, width, label='High', alpha=0.8, color='#2E86AB')
        bars2 = ax1.bar(x + width/2, low_values, width, label='Low', alpha=0.8, color='#A23B72')
        
        ax1.set_xlabel('Behavioral Metrics', fontsize=12)
        ax1.set_ylabel('Average Count/Score', fontsize=12)
        ax1.set_title(f'{trait.capitalize()} - High vs Low Comparison', fontsize=14, fontweight='bold')
        ax1.set_xticks(x)
        ax1.set_xticklabels(metrics, rotation=45, ha='right')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 右图：差异热力图
        differences = []
        for h, l in zip(high_values, low_values):
            if l != 0:
                diff = ((h - l) / l) * 100
            else:
                diff = 0
            differences.append(diff)
        
        # 创建热力图数据
        diff_matrix = np.array(differences).reshape(-1, 1)
        
        im = ax2.imshow(diff_matrix, cmap='RdBu_r', aspect='auto', vmin=-50, vmax=50)
        ax2.set_yticks(range(len(metrics)))
        ax2.set_yticklabels(metrics)
        ax2.set_xticks([0])
        ax2.set_xticklabels(['Percentage Difference\n(High vs Low)'])
        ax2.set_title(f'{trait.capitalize()} - Relative Differences (%)', fontsize=14, fontweight='bold')
        
        # 添加数值标签
        for i, diff in enumerate(differences):
            color = 'white' if abs(diff) > 25 else 'black'
            ax2.text(0, i, f'{diff:.1f}%', ha='center', va='center', color=color, fontweight='bold')
        
        # 添加颜色条
        cbar = plt.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)
        cbar.set_label('Percentage Difference (%)', rotation=270, labelpad=15)
        
        plt.tight_layout()
        
        # 保存图表
        if save_path:
            import os
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            file_path = f'{save_path}/{trait}_comparison.png'
            plt.savefig(file_path, dpi=300, bbox_inches='tight')
            print(f"保存图表: {file_path}")
        
        # 关闭图表而不显示（避免在批量生成时显示）
        plt.close()
    
    def create_comprehensive_heatmap(self, save_path=None):
        """
        创建所有性格特质的综合热力图，修复数字显示问题
        """
        # 准备数据矩阵
        all_metrics = set()
        for trait in self.traits:
            if trait in self.results:
                high_stats = self.results[trait].get('high', {}).get('summary_stats', {})
                for key in high_stats.keys():
                    if key.endswith('_mean'):
                        metric = key[:-5]
                        if metric in self.metric_labels:
                            all_metrics.add(metric)
        
        all_metrics = sorted(list(all_metrics))
        
        if not all_metrics:
            print("No metrics found for heatmap")
            return
        
        # 创建差异矩阵
        diff_matrix = []
        trait_labels = []
        
        for trait in self.traits:
            if trait not in self.results:
                continue
            
            high_stats = self.results[trait].get('high', {}).get('summary_stats', {})
            low_stats = self.results[trait].get('low', {}).get('summary_stats', {})
            
            if not high_stats or not low_stats:
                continue
            
            row = []
            for metric in all_metrics:
                high_val = high_stats.get(f'{metric}_mean', 0)
                low_val = low_stats.get(f'{metric}_mean', 0)
                
                if low_val != 0:
                    diff = ((high_val - low_val) / low_val) * 100
                else:
                    diff = 0
                
                row.append(diff)
            
            diff_matrix.append(row)
            trait_labels.append(trait.capitalize())
        
        if not diff_matrix:
            print("No data for heatmap")
            return
        
        diff_matrix = np.array(diff_matrix)
        
        # 创建热力图，调整字体大小和旋转角度
        plt.figure(figsize=(20, 12))  # 增大图表尺寸
        
        # 创建热力图时设置较小的字体
        sns.heatmap(diff_matrix, 
                   xticklabels=[self.metric_labels[m] for m in all_metrics],
                   yticklabels=trait_labels,
                   center=0,
                   cmap='RdBu_r',
                   annot=True,
                   fmt='.1f',
                   annot_kws={'size': 8, 'rotation': 45},  # 数字旋转45度，字体较小
                   cbar_kws={'label': 'Percentage Difference (%)'},
                   linewidths=0.5)
        
        plt.title('Comprehensive Personality Traits Comparison\n(High vs Low Levels)', 
                 fontsize=16, fontweight='bold', pad=20)
        plt.xlabel('Behavioral Metrics', fontsize=12)
        plt.ylabel('Personality Traits', fontsize=12)
        plt.xticks(rotation=90, ha='center', fontsize=10)  # X轴标签垂直显示
        plt.yticks(rotation=0, fontsize=12)
        
        plt.tight_layout()
        
        # 保存图表
        if save_path:
            import os
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            file_path = f'{save_path}/comprehensive_heatmap.png'
            plt.savefig(file_path, dpi=300, bbox_inches='tight')
            print(f"保存图表: {file_path}")
        
        plt.close()
    
    def create_statistical_significance_heatmap(self, save_path=None):
        """
        创建改进的统计显著性热力图 - 使用效应量和显著性结合的方式
        """
        # 收集统计检验结果
        significance_matrix = []
        effect_size_matrix = []
        trait_labels = []
        all_metrics = set()
        
        for trait in self.traits:
            if trait in self.results:
                statistical_tests = self.results[trait].get('statistical_tests', {})
                for metric in statistical_tests.keys():
                    all_metrics.add(metric)
        
        all_metrics = sorted(list(all_metrics))
        
        for trait in self.traits:
            if trait not in self.results:
                continue
            
            statistical_tests = self.results[trait].get('statistical_tests', {})
            
            sig_row = []
            effect_row = []
            for metric in all_metrics:
                if metric in statistical_tests:
                    test_result = statistical_tests[metric]
                    # 使用效应量 × 显著性指标的组合
                    p_value = test_result.get('t_p_value', 1.0)
                    cohens_d = abs(test_result.get('cohens_d', 0))
                    
                    # 显著性水平
                    if p_value < 0.001:
                        sig_level = 3  # 高度显著
                    elif p_value < 0.01:
                        sig_level = 2  # 显著
                    elif p_value < 0.05:
                        sig_level = 1  # 边际显著
                    else:
                        sig_level = 0  # 不显著
                    
                    # 结合效应量和显著性
                    combined_score = cohens_d * (sig_level + 1)  # 避免乘以0
                    sig_row.append(combined_score)
                    effect_row.append(cohens_d)
                else:
                    sig_row.append(0)
                    effect_row.append(0)
            
            significance_matrix.append(sig_row)
            effect_size_matrix.append(effect_row)
            trait_labels.append(trait.capitalize())
        
        if not significance_matrix:
            return
        
        significance_matrix = np.array(significance_matrix)
        effect_size_matrix = np.array(effect_size_matrix)
        
        # 创建双重热力图
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 8))
        
        # 左图：效应量热力图
        sns.heatmap(effect_size_matrix,
                   xticklabels=[self.metric_labels.get(m, m) for m in all_metrics],
                   yticklabels=trait_labels,
                   cmap='Blues',
                   annot=True,
                   fmt='.2f',
                   annot_kws={'size': 8},
                   cbar_kws={'label': "Cohen's d (Effect Size)"},
                   linewidths=0.5,
                   ax=ax1)
        
        ax1.set_title('Effect Sizes (Cohen\'s d)\nColors: Blue intensity = Effect magnitude', 
                     fontsize=14, fontweight='bold')
        ax1.set_xlabel('Behavioral Metrics', fontsize=12)
        ax1.set_ylabel('Personality Traits', fontsize=12)
        
        # 右图：组合显著性热力图
        sns.heatmap(significance_matrix,
                   xticklabels=[self.metric_labels.get(m, m) for m in all_metrics],
                   yticklabels=trait_labels,
                   cmap='Reds',
                   annot=True,
                   fmt='.2f',
                   annot_kws={'size': 8},
                   cbar_kws={'label': 'Effect Size × Significance Level'},
                   linewidths=0.5,
                   ax=ax2)
        
        ax2.set_title('Combined Effect Size & Statistical Significance\nColors: Red intensity = Practical importance', 
                     fontsize=14, fontweight='bold')
        ax2.set_xlabel('Behavioral Metrics', fontsize=12)
        ax2.set_ylabel('Personality Traits', fontsize=12)
        
        plt.tight_layout()
        
        if save_path:
            import os
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            file_path = f'{save_path}/improved_significance_heatmap.png'
            plt.savefig(file_path, dpi=300, bbox_inches='tight')
            print(f"保存图表: {file_path}")
        
        plt.close()
    
    def create_metric_cross_comparison(self, save_path=None):
        """
        创建测评维度的横向对比图 - 显示每个指标在五个人格特质上的高低对比
        """
        import os
        from tqdm import tqdm
        
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        
        # 获取所有指标
        all_metrics = set()
        for trait in self.traits:
            if trait in self.results:
                high_stats = self.results[trait].get('high', {}).get('summary_stats', {})
                for key in high_stats.keys():
                    if key.endswith('_mean'):
                        metric = key[:-5]
                        if metric in self.metric_labels:
                            all_metrics.add(metric)
        
        all_metrics = sorted(list(all_metrics))
        
        print(f"创建 {len(all_metrics)} 个指标的横向对比图...")
        
        with tqdm(total=len(all_metrics), desc="生成横向对比图", unit="指标") as pbar:
            for metric in all_metrics:
                pbar.set_description(f"绘制 {metric}")
                
                # 准备数据
                trait_labels = []
                high_values = []
                low_values = []
                differences = []
                effect_sizes = []
                significance_markers = []
                
                for trait in self.traits:
                    if trait in self.results:
                        high_stats = self.results[trait].get('high', {}).get('summary_stats', {})
                        low_stats = self.results[trait].get('low', {}).get('summary_stats', {})
                        statistical_tests = self.results[trait].get('statistical_tests', {})
                        
                        metric_key = f'{metric}_mean'
                        if metric_key in high_stats and metric_key in low_stats:
                            high_val = high_stats[metric_key]
                            low_val = low_stats[metric_key]
                            
                            trait_labels.append(trait.capitalize())
                            high_values.append(high_val)
                            low_values.append(low_val)
                            
                            # 计算相对差异
                            if low_val != 0:
                                diff = ((high_val - low_val) / low_val) * 100
                            else:
                                diff = 0
                            differences.append(diff)
                            
                            # 获取效应量和显著性
                            if metric in statistical_tests:
                                test_result = statistical_tests[metric]
                                effect_sizes.append(abs(test_result.get('cohens_d', 0)))
                                is_significant = test_result.get('significant_t', False)
                                significance_markers.append('***' if is_significant else '')
                            else:
                                effect_sizes.append(0)
                                significance_markers.append('')
                
                if not trait_labels:
                    pbar.update(1)
                    continue
                
                # 创建综合对比图
                fig = plt.figure(figsize=(20, 12))
                gs = fig.add_gridspec(2, 3, height_ratios=[1, 1], width_ratios=[2, 2, 1])
                
                # 子图1：柱状图对比
                ax1 = fig.add_subplot(gs[0, :2])
                x = np.arange(len(trait_labels))
                width = 0.35
                
                bars1 = ax1.bar(x - width/2, high_values, width, label='High Level', 
                               alpha=0.8, color='#2E86AB')
                bars2 = ax1.bar(x + width/2, low_values, width, label='Low Level', 
                               alpha=0.8, color='#A23B72')
                
                # 添加显著性标记
                for i, (sig_marker, high_val, low_val) in enumerate(zip(significance_markers, high_values, low_values)):
                    if sig_marker:
                        max_val = max(high_val, low_val)
                        ax1.text(i, max_val + max_val * 0.02, sig_marker, ha='center', va='bottom', 
                                fontsize=14, fontweight='bold', color='red')
                
                metric_title = self.metric_labels.get(metric, metric.replace('_', ' ').title())
                ax1.set_title(f'{metric_title} - Cross-Personality Comparison', 
                             fontsize=16, fontweight='bold')
                ax1.set_xlabel('Personality Traits', fontsize=12)
                ax1.set_ylabel('Average Score/Count', fontsize=12)
                ax1.set_xticks(x)
                ax1.set_xticklabels(trait_labels, fontsize=11)
                ax1.legend()
                ax1.grid(True, alpha=0.3)
                
                # 子图2：相对差异热力图
                ax2 = fig.add_subplot(gs[1, 0])
                diff_matrix = np.array(differences).reshape(1, -1)
                
                im = ax2.imshow(diff_matrix, cmap='RdBu_r', aspect='auto', vmin=-100, vmax=100)
                ax2.set_xticks(range(len(trait_labels)))
                ax2.set_xticklabels(trait_labels, fontsize=10, rotation=45)
                ax2.set_yticks([0])
                ax2.set_yticklabels(['Relative Diff %'])
                ax2.set_title('Percentage Differences\n(High vs Low)', fontsize=12, fontweight='bold')
                
                # 添加数值标签
                for i, diff in enumerate(differences):
                    color = 'white' if abs(diff) > 50 else 'black'
                    ax2.text(i, 0, f'{diff:.1f}%', ha='center', va='center', 
                            color=color, fontweight='bold', fontsize=10)
                
                # 子图3：效应量柱状图
                ax3 = fig.add_subplot(gs[1, 1])
                colors = ['red' if es > 0.8 else 'orange' if es > 0.5 else 'yellow' if es > 0.2 else 'lightgray' 
                         for es in effect_sizes]
                bars = ax3.bar(range(len(trait_labels)), effect_sizes, color=colors, alpha=0.7)
                
                ax3.set_title('Effect Sizes (Cohen\'s d)', fontsize=12, fontweight='bold')
                ax3.set_xlabel('Personality Traits', fontsize=10)
                ax3.set_ylabel('Effect Size', fontsize=10)
                ax3.set_xticks(range(len(trait_labels)))
                ax3.set_xticklabels(trait_labels, fontsize=9, rotation=45)
                
                # 添加效应量参考线
                ax3.axhline(y=0.2, color='green', linestyle='--', alpha=0.7, label='Small')
                ax3.axhline(y=0.5, color='orange', linestyle='--', alpha=0.7, label='Medium')
                ax3.axhline(y=0.8, color='red', linestyle='--', alpha=0.7, label='Large')
                ax3.legend(fontsize=8)
                ax3.grid(True, alpha=0.3)
                
                # 子图4：排名展示
                ax4 = fig.add_subplot(gs[:, 2])
                ax4.axis('off')
                
                # 创建排名表格
                ranking_data = list(zip(trait_labels, high_values, low_values, differences, effect_sizes))
                ranking_data.sort(key=lambda x: abs(x[3]), reverse=True)  # 按相对差异排序
                
                table_text = "Ranking by Relative Difference:\n\n"
                for i, (trait, high, low, diff, effect) in enumerate(ranking_data, 1):
                    direction = "↑" if diff > 0 else "↓"
                    table_text += f"{i}. {trait}\n"
                    table_text += f"   {direction} {abs(diff):.1f}%\n"
                    table_text += f"   Effect: {effect:.2f}\n\n"
                
                ax4.text(0.05, 0.95, table_text, transform=ax4.transAxes, fontsize=10, 
                        verticalalignment='top', fontfamily='monospace',
                        bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.5))
                
                plt.tight_layout()
                
                # 保存图表
                safe_metric_name = metric.replace('/', '_').replace(' ', '_')
                file_path = f'{save_path}/{safe_metric_name}_cross_comparison.png'
                plt.savefig(file_path, dpi=300, bbox_inches='tight')
                plt.close()
                
                pbar.update(1)
        
        print(f"横向对比图已保存到 {save_path}")
    
    def create_comprehensive_ranking_chart(self, save_path=None):
        """
        创建综合排名图表 - 显示各个指标在不同人格特质上的总体表现
        """
        # 收集所有数据
        ranking_data = []
        
        for trait in self.traits:
            if trait not in self.results:
                continue
                
            high_stats = self.results[trait].get('high', {}).get('summary_stats', {})
            low_stats = self.results[trait].get('low', {}).get('summary_stats', {})
            statistical_tests = self.results[trait].get('statistical_tests', {})
            
            for key in high_stats.keys():
                if key.endswith('_mean'):
                    metric = key[:-5]
                    if metric in self.metric_labels:
                        high_val = high_stats[key]
                        low_val = low_stats[key]
                        
                        # 计算相对差异
                        if low_val != 0:
                            rel_diff = ((high_val - low_val) / low_val) * 100
                        else:
                            rel_diff = 0
                        
                        # 获取效应量
                        effect_size = 0
                        if metric in statistical_tests:
                            effect_size = abs(statistical_tests[metric].get('cohens_d', 0))
                        
                        ranking_data.append({
                            'trait': trait,
                            'metric': metric,
                            'relative_diff': rel_diff,
                            'effect_size': effect_size,
                            'abs_rel_diff': abs(rel_diff)
                        })
        
        # 转换为DataFrame
        df = pd.DataFrame(ranking_data)
        
        if df.empty:
            return
        
        # 创建综合排名图
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 16))
        
        # 1. 各特质的平均效应量
        trait_avg_effect = df.groupby('trait')['effect_size'].mean().sort_values(ascending=False)
        bars1 = ax1.bar(range(len(trait_avg_effect)), trait_avg_effect.values, 
                       color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7'])
        ax1.set_title('Average Effect Size by Personality Trait', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Personality Traits', fontsize=12)
        ax1.set_ylabel('Average Effect Size', fontsize=12)
        ax1.set_xticks(range(len(trait_avg_effect)))
        ax1.set_xticklabels([t.capitalize() for t in trait_avg_effect.index], fontsize=11)
        ax1.grid(True, alpha=0.3)
        
        # 添加数值标签
        for i, v in enumerate(trait_avg_effect.values):
            ax1.text(i, v + 0.01, f'{v:.2f}', ha='center', va='bottom', fontweight='bold')
        
        # 2. 各指标的平均效应量
        metric_avg_effect = df.groupby('metric')['effect_size'].mean().sort_values(ascending=False)[:15]
        ax2.barh(range(len(metric_avg_effect)), metric_avg_effect.values, color='skyblue')
        ax2.set_title('Top 15 Metrics by Average Effect Size', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Average Effect Size', fontsize=12)
        ax2.set_yticks(range(len(metric_avg_effect)))
        ax2.set_yticklabels([self.metric_labels.get(m, m) for m in metric_avg_effect.index], fontsize=10)
        ax2.grid(True, alpha=0.3)
        
        # 3. 各特质的差异程度分布
        trait_avg_diff = df.groupby('trait')['abs_rel_diff'].mean().sort_values(ascending=False)
        bars3 = ax3.bar(range(len(trait_avg_diff)), trait_avg_diff.values,
                       color=['#E74C3C', '#3498DB', '#2ECC71', '#F39C12', '#9B59B6'])
        ax3.set_title('Average Absolute Difference by Personality Trait', fontsize=14, fontweight='bold')
        ax3.set_xlabel('Personality Traits', fontsize=12)
        ax3.set_ylabel('Average Absolute Difference (%)', fontsize=12)
        ax3.set_xticks(range(len(trait_avg_diff)))
        ax3.set_xticklabels([t.capitalize() for t in trait_avg_diff.index], fontsize=11)
        ax3.grid(True, alpha=0.3)
        
        # 添加数值标签
        for i, v in enumerate(trait_avg_diff.values):
            ax3.text(i, v + 1, f'{v:.1f}%', ha='center', va='bottom', fontweight='bold')
        
        # 4. 效应量vs差异程度散点图
        scatter = ax4.scatter(df['abs_rel_diff'], df['effect_size'], 
                            c=[hash(trait) for trait in df['trait']], 
                            alpha=0.6, s=50, cmap='tab10')
        ax4.set_title('Effect Size vs Absolute Difference', fontsize=14, fontweight='bold')
        ax4.set_xlabel('Absolute Relative Difference (%)', fontsize=12)
        ax4.set_ylabel('Effect Size (Cohen\'s d)', fontsize=12)
        ax4.grid(True, alpha=0.3)
        
        # 添加参考线
        ax4.axhline(y=0.5, color='orange', linestyle='--', alpha=0.7, label='Medium Effect')
        ax4.axhline(y=0.8, color='red', linestyle='--', alpha=0.7, label='Large Effect')
        ax4.axvline(x=50, color='green', linestyle='--', alpha=0.7, label='50% Difference')
        ax4.legend()
        
        plt.tight_layout()
        
        if save_path:
            import os
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            file_path = f'{save_path}/comprehensive_ranking_analysis.png'
            plt.savefig(file_path, dpi=300, bbox_inches='tight')
            print(f"保存图表: {file_path}")
        
        plt.close()

    def create_category_comparison_plot(self, trait, save_path=None):
        """
        创建按类别分组的对比图
        
        参数:
            trait: str, 性格特质名称
            save_path: str, 保存路径（可选）
        """
        if trait not in self.results:
            return
        
        high_stats = self.results[trait].get('high', {}).get('summary_stats', {})
        low_stats = self.results[trait].get('low', {}).get('summary_stats', {})
        
        if not high_stats or not low_stats:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        axes = axes.flatten()
        
        for i, (category, metrics) in enumerate(self.metric_categories.items()):
            if i >= len(axes):
                break
                
            ax = axes[i]
            
            # 准备该类别的数据
            cat_metrics = []
            high_values = []
            low_values = []
            
            for metric in metrics:
                if f'{metric}_mean' in high_stats and f'{metric}_mean' in low_stats:
                    cat_metrics.append(self.metric_labels[metric])
                    high_values.append(high_stats[f'{metric}_mean'])
                    low_values.append(low_stats[f'{metric}_mean'])
            
            if not cat_metrics:
                ax.set_visible(False)
                continue
            
            x = np.arange(len(cat_metrics))
            width = 0.35
            
            ax.bar(x - width/2, high_values, width, label='High', alpha=0.8, color='#2E86AB')
            ax.bar(x + width/2, low_values, width, label='Low', alpha=0.8, color='#A23B72')
            
            ax.set_title(f'{category}', fontsize=12, fontweight='bold')
            ax.set_xticks(x)
            ax.set_xticklabels(cat_metrics, rotation=45, ha='right', fontsize=10)
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # 隐藏多余的子图
        for i in range(len(self.metric_categories), len(axes)):
            axes[i].set_visible(False)
        
        plt.suptitle(f'{trait.capitalize()} - Category-wise Comparison', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        # 保存图表
        if save_path:
            import os
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            file_path = f'{save_path}/{trait}_category_comparison.png'
            plt.savefig(file_path, dpi=300, bbox_inches='tight')
            print(f"保存图表: {file_path}")
        
        plt.close()

    def create_radar_chart(self, trait, save_path=None):
        """
        为性格特质创建雷达图（更新关键指标）
        
        参数:
            trait: str, 性格特质名称
            save_path: str, 保存路径（可选）
        """
        if trait not in self.results:
            return
        
        high_stats = self.results[trait].get('high', {}).get('summary_stats', {})
        low_stats = self.results[trait].get('low', {}).get('summary_stats', {})
        
        if not high_stats or not low_stats:
            return
        
        # 更新关键指标选择（使用新的指标）
        key_metrics = ['vader_compound', 'lexical_density', 'vocabulary_size',
                      'social_words_count', 'achievement_words_count', 
                      'uncertainty_words_count', 'politeness_words_count']
        
        # 准备数据
        high_values = []
        low_values = []
        labels = []
        
        for metric in key_metrics:
            if f'{metric}_mean' in high_stats and f'{metric}_mean' in low_stats:
                high_val = high_stats[f'{metric}_mean']
                low_val = low_stats[f'{metric}_mean']
                
                # 标准化到0-1范围
                max_val = max(high_val, low_val)
                if max_val > 0:
                    high_values.append(high_val / max_val)
                    low_values.append(low_val / max_val)
                    labels.append(self.metric_labels[metric])
        
        if not labels:
            return
        
        # 创建雷达图
        angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False)
        angles = np.concatenate((angles, [angles[0]]))
        
        high_values = np.concatenate((high_values, [high_values[0]]))
        low_values = np.concatenate((low_values, [low_values[0]]))
        
        fig, ax = plt.subplots(figsize=(12, 12), subplot_kw=dict(projection='polar'))
        
        ax.plot(angles, high_values, 'o-', linewidth=2, label='High', color='#2E86AB')
        ax.fill(angles, high_values, alpha=0.25, color='#2E86AB')
        
        ax.plot(angles, low_values, 'o-', linewidth=2, label='Low', color='#A23B72')
        ax.fill(angles, low_values, alpha=0.25, color='#A23B72')
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(labels, fontsize=11)
        ax.set_ylim(0, 1)
        ax.set_title(f'{trait.capitalize()} - Behavioral Profile Comparison', 
                    fontsize=14, fontweight='bold', pad=20)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
        ax.grid(True)
        
        # 保存图表
        if save_path:
            import os
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            file_path = f'{save_path}/{trait}_radar.png'
            plt.savefig(file_path, dpi=300, bbox_inches='tight')
            print(f"保存图表: {file_path}")
        
        plt.close()

    def create_all_visualizations(self, save_path='./plots'):
        """
        创建所有可视化图表（包含新的横向对比功能）
        
        参数:
            save_path: str, 保存路径
        """
        import os
        from tqdm import tqdm
        
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        
        print("创建综合可视化图表...")
        
        # 计算总的可视化任务数
        valid_traits = [trait for trait in self.traits if trait in self.results]
        # 增加新的图表类型
        total_tasks = 4 + len(valid_traits) * 3  # 4个综合图 + 每个特质3个图
        
        with tqdm(total=total_tasks, desc="可视化总进度", unit="图表") as pbar:
            # 1. 综合热力图
            pbar.set_description("生成综合热力图")
            self.create_comprehensive_heatmap(save_path)
            pbar.update(1)
            
            # 2. 改进的统计显著性热力图
            pbar.set_description("生成改进统计显著性热力图")
            self.create_statistical_significance_heatmap(save_path)
            pbar.update(1)
            
            # 3. 横向对比图
            pbar.set_description("生成横向对比图")
            cross_comparison_path = f'{save_path}/cross_comparisons'
            self.create_metric_cross_comparison(cross_comparison_path)
            pbar.update(1)
            
            # 4. 综合排名分析
            pbar.set_description("生成综合排名分析")
            self.create_comprehensive_ranking_chart(save_path)
            pbar.update(1)
            
            # 5. 为每个性格特质创建对比图
            for trait in valid_traits:
                # 特质对比图
                pbar.set_description(f"生成 {trait} 对比图")
                self.create_trait_comparison_plot(trait, save_path)
                pbar.update(1)
                
                # 分类对比图
                pbar.set_description(f"生成 {trait} 分类对比")
                self.create_category_comparison_plot(trait, save_path)
                pbar.update(1)
                
                # 雷达图
                pbar.set_description(f"生成 {trait} 雷达图")
                self.create_radar_chart(trait, save_path)
                pbar.update(1)
        
        print(f"所有可视化图表已保存到 {save_path}")
        print(f"共生成 {total_tasks} 个图表文件")
        print(f"横向对比图保存在 {save_path}/cross_comparisons")