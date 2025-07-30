import pandas as pd
import numpy as np
from val_ocean_b import filter_by_personality, PersonalityBehaviorEvaluator
from visualization import PersonalityVisualization
from tqdm import tqdm  # 添加进度条
import time
import pickle
from scipy import stats

class PersonalityComparisonAnalyzer:
    """
    性格特质对比分析器
    """
    
    def __init__(self):
        """初始化分析器"""
        self.traits = ['openness', 'agreeableness', 'conscientiousness', 'extraversion', 'neuroticism']
        self.levels = ['high', 'low']
        self.results = {}
        
    def analyze_trait_behavior(self, trait, level, text_column='train_output'):
        """
        分析特定性格特质和程度的行为特征
        
        参数:
            trait: str, 性格特质
            level: str, 程度 ('high' 或 'low')
            text_column: str, 文本列名
            
        返回:
            dict: 包含评估结果和统计信息的字典
        """
        # 筛选数据
        print(f"筛选 {trait} {level} 数据...")
        filtered_data = filter_by_personality(trait=trait, level=level)
        
        if filtered_data.empty:
            print(f"警告: 没有找到 {trait} {level} 的数据")
            return None
        
        print(f"找到 {len(filtered_data)} 条 {trait} {level} 记录")
        
        # 创建评估器并进行评估
        print(f"开始评估 {trait} {level} 行为特征...")
        evaluator = PersonalityBehaviorEvaluator(filtered_data)
        
        start_time = time.time()
        evaluation_results = evaluator.evaluate_all(text_column)
        processing_time = time.time() - start_time
        
        print(f"{trait} {level} 评估完成，耗时 {processing_time:.2f} 秒")
        
        # 计算汇总统计
        summary_stats = self.get_summary_statistics_from_results(evaluation_results)
        
        return {
            'data_count': len(filtered_data),
            'evaluation_results': evaluation_results,
            'summary_stats': summary_stats,
            'processing_time': processing_time
        }
    
    def get_summary_statistics_from_results(self, evaluation_results):
        """
        从评估结果计算汇总统计
        
        参数:
            evaluation_results: DataFrame, 评估结果
            
        返回:
            dict: 汇总统计
        """
        if evaluation_results.empty:
            return {}
        
        numeric_columns = evaluation_results.select_dtypes(include=[np.number]).columns
        summary = {}
        
        for col in numeric_columns:
            if col != 'index':
                values = evaluation_results[col].dropna()
                
                # 基础统计
                summary[f'{col}_mean'] = values.mean()
                summary[f'{col}_std'] = values.std()
                summary[f'{col}_median'] = values.median()
                summary[f'{col}_count'] = len(values)
        
        return summary
    
    def perform_statistical_tests(self, high_results, low_results):
        """
        执行统计检验，只对有足够差异和数据的指标进行检验
        
        参数:
            high_results: dict, 高程度评估结果
            low_results: dict, 低程度评估结果
            
        返回:
            dict: 统计检验结果
        """
        high_data = high_results['evaluation_results']
        low_data = low_results['evaluation_results']
        
        test_results = {}
        
        # 获取数值列
        numeric_columns = high_data.select_dtypes(include=[np.number]).columns
        
        for col in numeric_columns:
            if col != 'index':
                high_values = high_data[col].dropna()
                low_values = low_data[col].dropna()
                
                # 检查数据质量：需要足够的样本且存在差异
                if len(high_values) > 10 and len(low_values) > 10:
                    # 检查是否有足够的变异性
                    high_var = high_values.var()
                    low_var = low_values.var()
                    
                    # 至少一组数据有变异性，且均值差异超过某个阈值
                    mean_diff = abs(high_values.mean() - low_values.mean())
                    relative_diff = mean_diff / (low_values.mean() + 1e-10)  # 防止除零
                    
                    if (high_var > 1e-10 or low_var > 1e-10) and relative_diff > 0.01:  # 1%的相对差异阈值
                        try:
                            # t检验
                            t_stat, t_p_value = stats.ttest_ind(high_values, low_values)
                            
                            # Mann-Whitney U检验（非参数检验）
                            u_stat, u_p_value = stats.mannwhitneyu(high_values, low_values, alternative='two-sided')
                            
                            # 效应量（Cohen's d）
                            pooled_std = np.sqrt(((len(high_values) - 1) * high_values.std()**2 + 
                                                (len(low_values) - 1) * low_values.std()**2) / 
                                               (len(high_values) + len(low_values) - 2))
                            cohens_d = (high_values.mean() - low_values.mean()) / pooled_std if pooled_std > 0 else 0
                            
                            # 处理极小的p值 - 避免显示为0.0
                            formatted_t_p = float(t_p_value) if t_p_value > 1e-300 else 1e-300
                            formatted_u_p = float(u_p_value) if u_p_value > 1e-300 else 1e-300
                            
                            # 只保存有意义的结果（避免无穷大和NaN）
                            if not (np.isnan(t_p_value) or np.isnan(u_p_value) or np.isinf(abs(cohens_d))):
                                test_results[col] = {
                                    't_statistic': float(t_stat),
                                    't_p_value': formatted_t_p,
                                    't_p_value_raw': float(t_p_value),  # 保存原始p值
                                    'u_statistic': float(u_stat),
                                    'u_p_value': formatted_u_p,
                                    'u_p_value_raw': float(u_p_value),  # 保存原始p值
                                    'cohens_d': float(cohens_d),
                                    'high_n': len(high_values),
                                    'low_n': len(low_values),
                                    'high_mean': float(high_values.mean()),
                                    'low_mean': float(low_values.mean()),
                                    'significant_t': t_p_value < 0.05,
                                    'significant_u': u_p_value < 0.05,
                                    'effect_size_interpretation': self.interpret_cohens_d(cohens_d),
                                    # 添加统计显著性程度标记
                                    'significance_level_t': self.get_significance_level(t_p_value),
                                    'significance_level_u': self.get_significance_level(u_p_value)
                                }
                        except Exception as e:
                            # 如果统计检验失败，跳过这个指标
                            print(f"统计检验失败 for {col}: {e}")
                            continue
        
        return test_results
    
    def get_significance_level(self, p_value):
        """
        获取统计显著性程度
        
        参数:
            p_value: float, p值
            
        返回:
            str: 显著性程度描述
        """
        if p_value < 1e-100:
            return "extremely_significant"
        elif p_value < 1e-50:
            return "highly_significant"
        elif p_value < 0.001:
            return "very_significant"
        elif p_value < 0.01:
            return "significant"
        elif p_value < 0.05:
            return "marginally_significant"
        else:
            return "not_significant"

    def interpret_cohens_d(self, d):
        """解释Cohen's d效应量"""
        abs_d = abs(d)
        if abs_d < 0.2:
            return "negligible"
        elif abs_d < 0.5:
            return "small"
        elif abs_d < 0.8:
            return "medium"
        else:
            return "large"
    
    def compare_trait_levels(self, trait, text_column='train_output'):
        """
        对比特定性格特质的高低程度，包含统计检验
        
        参数:
            trait: str, 性格特质
            text_column: str, 文本列名
            
        返回:
            dict: 包含高低对比结果的字典
        """
        print(f"\n=== 分析性格特质: {trait.upper()} ===")
        
        # 分析高程度和低程度
        high_results = self.analyze_trait_behavior(trait, 'high', text_column)
        low_results = self.analyze_trait_behavior(trait, 'low', text_column)
        
        if not high_results or not low_results:
            print(f"跳过 {trait}：数据不足")
            return None
        
        # 计算差异
        differences = self.calculate_differences(high_results['summary_stats'], 
                                               low_results['summary_stats'])
        
        # 执行统计检验
        statistical_tests = self.perform_statistical_tests(high_results, low_results)
        
        return {
            'trait': trait,
            'high': high_results,
            'low': low_results,
            'differences': differences,
            'statistical_tests': statistical_tests
        }
    
    def calculate_differences(self, high_stats, low_stats):
        """
        计算高低程度之间的差异
        
        参数:
            high_stats: dict, 高程度统计结果
            low_stats: dict, 低程度统计结果
            
        返回:
            dict: 差异分析结果
        """
        differences = {}
        
        for key in high_stats:
            if key.endswith('_mean') and key in low_stats:
                high_val = high_stats[key]
                low_val = low_stats[key]
                
                # 绝对差异
                abs_diff = high_val - low_val
                
                # 相对差异（百分比）
                if low_val != 0:
                    rel_diff = ((high_val - low_val) / low_val) * 100
                else:
                    rel_diff = 0
                
                differences[key] = {
                    'high_value': high_val,
                    'low_value': low_val,
                    'absolute_difference': abs_diff,
                    'relative_difference_percent': rel_diff
                }
        
        return differences
    
    def run_complete_analysis(self, text_column='train_output'):
        """
        运行完整的分析流程
        
        参数:
            text_column: str, 文本列名
            
        返回:
            dict: 完整的分析结果
        """
        print("开始进行性格特质行为对比分析...")
        print("=" * 60)
        
        all_results = {}
        total_traits = len(self.traits)
        
        # 使用进度条跟踪整体进度
        with tqdm(total=total_traits, desc="分析特质进度", unit="特质") as trait_pbar:
            for i, trait in enumerate(self.traits):
                trait_pbar.set_description(f"分析 {trait.capitalize()}")
                
                start_time = time.time()
                result = self.compare_trait_levels(trait, text_column)
                trait_time = time.time() - start_time
                
                if result:
                    all_results[trait] = result
                    self.results[trait] = result
                    
                    # 更新进度条信息
                    trait_pbar.set_postfix({
                        '已完成': f'{i+1}/{total_traits}',
                        '特质耗时': f'{trait_time:.1f}s',
                        '预计剩余': f'{(total_traits - i - 1) * trait_time:.1f}s'
                    })
                else:
                    trait_pbar.set_postfix({
                        '状态': '跳过（数据不足)',
                        '已完成': f'{i+1}/{total_traits}'
                    })
                
                trait_pbar.update(1)
        
        print("\n" + "=" * 60)
        print("分析完成！")
        print(f"成功分析了 {len(all_results)} 个性格特质")
        
        # 显示总体统计
        total_records = sum(
            result['high']['data_count'] + result['low']['data_count'] 
            for result in all_results.values()
        )
        total_time = sum(
            result['high'].get('processing_time', 0) + result['low'].get('processing_time', 0)
            for result in all_results.values()
        )
        
        print(f"总计处理记录: {total_records}")
        print(f"总计耗时: {total_time:.2f} 秒")
        print(f"平均处理速度: {total_records/total_time:.1f} 条记录/秒")
        
        return all_results
    
    def generate_summary_report(self):
        """
        生成分析结果的汇总报告
        
        返回:
            pandas.DataFrame: 汇总报告
        """
        if not self.results:
            print("没有可用的分析结果")
            return pd.DataFrame()
        
        report_data = []
        
        for trait, result in self.results.items():
            high_count = result['high']['data_count']
            low_count = result['low']['data_count']
            
            # 选择几个关键指标进行报告（更新为新指标）
            key_metrics = ['ttr_mean', 'vocabulary_size_mean', 'social_words_count_mean', 
                          'achievement_words_count_mean', 'politeness_words_count_mean', 
                          'vader_positive_mean']
            
            row = {
                'trait': trait,
                'high_count': high_count,
                'low_count': low_count,
                'total_count': high_count + low_count
            }
            
            # 添加关键指标的差异
            for metric in key_metrics:
                if metric in result['differences']:
                    diff_info = result['differences'][metric]
                    row[f'{metric}_high'] = round(diff_info['high_value'], 3)
                    row[f'{metric}_low'] = round(diff_info['low_value'], 3)
                    row[f'{metric}_diff_pct'] = round(diff_info['relative_difference_percent'], 1)
            
            report_data.append(row)
        
        return pd.DataFrame(report_data)
    
    def print_detailed_findings(self):
        """
        打印详细的发现和洞察
        """
        if not self.results:
            print("没有可用的分析结果")
            return
        
        print("\n" + "=" * 60)
        print("详细分析发现")
        print("=" * 60)
        
        for trait, result in self.results.items():
            print(f"\n{trait.upper()} 特质分析:")
            print("-" * 40)
            
            high_count = result['high']['data_count']
            low_count = result['low']['data_count']
            print(f"数据量: 高程度 {high_count} 条, 低程度 {low_count} 条")
            
            # 找出最显著的差异
            significant_diffs = []
            for metric, diff_info in result['differences'].items():
                if abs(diff_info['relative_difference_percent']) > 10:  # 超过10%的差异
                    significant_diffs.append((metric, diff_info))
            
            if significant_diffs:
                print(f"显著差异指标 (>10%):")
                for metric, diff_info in sorted(significant_diffs, 
                                              key=lambda x: abs(x[1]['relative_difference_percent']), 
                                              reverse=True)[:5]:
                    metric_name = metric.replace('_mean', '').replace('_', ' ').title()
                    pct = diff_info['relative_difference_percent']
                    direction = "高于" if pct > 0 else "低于"
                    print(f"  • {metric_name}: 高程度比低程度{direction} {abs(pct):.1f}%")
            else:
                print("  未发现显著差异 (>10%)")
    
    def save_complete_results(self, save_path='./complete_analysis_results_summary.csv'):
        """
        保存完整的分析结果为CSV格式
        
        参数:
            save_path: str, 保存路径
        """
        if self.results:
            summary_df = self.generate_detailed_summary()
            summary_df.to_csv(save_path, index=False)
            print(f"完整结果已保存到: {save_path}")
        else:
            print("没有可用的分析结果")
    
    def generate_detailed_summary(self):
        """
        生成详细汇总报告，包含统计检验结果
        """
        if not self.results:
            return pd.DataFrame()
        
        summary_data = []
        
        for trait, result in self.results.items():
            high_stats = result['high']['summary_stats']
            low_stats = result['low']['summary_stats']
            statistical_tests = result.get('statistical_tests', {})
            
            # 获取所有指标
            metrics = set()
            for key in high_stats.keys():
                if key.endswith('_mean'):
                    metrics.add(key[:-5])
            
            for metric in metrics:
                metric_mean = f'{metric}_mean'
                metric_count = f'{metric}_count'
                
                if metric_mean in high_stats and metric_mean in low_stats:
                    high_mean = high_stats.get(metric_mean, 0)
                    low_mean = low_stats.get(metric_mean, 0)
                    
                    # 存在一定程度的差异
                    abs_diff = abs(high_mean - low_mean)
                    relative_diff_percent = abs_diff / (low_mean + 1e-10) * 100
                    
                    if abs_diff > 1e-6 or relative_diff_percent > 0.1:
                        row = {
                            'trait': trait,
                            'metric': metric,
                            'high_mean': high_mean,
                            'low_mean': low_mean,
                            'high_count': high_stats.get(metric_count, 0),
                            'low_count': low_stats.get(metric_count, 0),
                            'absolute_difference': abs_diff,
                            'relative_difference_percent': ((high_mean - low_mean) / (low_mean + 1e-10)) * 100
                        }
                        
                        # 添加统计检验结果
                        if metric in statistical_tests:
                            test_result = statistical_tests[metric]
                            # 使用格式化后的p值避免显示0.0
                            row.update({
                                't_p_value': test_result.get('t_p_value', 1.0),
                                't_p_value_scientific': f"{test_result.get('t_p_value_raw', 1.0):.2e}",
                                'u_p_value': test_result.get('u_p_value', 1.0),
                                'u_p_value_scientific': f"{test_result.get('u_p_value_raw', 1.0):.2e}",
                                'cohens_d': test_result.get('cohens_d', 0),
                                'effect_size': test_result.get('effect_size_interpretation', 'negligible'),
                                'significant_t': test_result.get('significant_t', False),
                                'significant_u': test_result.get('significant_u', False),
                                'significance_level_t': test_result.get('significance_level_t', 'not_significant'),
                                'significance_level_u': test_result.get('significance_level_u', 'not_significant')
                            })
                        else:
                            row.update({
                                't_p_value': None,
                                't_p_value_scientific': 'N/A',
                                'u_p_value': None,
                                'u_p_value_scientific': 'N/A',
                                'cohens_d': None,
                                'effect_size': 'not_tested',
                                'significant_t': False,
                                'significant_u': False,
                                'significance_level_t': 'not_tested',
                                'significance_level_u': 'not_tested'
                            })
                        
                        summary_data.append(row)
        
        df = pd.DataFrame(summary_data)
        
        # 按相对差异的绝对值排序，显示最有意义的差异
        if not df.empty:
            df['abs_relative_diff'] = abs(df['relative_difference_percent'])
            df = df.sort_values('abs_relative_diff', ascending=False)
            df = df.drop('abs_relative_diff', axis=1)
        
        return df
    
def comprehensive_analysis_example():
    """
    完整分析示例
    """
    # 创建分析器
    analyzer = PersonalityComparisonAnalyzer()
    
    # 运行完整分析
    comparison_results = analyzer.run_complete_analysis('train_output')
    
    # 保存完整结果
    analyzer.save_complete_results()
    
    # 生成汇总报告
    results_df = analyzer.generate_summary_report()
    
    # 打印详细发现
    analyzer.print_detailed_findings()
    
    # 显示汇总报告
    print("\n" + "=" * 60)
    print("汇总报告")
    print("=" * 60)
    print(results_df.to_string(index=False))
    
    return analyzer, comparison_results, results_df

def run_analysis_with_visualization():
    """
    运行分析并生成可视化图表
    """
    print("开始完整的性格特质分析和可视化流程...")
    start_total_time = time.time()
    
    # 1. 运行分析
    print("\n第一阶段：数据分析")
    analyzer, comparison_results, results_df = comprehensive_analysis_example()
    analysis_time = time.time() - start_total_time
    
    # 2. 创建可视化
    print(f"\n第二阶段：创建可视化图表（分析耗时: {analysis_time:.2f}s）")
    
    visualizer = PersonalityVisualization(analyzer)
    visualizer.create_all_visualizations('./analysis_plots')
    
    viz_time = time.time() - start_total_time - analysis_time
    total_time = time.time() - start_total_time
    
    print(f"\n分析完成！")
    print(f"数据分析耗时: {analysis_time:.2f} 秒")
    print(f"可视化耗时: {viz_time:.2f} 秒")
    print(f"总耗时: {total_time:.2f} 秒")
    print(f"图表已保存到 './analysis_plots' 文件夹")

    return analyzer, visualizer, results_df

if __name__ == "__main__":
    # 运行完整的分析和可视化流程
    analyzer, visualizer, results_df = run_analysis_with_visualization()
    print("\n" + "=" * 60)
    print("汇总报告")
    print("=" * 60)
    print(results_df.to_string(index=False))

