#!/usr/bin/env python3
"""
简化人格评估流程运行脚本
执行三种对比并生成可视化结果
"""

import sys
import os
import time

# 添加项目根目录到路径
project_root = "/Users/zhuruichen/PycharmProjects/PA-TOD-Dataset"
sys.path.append(project_root)

from evaluator.simplified_personality_comparator import SimplifiedPersonalityComparator
from evaluator.comparison_visualizer import ComparisonResultsVisualizer


def run_simplified_evaluation():
    """运行简化的人格评估流程"""
    print("🚀 启动简化人格评估流程")
    print("=" * 60)
    
    # 配置路径
    json_file_path = "/Users/zhuruichen/PycharmProjects/PA-TOD-Dataset/output/mAPipelineOutput/rs_42_d_100/complete_transformed_dialogues.json"
    model_path = "/Users/zhuruichen/PycharmProjects/PA-TOD-Dataset/Model/Bert_personality"
    output_dir = "/Users/zhuruichen/PycharmProjects/PA-TOD-Dataset/output/mAPipelineOutput/rs_42_d_100"
    
    print(f"📁 数据文件: {json_file_path}")
    print(f"🤖 模型路径: {model_path}")
    print(f"📊 输出目录: {output_dir}")
    
    # 检查文件是否存在
    if not os.path.exists(json_file_path):
        print(f"❌ 错误: 找不到数据文件 {json_file_path}")
        return False
    
    if not os.path.exists(model_path):
        print(f"❌ 错误: 找不到模型文件夹 {model_path}")
        return False
    
    try:
        # 第一步: 运行人格对比分析
        print("\n📈 第一步: 运行人格对比分析...")
        start_time = time.time()
        
        comparator = SimplifiedPersonalityComparator(
            json_file_path=json_file_path,
            model_path=model_path,
            output_dir=output_dir
        )
        
        success = comparator.run_comparison()
        if not success:
            print("❌ 人格对比分析失败")
            return False
        
        comparison_time = time.time() - start_time
        print(f"✅ 人格对比分析完成 (耗时: {comparison_time:.2f}秒)")
        
        # 第二步: 生成可视化结果
        print("\n📊 第二步: 生成可视化结果...")
        start_time = time.time()
        
        results_path = os.path.join(output_dir, "personality_comparison_results.csv")
        if not os.path.exists(results_path):
            print(f"❌ 错误: 找不到结果文件 {results_path}")
            return False
        
        visualizer = ComparisonResultsVisualizer(results_path)
        visualizer.create_all_visualizations()
        
        visualization_time = time.time() - start_time
        print(f"✅ 可视化完成 (耗时: {visualization_time:.2f}秒)")
        
        # 总结
        total_time = comparison_time + visualization_time
        print("\n🎉 评估流程完成!")
        print("=" * 60)
        print(f"⏱️  总耗时: {total_time:.2f}秒")
        print(f"📁 结果文件: {results_path}")
        print(f"📊 图表保存在: {output_dir}")
        print("\n生成的文件:")
        print("- personality_comparison_results.csv (详细对比结果)")
        print("- average_differences_comparison.png (平均差异对比)")
        print("- dimension_comparison_heatmap.png (维度差异热力图)")
        print("- difference_distributions.png (差异分布图)")
        print("- score_comparison_radar.png (得分对比雷达图)")
        print("- correlation_analysis.png (相关性分析)")
        
        return True
        
    except Exception as e:
        print(f"❌ 运行过程中发生错误: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """主函数"""
    print("简化人格评估与可视化工具")
    print("用于对比转换后对话与三个参考值的差异")
    print("参考值: 1) 原始对话模型评分 2) 转换目标值 3) LLM评分值")
    
    success = run_simplified_evaluation()
    
    if success:
        print("\n✨ 评估完成! 请查看生成的图表和结果文件。")
    else:
        print("\n💥 评估失败，请检查错误信息并重试。")
        sys.exit(1)


if __name__ == "__main__":
    main()
