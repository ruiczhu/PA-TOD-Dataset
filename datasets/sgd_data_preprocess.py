import json
import os
from pathlib import Path
from typing import Dict, List, Any
import glob

def process_dialogue(dialogue: Dict[str, Any]) -> Dict[str, Any]:
    """
    处理单个对话，添加personality字段到dialogue级别，
    添加transformed_utterance字段到每个turn的utterance级别
    
    Args:
        dialogue: 原始对话数据
        
    Returns:
        处理后的对话数据
    """
    # 深拷贝对话数据以避免修改原始数据
    processed_dialogue = dialogue.copy()
    
    # 在dialogue级别添加personality字段（初始为空）
    processed_dialogue['personality'] = ""
    
    # 处理每个turn
    processed_turns = []
    for turn in dialogue.get('turns', []):
        processed_turn = turn.copy()
        
        # 在每个turn中添加transformed_utterance字段（初始为空）
        if 'utterance' in processed_turn:
            processed_turn['transformed_utterance'] = ""
            
        processed_turns.append(processed_turn)
    
    processed_dialogue['turns'] = processed_turns
    
    return processed_dialogue

def process_json_file(input_file_path: str, output_file_path: str) -> None:
    """
    处理单个JSON文件
    
    Args:
        input_file_path: 输入文件路径
        output_file_path: 输出文件路径
    """
    try:
        print(f"处理文件: {input_file_path}")
        
        # 读取原始JSON文件
        with open(input_file_path, 'r', encoding='utf-8') as f:
            dialogues = json.load(f)
        
        # 处理每个对话
        processed_dialogues = []
        for dialogue in dialogues:
            processed_dialogue = process_dialogue(dialogue)
            processed_dialogues.append(processed_dialogue)
        
        # 确保输出目录存在
        os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
        
        # 保存处理后的数据
        with open(output_file_path, 'w', encoding='utf-8') as f:
            json.dump(processed_dialogues, f, ensure_ascii=False, indent=2)
        
        print(f"完成处理: {output_file_path}")
        
    except Exception as e:
        print(f"处理文件 {input_file_path} 时出错: {str(e)}")

def process_sgd_dataset():
    """
    处理整个SGD数据集
    """
    # 定义路径
    # input_dir = "datasets/original_data/dstc8-schema-guided-dialogue/train"
    input_dir = "datasets/original_data/dstc8-schema-guided-dialogue/test"
    # output_dir = "datasets/sgd_processed_train"
    output_dir = "datasets/sgd_processed_test"
    
    # 获取当前脚本的绝对路径，并构建相对于项目根目录的路径
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    
    input_path = os.path.join(project_root, input_dir)
    output_path = os.path.join(project_root, output_dir)
    
    print(f"输入目录: {input_path}")
    print(f"输出目录: {output_path}")
    
    # 检查输入目录是否存在
    if not os.path.exists(input_path):
        print(f"错误: 输入目录不存在 {input_path}")
        return
    
    # 创建输出目录
    os.makedirs(output_path, exist_ok=True)
    
    # 获取所有JSON文件（除了schema.json）
    pattern = os.path.join(input_path, "dialogues_*.json")
    json_files = glob.glob(pattern)
    
    if not json_files:
        print(f"在 {input_path} 中未找到任何dialogues_*.json文件")
        return
    
    print(f"找到 {len(json_files)} 个文件需要处理")
    
    # 处理每个文件
    processed_count = 0
    error_count = 0
    
    for input_file in sorted(json_files):
        try:
            filename = os.path.basename(input_file)
            output_file = os.path.join(output_path, filename)
            
            process_json_file(input_file, output_file)
            processed_count += 1
            
        except Exception as e:
            print(f"处理文件 {input_file} 时发生错误: {str(e)}")
            error_count += 1
    
    print(f"\n处理完成!")
    print(f"成功处理: {processed_count} 个文件")
    print(f"处理失败: {error_count} 个文件")
    print(f"输出目录: {output_path}")



def main():
    """
    主函数
    """
    print("开始处理SGD数据集...")
    print("=" * 50)
    
    # 处理数据集
    process_sgd_dataset()

if __name__ == "__main__":
    main()
