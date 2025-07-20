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
    input_dir = "datasets/original_data/dstc8-schema-guided-dialogue/train"
    output_dir = "datasets/sgd_processed"
    
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

def verify_processing(input_file_path: str, output_file_path: str) -> bool:
    """
    验证处理结果
    
    Args:
        input_file_path: 原始文件路径
        output_file_path: 处理后文件路径
        
    Returns:
        验证是否通过
    """
    try:
        # 读取原始文件和处理后的文件
        with open(input_file_path, 'r', encoding='utf-8') as f:
            original_data = json.load(f)
        
        with open(output_file_path, 'r', encoding='utf-8') as f:
            processed_data = json.load(f)
        
        # 验证对话数量是否一致
        if len(original_data) != len(processed_data):
            print(f"错误: 对话数量不一致 - 原始: {len(original_data)}, 处理后: {len(processed_data)}")
            return False
        
        # 验证每个对话的结构
        for i, (orig_dialogue, proc_dialogue) in enumerate(zip(original_data, processed_data)):
            # 检查是否添加了personality字段
            if 'personality' not in proc_dialogue:
                print(f"错误: 对话 {i} 缺少personality字段")
                return False
            
            # 检查turns数量是否一致
            if len(orig_dialogue.get('turns', [])) != len(proc_dialogue.get('turns', [])):
                print(f"错误: 对话 {i} 的turns数量不一致")
                return False
            
            # 检查每个turn是否添加了transformed_utterance字段
            for j, (orig_turn, proc_turn) in enumerate(zip(orig_dialogue.get('turns', []), proc_dialogue.get('turns', []))):
                if 'utterance' in orig_turn and 'transformed_utterance' not in proc_turn:
                    print(f"错误: 对话 {i}, turn {j} 缺少transformed_utterance字段")
                    return False
        
        print(f"验证通过: {output_file_path}")
        return True
        
    except Exception as e:
        print(f"验证文件时出错: {str(e)}")
        return False

def main():
    """
    主函数
    """
    print("开始处理SGD数据集...")
    print("=" * 50)
    
    # 处理数据集
    process_sgd_dataset()
    
    # 验证处理结果（验证第一个文件作为示例）
    print("\n" + "=" * 50)
    print("验证处理结果...")
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    
    input_sample = os.path.join(project_root, "datasets/original_data/dstc8-schema-guided-dialogue/train/dialogues_001.json")
    output_sample = os.path.join(project_root, "datasets/sgd_processed/dialogues_001.json")
    
    if os.path.exists(input_sample) and os.path.exists(output_sample):
        verify_processing(input_sample, output_sample)
    else:
        print("无法找到样本文件进行验证")

if __name__ == "__main__":
    main()
