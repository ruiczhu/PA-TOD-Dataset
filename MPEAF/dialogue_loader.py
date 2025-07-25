import json
import os
import glob
import random
from typing import Dict, List, Any, Optional, Set, Tuple
from pathlib import Path
from collections import defaultdict, Counter

class DialogueLoader:
    """
    SGD数据集对话加载器类
    
    提供多种方式加载和过滤对话数据：
    1. 根据dialogue_id获取指定对话
    2. 根据services类型获取对话（可指定数量和随机种子）
    3. 随机获取n个对话（可设置随机种子）
    4. 统计所有services类型
    5. 依据speaker分类统计所有act类型
    """
    
    def __init__(self, data_dir: str = "datasets/sgd_processed_train", load_all: bool = True):
        """
        初始化数据加载器
        
        Args:
            data_dir: 数据目录路径
            load_all: 是否在初始化时加载所有数据到内存
        """
        # 获取绝对路径
        if not os.path.isabs(data_dir):
            script_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(script_dir)
            self.data_dir = os.path.join(project_root, data_dir)
        else:
            self.data_dir = data_dir
            
        self.load_all = load_all
        self._dialogues = {}  # dialogue_id -> dialogue_data
        self._services_index = defaultdict(list)  # service -> [dialogue_ids]
        self._all_services = set()
        self._all_acts = {"USER": set(), "SYSTEM": set()}
        self._loaded_files = set()
        
        print(f"初始化DialogueLoader，数据目录: {self.data_dir}")
        
        # 检查数据目录是否存在
        if not os.path.exists(self.data_dir):
            raise FileNotFoundError(f"数据目录不存在: {self.data_dir}")
        
        # 获取所有JSON文件
        pattern = os.path.join(self.data_dir, "dialogues_*.json")
        self.json_files = sorted(glob.glob(pattern))
        
        if not self.json_files:
            raise FileNotFoundError(f"在 {self.data_dir} 中未找到任何dialogues_*.json文件")
        
        print(f"找到 {len(self.json_files)} 个数据文件")
        
        # 根据load_all参数决定是否立即加载所有数据
        if self.load_all:
            self._load_all_data()
        else:
            # 只加载索引信息
            self._build_indexes()
    
    def _load_file(self, file_path: str) -> List[Dict[str, Any]]:
        """加载单个JSON文件"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"加载文件 {file_path} 失败: {str(e)}")
            return []
    
    def _load_all_data(self):
        """加载所有数据到内存"""
        print("正在加载所有数据到内存...")
        
        for file_path in self.json_files:
            filename = os.path.basename(file_path)
            if filename in self._loaded_files:
                continue
                
            dialogues = self._load_file(file_path)
            
            for dialogue in dialogues:
                dialogue_id = dialogue.get('dialogue_id')
                if dialogue_id:
                    self._dialogues[dialogue_id] = dialogue
                    
                    # 构建服务索引
                    services = dialogue.get('services', [])
                    for service in services:
                        self._all_services.add(service)
                        self._services_index[service].append(dialogue_id)
                    
                    # 提取所有act类型
                    self._extract_acts_from_dialogue(dialogue)
            
            self._loaded_files.add(filename)
        
        print(f"数据加载完成！总共 {len(self._dialogues)} 个对话")
        print(f"发现 {len(self._all_services)} 种服务类型")
        print(f"USER acts: {len(self._all_acts['USER'])} 种")
        print(f"SYSTEM acts: {len(self._all_acts['SYSTEM'])} 种")
    
    def _build_indexes(self):
        """仅构建索引，不加载完整数据"""
        print("正在构建索引...")
        
        for file_path in self.json_files:
            dialogues = self._load_file(file_path)
            
            for dialogue in dialogues:
                dialogue_id = dialogue.get('dialogue_id')
                if dialogue_id:
                    # 只存储基本信息用于索引
                    services = dialogue.get('services', [])
                    for service in services:
                        self._all_services.add(service)
                        self._services_index[service].append(dialogue_id)
                    
                    # 提取act类型
                    self._extract_acts_from_dialogue(dialogue)
        
        print(f"索引构建完成！发现 {len(self._all_services)} 种服务类型")
    
    def _extract_acts_from_dialogue(self, dialogue: Dict[str, Any]):
        """从对话中提取所有act类型"""
        turns = dialogue.get('turns', [])
        
        for turn in turns:
            speaker = turn.get('speaker')
            if speaker in ['USER', 'SYSTEM']:
                frames = turn.get('frames', [])
                for frame in frames:
                    actions = frame.get('actions', [])
                    for action in actions:
                        act = action.get('act')
                        if act:
                            self._all_acts[speaker].add(act)
    
    def get_dialogue_by_id(self, dialogue_id: str) -> Optional[Dict[str, Any]]:
        """
        根据dialogue_id获取指定对话
        
        Args:
            dialogue_id: 对话ID
            
        Returns:
            对话数据字典，如果未找到返回None
        """
        if self.load_all:
            return self._dialogues.get(dialogue_id)
        else:
            # 懒加载模式：在需要时搜索文件
            for file_path in self.json_files:
                dialogues = self._load_file(file_path)
                for dialogue in dialogues:
                    if dialogue.get('dialogue_id') == dialogue_id:
                        return dialogue
            return None
    
    def get_dialogues_by_service(self, service: str, count: Optional[int] = None, 
                                random_seed: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        根据指定services类型获取对话
        
        Args:
            service: 服务类型（如 'Restaurants_1'）
            count: 指定返回的对话数量，如果为None则返回所有匹配的对话
            random_seed: 随机种子，用于随机选择对话
            
        Returns:
            对话数据列表
        """
        if service not in self._all_services:
            print(f"警告: 服务类型 '{service}' 不存在")
            return []
        
        dialogue_ids = self._services_index[service].copy()
        
        # 如果指定了数量和随机种子，则随机选择
        if count is not None and len(dialogue_ids) > count:
            if random_seed is not None:
                random.seed(random_seed)
            dialogue_ids = random.sample(dialogue_ids, count)
        
        dialogues = []
        if self.load_all:
            # 从内存中获取
            dialogues = [self._dialogues[did] for did in dialogue_ids if did in self._dialogues]
        else:
            # 懒加载模式：搜索文件
            dialogue_ids_set = set(dialogue_ids)
            for file_path in self.json_files:
                if not dialogue_ids_set:  # 已找到所有需要的对话
                    break
                    
                file_dialogues = self._load_file(file_path)
                for dialogue in file_dialogues:
                    did = dialogue.get('dialogue_id')
                    if did in dialogue_ids_set:
                        dialogues.append(dialogue)
                        dialogue_ids_set.remove(did)
        
        print(f"从服务类型 '{service}' 中获取了 {len(dialogues)} 个对话")
        return dialogues
    
    def get_random_dialogues(self, count: int, random_seed: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        随机获取n个对话
        
        Args:
            count: 要获取的对话数量
            random_seed: 随机种子
            
        Returns:
            随机选择的对话数据列表
        """
        if random_seed is not None:
            random.seed(random_seed)
        
        if self.load_all:
            # 从内存中随机选择
            all_dialogue_ids = list(self._dialogues.keys())
            if len(all_dialogue_ids) < count:
                print(f"警告: 请求 {count} 个对话，但只有 {len(all_dialogue_ids)} 个对话可用")
                count = len(all_dialogue_ids)
            
            selected_ids = random.sample(all_dialogue_ids, count)
            dialogues = [self._dialogues[did] for did in selected_ids]
        else:
            # 懒加载模式：需要先收集所有对话ID
            all_dialogue_ids = []
            for file_path in self.json_files:
                file_dialogues = self._load_file(file_path)
                for dialogue in file_dialogues:
                    did = dialogue.get('dialogue_id')
                    if did:
                        all_dialogue_ids.append(did)
            
            if len(all_dialogue_ids) < count:
                print(f"警告: 请求 {count} 个对话，但只有 {len(all_dialogue_ids)} 个对话可用")
                count = len(all_dialogue_ids)
            
            selected_ids = random.sample(all_dialogue_ids, count)
            selected_ids_set = set(selected_ids)
            
            # 加载选中的对话
            dialogues = []
            for file_path in self.json_files:
                if not selected_ids_set:
                    break
                    
                file_dialogues = self._load_file(file_path)
                for dialogue in file_dialogues:
                    did = dialogue.get('dialogue_id')
                    if did in selected_ids_set:
                        dialogues.append(dialogue)
                        selected_ids_set.remove(did)
        
        print(f"随机获取了 {len(dialogues)} 个对话")
        return dialogues
    
    def get_all_services(self) -> List[str]:
        """
        获取所有服务类型
        
        Returns:
            所有服务类型的列表
        """
        return sorted(list(self._all_services))
    
    def get_service_statistics(self) -> Dict[str, int]:
        """
        获取每个服务类型的对话数量统计
        
        Returns:
            服务类型 -> 对话数量的字典
        """
        stats = {}
        for service in self._all_services:
            stats[service] = len(self._services_index[service])
        return dict(sorted(stats.items()))
    
    def get_acts_by_speaker(self) -> Dict[str, List[str]]:
        """
        依据speaker分类获取所有act类型
        
        Returns:
            speaker -> act类型列表的字典
        """
        return {
            speaker: sorted(list(acts)) 
            for speaker, acts in self._all_acts.items()
        }
