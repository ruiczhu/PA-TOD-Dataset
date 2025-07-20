# DialogueLoader 使用文档

## 概述

`DialogueLoader` 是一个专为SGD数据集设计的对话数据加载器，提供多种灵活的数据访问方式。

## 功能特性

### 核心功能
1. **根据dialogue_id获取指定对话** - 精确查找特定对话
2. **根据服务类型获取对话** - 按服务类型筛选，支持随机抽样
3. **随机获取对话** - 随机选择指定数量的对话
4. **统计所有服务类型** - 获取数据集中所有服务类型及统计信息
5. **Act类型分析** - 按speaker分类统计所有act类型

### 高级特性
- **懒加载模式** - 可选择立即加载全部数据或按需加载
- **随机种子支持** - 确保随机选择结果可重现
- **详细统计信息** - 提供数据集概览和各维度统计

## 快速开始

### 基本初始化

```python
from MPEAF.dialogue_loader import DialogueLoader

# 初始化加载器（立即加载所有数据）
loader = DialogueLoader(data_dir="datasets/sgd_processed_train", load_all=True)

# 或使用懒加载模式（节省内存）
loader_lazy = DialogueLoader(data_dir="datasets/sgd_processed_train", load_all=False)
```

### 1. 根据dialogue_id获取对话

```python
# 获取指定对话
dialogue = loader.get_dialogue_by_id("1_00000")

if dialogue:
    print(f"对话ID: {dialogue['dialogue_id']}")
    print(f"服务类型: {dialogue['services']}")
    print(f"个性化信息: {dialogue['personality']}")
    print(f"轮次数量: {len(dialogue['turns'])}")
    
    # 访问具体轮次
    for turn in dialogue['turns']:
        speaker = turn['speaker']
        utterance = turn['utterance']
        transformed = turn['transformed_utterance']
        print(f"{speaker}: {utterance}")
```

### 2. 根据服务类型获取对话

```python
# 获取所有餐厅相关对话
restaurant_dialogues = loader.get_dialogues_by_service("Restaurants_1")

# 随机获取5个酒店相关对话（设置随机种子）
hotel_dialogues = loader.get_dialogues_by_service(
    service="Hotels_1", 
    count=5, 
    random_seed=42
)

print(f"获取了 {len(hotel_dialogues)} 个酒店对话")
```

### 3. 随机获取对话

```python
# 随机获取10个对话
random_dialogues = loader.get_random_dialogues(count=10, random_seed=123)

for dialogue in random_dialogues:
    print(f"{dialogue['dialogue_id']}: {dialogue['services']}")
```

### 4. 获取服务类型信息

```python
# 获取所有服务类型
all_services = loader.get_all_services()
print(f"共有 {len(all_services)} 种服务类型:")
print(all_services)

# 获取服务类型统计
service_stats = loader.get_service_statistics()
for service, count in service_stats.items():
    print(f"{service}: {count} 个对话")
```

### 5. Act类型分析

```python
# 按speaker获取所有act类型
acts_by_speaker = loader.get_acts_by_speaker()
print("USER Acts:", acts_by_speaker['USER'])
print("SYSTEM Acts:", acts_by_speaker['SYSTEM'])

# 获取act频次统计
act_stats = loader.get_act_statistics()
for speaker, acts in act_stats.items():
    print(f"\n{speaker} Act频次:")
    for act, count in sorted(acts.items(), key=lambda x: x[1], reverse=True):
        print(f"  {act}: {count} 次")
```

### 6. 数据集概览

```python
# 打印完整的数据集概览
loader.print_summary()

# 获取概览数据（用于程序处理）
summary = loader.get_data_summary()
print(f"总对话数: {summary['total_dialogues']}")
print(f"总轮次数: {summary['total_turns']}")
```

## 实际应用场景

### 场景1: 构建训练数据集

```python
# 为对话系统训练收集特定领域数据
training_data = []

# 从各个服务类型中采样
services_of_interest = ["Restaurants_1", "Hotels_1", "Flights_1"]
for service in services_of_interest:
    dialogues = loader.get_dialogues_by_service(service, count=100, random_seed=42)
    training_data.extend(dialogues)

print(f"构建了包含 {len(training_data)} 个对话的训练集")
```

### 场景2: 数据质量分析

```python
# 分析对话长度分布
sample_dialogues = loader.get_random_dialogues(count=1000, random_seed=42)
turn_counts = [len(d['turns']) for d in sample_dialogues]

avg_turns = sum(turn_counts) / len(turn_counts)
print(f"平均对话轮次: {avg_turns:.1f}")
```

### 场景3: 创建测试集

```python
# 从每个服务类型中均匀采样创建测试集
test_set = []
all_services = loader.get_all_services()

for service in all_services:
    # 从每个服务中选择2个对话
    service_dialogues = loader.get_dialogues_by_service(
        service, count=2, random_seed=42
    )
    test_set.extend(service_dialogues)

print(f"创建了包含 {len(test_set)} 个对话的平衡测试集")
```

### 场景4: 数据导出

```python
import json

# 筛选并保存数据
selected_dialogues = loader.get_dialogues_by_service("Travel_1", count=50)

# 保存到文件
with open("travel_dialogues.json", 'w', encoding='utf-8') as f:
    json.dump(selected_dialogues, f, ensure_ascii=False, indent=2)

print("数据已保存到 travel_dialogues.json")
```

## 性能考虑

- **load_all=True**: 初始化时间较长，但后续访问速度快，适合频繁访问
- **load_all=False**: 初始化快速，按需加载，适合一次性操作或内存受限场景
- **随机种子**: 建议在需要可重现结果时设置随机种子

## 数据结构

每个对话包含以下字段：
- `dialogue_id`: 对话唯一标识符
- `services`: 涉及的服务类型列表
- `personality`: 个性化信息（预处理后添加的字段）
- `turns`: 对话轮次列表
  - `speaker`: 说话者（USER/SYSTEM）
  - `utterance`: 原始话语
  - `transformed_utterance`: 转换后话语（预处理后添加的字段）
  - `frames`: 对话框架信息（包含act、slot等）

## 错误处理

加载器包含完善的错误处理机制：
- 文件不存在时抛出 `FileNotFoundError`
- 无效的dialogue_id返回 `None`
- 不存在的服务类型会显示警告并返回空列表

## 扩展使用

如需处理其他数据目录（如测试集），只需改变 `data_dir` 参数：

```python
# 加载测试数据
test_loader = DialogueLoader(data_dir="datasets/sgd_processed_test", load_all=True)
```
