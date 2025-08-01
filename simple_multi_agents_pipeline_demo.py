# 调用CompletePipeline进行一次转换
import logging
from multi_agents.complete_pipeline import CompletePipeline

# 启用警告级别日志（减少输出）
logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

pipeline = CompletePipeline()
def test_basic_functionality():
    input_data_path = "datasets/sgd_processed_train"
    output_directory = "output/mAPipelineOutput/1"
    batch_size = 1
    max_dialogues = 1
    random_seed = 42
    result = pipeline.run_complete_pipeline(input_data_path, output_directory, batch_size=batch_size,
                                            max_dialogues=max_dialogues, random_seed=random_seed)
    assert result.get("success", False)

test_basic_functionality()
print("Pipeline test completed successfully!")