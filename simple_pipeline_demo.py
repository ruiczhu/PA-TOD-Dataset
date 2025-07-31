from MPEAF.transformation_pipeline import TransformationPipeline


pipeline = TransformationPipeline(data_dir="datasets/sgd_processed_train",output_dir="output")
        
services = pipeline.get_available_services()
print(f"Available services: {len(services)}")

transformed_dialogues = pipeline.transform_dialogues(
    dialogue_count=20,
    random_seed=42,
    temperature=0.7,
    max_tokens=2000
)

print(f"Results: {len(transformed_dialogues)} dialogues transformed")

# Show sample results
for i, dialogue in enumerate(transformed_dialogues):
    dialogue_id = dialogue.get('dialogue_id', f'dialogue_{i}')
    turns = dialogue.get('turns', [])
    transformed_count = sum(1 for turn in turns if 'transformed_utterance' in turn)
    print(f"  • {dialogue_id}: {transformed_count}/{len(turns)} turns transformed")

print(f"\nResults saved to: output/")