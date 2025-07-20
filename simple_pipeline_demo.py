"""
Simple example demonstrating the streamlined TransformationPipeline usage
"""

import sys
import os

# Add MPEAF directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'MPEAF'))

def main():
    """Demonstrate simplified pipeline usage"""
    print("=== Simplified Transformation Pipeline Demo ===")
    
    try:
        from MPEAF.transformation_pipeline import TransformationPipeline
        
        # Initialize pipeline with minimal configuration
        pipeline = TransformationPipeline(
            data_dir="datasets/sgd_processed_train",
            output_dir="output"
        )
        
        print("Pipeline initialized")
        
        # Show available services
        services = pipeline.get_available_services()
        print(f"Available services: {len(services)}")
        
        # Transform 4 dialogues for demonstration
        print("\nTransforming 4 dialogues...")
        transformed_dialogues = pipeline.transform_dialogues(
            dialogue_count=2,
            random_seed=42,
            temperature=0.7,
            max_tokens=2000
        )
        
        print(f"Transformation completed!")
        print(f"Results: {len(transformed_dialogues)} dialogues transformed")
        
        # Show sample results
        for i, dialogue in enumerate(transformed_dialogues):
            dialogue_id = dialogue.get('dialogue_id', f'dialogue_{i}')
            turns = dialogue.get('turns', [])
            transformed_count = sum(1 for turn in turns if 'transformed_utterance' in turn)
            print(f"  • {dialogue_id}: {transformed_count}/{len(turns)} turns transformed")
        
        print(f"\nResults saved to: output/")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
