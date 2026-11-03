#!/usr/bin/env python3
"""
Data Preparation Script for Assignment 7.
This script downloads and prepares the Anthropic HH-RLHF dataset from Hugging Face.
"""

import os
import sys
import json
import random
import argparse
from typing import List, Dict, Any, Tuple
from tqdm import tqdm

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    from datasets import load_dataset
    DATASETS_AVAILABLE = True
except ImportError:
    DATASETS_AVAILABLE = False
    print("Datasets library not available. Please install with: pip install datasets")

from src.utils import save_json_data


def load_hh_rlhf_dataset(max_samples: int = 1000) -> Tuple[List[Dict], List[Dict]]:
    """
    Load and process the Anthropic HH-RLHF dataset.
    
    Args:
        max_samples: Maximum number of samples to load
        
    Returns:
        Tuple of (training_data, validation_data)
    """
    if not DATASETS_AVAILABLE:
        raise ImportError("datasets library is required. Install with: pip install datasets")
    
    print("📥 Loading Anthropic HH-RLHF dataset from Hugging Face...")
    
    if max_samples < 10:
        raise ValueError("max_samples must be at least 10")
    helpful = load_dataset("Anthropic/hh-rlhf", data_dir="helpful-base", split="train")
    harmless = load_dataset("Anthropic/hh-rlhf", data_dir="harmless-base", split="train")
    # Shuffle each subset before sampling and alternate to represent both sources.
    helpful = helpful.shuffle(seed=random.randrange(2**32))
    harmless = harmless.shuffle(seed=random.randrange(2**32))
    all_data = []
    seen = set()
    for index in range(max(len(helpful), len(harmless))):
        for dataset in (helpful, harmless):
            if index >= len(dataset):
                continue
            example = process_hh_example(dataset[index])
            if example and example['prompt'] not in seen:
                seen.add(example['prompt'])
                all_data.append(example)
            if len(all_data) >= max_samples:
                break
        if len(all_data) >= max_samples:
            break
    random.shuffle(all_data)
    # Splitting is performed once by prepare_dataset.
    return all_data, []


def process_hh_example(example: Dict) -> Dict:
    """
    Process a single example from the HH-RLHF dataset.
    
    Args:
        example: Raw example from the dataset
        
    Returns:
        Processed example in our format
    """
    try:
        # Extract the conversation components
        chosen = example.get('chosen', '')
        rejected = example.get('rejected', '')
        
        if not chosen or not rejected:
            return None
        
        # Extract the prompt (everything before the last Assistant response)
        # The format is typically: "Human: ... Assistant: ... Human: ... Assistant:"
        # We want to extract up to the last "Assistant:" as the prompt
        
        marker = "\n\nAssistant:"
        chosen_prefix, separator, chosen_response = chosen.rpartition(marker)
        rejected_prefix, rejected_separator, rejected_response = rejected.rpartition(marker)
        if not separator or not rejected_separator or chosen_prefix != rejected_prefix:
            return None
        prompt = (chosen_prefix + marker).strip()
        chosen_response = chosen_response.strip()
        rejected_response = rejected_response.strip()

        # Skip if responses are too short or too long
        if len(chosen_response) < 10 or len(chosen_response) > 500:
            return None
        if len(rejected_response) < 10 or len(rejected_response) > 500:
            return None
        
        return {
            'prompt': prompt,
            'chosen': prompt + " " + chosen_response,
            'rejected': prompt + " " + rejected_response
        }
        
    except Exception as e:
        print(f"⚠️  Error processing example: {e}")
        return None


def create_prompts_from_preferences(preference_data: List[Dict]) -> List[Dict]:
    """
    Extract unique prompts from preference data.
    
    Args:
        preference_data: List of preference examples
        
    Returns:
        List of unique prompts
    """
    unique_prompts = set()
    
    for example in preference_data:
        prompt = example['prompt']
        unique_prompts.add(prompt)
    
    return [{'prompt': prompt} for prompt in sorted(unique_prompts)]


def prepare_dataset(
    dataset_name: str = "Anthropic/hh-rlhf",
    output_dir: str = "data",
    max_samples: int = 1000,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1
):
    """
    Prepare the dataset for training.
    
    Args:
        dataset_name: Name of the dataset to load
        output_dir: Directory to save processed data
        max_samples: Maximum number of samples to process
        train_ratio: Ratio for training set
        val_ratio: Ratio for validation set
        test_ratio: Ratio for test set
    """
    # Validate ratios
    if any(not 0 < ratio < 1 for ratio in (train_ratio, val_ratio, test_ratio)) or abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
        raise ValueError("Train, validation, and test ratios must sum to 1.0")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load the dataset
    if dataset_name == "Anthropic/hh-rlhf":
        train_data, val_data = load_hh_rlhf_dataset(max_samples)
    else:
        raise ValueError(f"Dataset {dataset_name} not supported yet")
    
    train_end = int(len(train_data) * train_ratio)
    val_end = train_end + int(len(train_data) * val_ratio)
    train_data, val_data, test_data = (
        train_data[:train_end], train_data[train_end:val_end], train_data[val_end:]
    )
    if not all((train_data, val_data, test_data)):
        raise ValueError("Not enough valid examples for nonempty train/validation/test splits")

    print(f"📊 Data splits:")
    print(f"   Training: {len(train_data)} examples")
    print(f"   Validation: {len(val_data)} examples") 
    print(f"   Test: {len(test_data)} examples")
    
    # Create training prompts (unique prompts from training set)
    train_prompts = create_prompts_from_preferences(train_data)
    print(f"   Unique training prompts: {len(train_prompts)}")
    
    # Create evaluation prompts (unique prompts from test set)
    eval_prompts = create_prompts_from_preferences(test_data)
    print(f"   Unique evaluation prompts: {len(eval_prompts)}")
    
    # Save all data files
    save_json_data(train_prompts, os.path.join(output_dir, "train_prompts.json"))
    save_json_data(eval_prompts, os.path.join(output_dir, "eval_prompts.json"))
    save_json_data(create_prompts_from_preferences(val_data), os.path.join(output_dir, "val_prompts.json"))
    save_json_data(train_data, os.path.join(output_dir, "preference_data.json"))
    save_json_data(val_data, os.path.join(output_dir, "preference_data_val.json"))
    save_json_data(test_data, os.path.join(output_dir, "preference_data_test.json"))
    
    # Create a small sample for quick testing
    sample_size = min(10, len(train_data))
    sample_data = train_data[:sample_size]
    save_json_data(sample_data, os.path.join(output_dir, "preference_data_sample.json"))
    
    print(f"\n✅ Dataset preparation complete!")
    print(f"📁 Files saved to {output_dir}/:")
    print(f"   - train_prompts.json ({len(train_prompts)} prompts)")
    print(f"   - eval_prompts.json ({len(eval_prompts)} prompts)")
    print(f"   - preference_data.json ({len(train_data)} pairs)")
    print(f"   - preference_data_val.json ({len(val_data)} pairs)")
    print(f"   - preference_data_test.json ({len(test_data)} pairs)")
    print(f"   - preference_data_sample.json ({sample_size} pairs)")


def validate_data(data_dir: str = "data"):
    """
    Validate the prepared data files.
    
    Args:
        data_dir: Directory containing data files
    """
    print("🔍 Validating prepared data files...")
    
    required_files = [
        ("train_prompts.json", ["prompt"]),
        ("eval_prompts.json", ["prompt"]),
        ("val_prompts.json", ["prompt"]),
        ("preference_data_val.json", ["prompt", "chosen", "rejected"]),
        ("preference_data_test.json", ["prompt", "chosen", "rejected"]),
        ("preference_data.json", ["prompt", "chosen", "rejected"]),
    ]
    
    all_valid = True
    
    for filename, required_keys in required_files:
        filepath = os.path.join(data_dir, filename)
        
        if not os.path.exists(filepath):
            print(f"Missing file: {filepath}")
            all_valid = False
            continue
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            if not isinstance(data, list):
                print(f"{filename}: Data should be a list")
                all_valid = False
                continue
            
            if len(data) == 0:
                print(f"{filename}: File is empty")
                all_valid = False
                continue
            
            # Check required keys
            for i, item in enumerate(data):
                if not isinstance(item, dict):
                    print(f"{filename}: Item {i} should be a dictionary")
                    all_valid = False
                    break
                
                missing_keys = [key for key in required_keys if key not in item]
                if missing_keys:
                    print(f"{filename}: Item {i} missing keys: {missing_keys}")
                    all_valid = False
                    break
            else:
                print(f"{filename}: {len(data)} valid items")
                
        except json.JSONDecodeError as e:
            print(f"{filename}: Invalid JSON - {e}")
            all_valid = False
        except Exception as e:
            print(f"{filename}: Error - {e}")
            all_valid = False
    
    if all_valid:
        print("All data files are valid!")
    else:
        print("Some data files have issues. Please run preparation again.")
    return all_valid


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Prepare training data for Assignment 7")
    parser.add_argument(
        "--dataset",
        type=str,
        default="Anthropic/hh-rlhf",
        help="Dataset name from Hugging Face"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data",
        help="Output directory for prepared data"
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=1000,
        help="Maximum number of samples to process"
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Validate existing data files instead of preparing new ones"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )
    
    args = parser.parse_args()
    
    # Set random seed
    random.seed(args.seed)
    
    if args.validate:
        if not validate_data(args.output_dir):
            sys.exit(1)
    else:
        try:
            prepare_dataset(
                dataset_name=args.dataset,
                output_dir=args.output_dir,
                max_samples=args.max_samples
            )
            print("\nValidating prepared data...")
            validate_data(args.output_dir)
        except Exception as e:
            print(f"Error preparing dataset: {e}")
            sys.exit(1)


if __name__ == "__main__":
    main()
