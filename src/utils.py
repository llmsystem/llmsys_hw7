"""
Utility functions for Assignment 6 - RLHF Training.
This module contains helper functions for data processing, logging, and evaluation.
"""

import json
import os
import random
import logging
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from typing import List, Dict, Any, Tuple, Optional
from pathlib import Path
from datetime import datetime
from transformers import PreTrainedTokenizer
import pandas as pd

logger = logging.getLogger(__name__)


def set_seed(seed: int) -> None:
    """
    Set random seeds for reproducibility.
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    logger.info(f"Set random seed to {seed}")


def setup_logging(log_dir: str, experiment_name: str) -> logging.Logger:
    """
    Set up logging configuration.
    
    Args:
        log_dir: Directory to save log files
        experiment_name: Name of the experiment
        
    Returns:
        Configured logger
    """
    # Create log directory
    os.makedirs(log_dir, exist_ok=True)
    
    # Create log filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"{experiment_name}_{timestamp}.log")
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.info(f"Logging configured. Log file: {log_file}")
    
    return logger


def load_json_data(file_path: str) -> List[Dict[str, Any]]:
    """
    Load data from a JSON file.
    
    Args:
        file_path: Path to the JSON file
        
    Returns:
        List of data entries
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        logger.info(f"Loaded {len(data)} entries from {file_path}")
        return data
    except FileNotFoundError:
        logger.error(f"File not found: {file_path}")
        raise
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in {file_path}: {e}")
        raise


def save_json_data(data: List[Dict[str, Any]], file_path: str) -> None:
    """
    Save data to a JSON file.
    
    Args:
        data: Data to save
        file_path: Path to save the JSON file
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Saved {len(data)} entries to {file_path}")


def create_preference_dataset(
    prompts: List[str], 
    responses_a: List[str], 
    responses_b: List[str],
    preferences: List[int]  # 0 for A, 1 for B
) -> List[Dict[str, str]]:
    """
    Create a preference dataset for reward model training.
    
    Args:
        prompts: List of prompts
        responses_a: List of responses A
        responses_b: List of responses B
        preferences: List of preferences (0 for A, 1 for B)
        
    Returns:
        List of preference pairs
    """
    preference_data = []
    
    for prompt, resp_a, resp_b, pref in zip(prompts, responses_a, responses_b, preferences):
        if pref == 0:  # A is preferred
            chosen = f"{prompt} {resp_a}"
            rejected = f"{prompt} {resp_b}"
        else:  # B is preferred
            chosen = f"{prompt} {resp_b}"
            rejected = f"{prompt} {resp_a}"
        
        preference_data.append({
            "prompt": prompt,
            "chosen": chosen,
            "rejected": rejected
        })
    
    return preference_data


def split_data(
    data: List[Any], 
    train_ratio: float = 0.8, 
    val_ratio: float = 0.1, 
    test_ratio: float = 0.1,
    shuffle: bool = True
) -> Tuple[List[Any], List[Any], List[Any]]:
    """
    Split data into train, validation, and test sets.
    
    Args:
        data: Data to split
        train_ratio: Ratio for training set
        val_ratio: Ratio for validation set
        test_ratio: Ratio for test set
        shuffle: Whether to shuffle data before splitting
        
    Returns:
        Tuple of (train_data, val_data, test_data)
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        "Ratios must sum to 1.0"
    
    if shuffle:
        data = data.copy()
        random.shuffle(data)
    
    n_total = len(data)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)
    
    train_data = data[:n_train]
    val_data = data[n_train:n_train + n_val]
    test_data = data[n_train + n_val:]
    
    logger.info(f"Data split: {len(train_data)} train, {len(val_data)} val, {len(test_data)} test")
    
    return train_data, val_data, test_data


def create_batch_iterator(
    data: List[Any], 
    batch_size: int, 
    shuffle: bool = True
) -> List[List[Any]]:
    """
    Create batches from data.
    
    Args:
        data: Data to batch
        batch_size: Size of each batch
        shuffle: Whether to shuffle data
        
    Returns:
        List of batches
    """
    if shuffle:
        data = data.copy()
        random.shuffle(data)
    
    batches = []
    for i in range(0, len(data), batch_size):
        batch = data[i:i + batch_size]
        batches.append(batch)
    
    return batches


def compute_text_statistics(texts: List[str], tokenizer: PreTrainedTokenizer) -> Dict[str, float]:
    """
    Compute statistics for a list of texts.
    
    Args:
        texts: List of text strings
        tokenizer: Tokenizer for text processing
        
    Returns:
        Dictionary of text statistics
    """
    if not texts:
        return {}
    
    # Word-level statistics
    word_counts = [len(text.split()) for text in texts]
    char_counts = [len(text) for text in texts]
    
    # Token-level statistics
    token_counts = []
    for text in texts:
        tokens = tokenizer.encode(text, add_special_tokens=False)
        token_counts.append(len(tokens))
    
    return {
        'mean_word_count': np.mean(word_counts),
        'std_word_count': np.std(word_counts),
        'mean_char_count': np.mean(char_counts),
        'std_char_count': np.std(char_counts),
        'mean_token_count': np.mean(token_counts),
        'std_token_count': np.std(token_counts),
        'min_word_count': min(word_counts),
        'max_word_count': max(word_counts),
        'num_texts': len(texts)
    }


def plot_training_curves(
    metrics_history: List[Dict[str, float]], 
    save_path: str,
    title: str = "Training Curves"
) -> None:
    """
    Plot training curves from metrics history.
    
    Args:
        metrics_history: List of metrics dictionaries
        save_path: Path to save the plot
        title: Plot title
    """
    if not metrics_history:
        logger.warning("No metrics to plot")
        return
    
    # Convert to DataFrame for easier plotting
    df = pd.DataFrame(metrics_history)
    
    # X-axis: prefer 'step' column if available
    x = df['step'] if 'step' in df.columns else df.index
    
    # Create subplots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(title, fontsize=16)
    
    # Top-left: losses
    if 'policy_loss' in df.columns:
        axes[0, 0].plot(x, df['policy_loss'], label='Policy Loss')
        axes[0, 0].set_title('Policy Loss')
        axes[0, 0].set_xlabel('Step')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].grid(True)
    else:
        plotted = False
        if 'train_loss' in df.columns:
            axes[0, 0].plot(x, df['train_loss'], label='Train Loss')
            plotted = True
        if 'val_loss' in df.columns:
            axes[0, 0].plot(x, df['val_loss'], label='Val Loss')
            plotted = True
        if plotted:
            axes[0, 0].set_title('Loss')
            axes[0, 0].set_xlabel('Step' if 'step' in df.columns else 'Index')
            axes[0, 0].set_ylabel('Loss')
            axes[0, 0].legend()
            axes[0, 0].grid(True)
    
    # Top-right: value/accuracy
    if 'value_loss' in df.columns:
        axes[0, 1].plot(x, df['value_loss'], label='Value Loss')
        axes[0, 1].set_title('Value Loss')
        axes[0, 1].set_xlabel('Step')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].grid(True)
    else:
        plotted = False
        if 'train_accuracy' in df.columns:
            axes[0, 1].plot(x, df['train_accuracy'], label='Train Acc')
            plotted = True
        if 'val_accuracy' in df.columns:
            axes[0, 1].plot(x, df['val_accuracy'], label='Val Acc')
            plotted = True
        if plotted:
            axes[0, 1].set_title('Accuracy')
            axes[0, 1].set_xlabel('Step' if 'step' in df.columns else 'Index')
            axes[0, 1].set_ylabel('Accuracy')
            axes[0, 1].legend()
            axes[0, 1].grid(True)
    
    # Bottom-left: rewards or reward diff
    if 'reward_mean' in df.columns:
        axes[1, 0].plot(x, df['reward_mean'], label='Reward Mean')
        if 'reward_std' in df.columns:
            axes[1, 0].fill_between(
                x, 
                df['reward_mean'] - df['reward_std'],
                df['reward_mean'] + df['reward_std'],
                alpha=0.3
            )
        axes[1, 0].set_title('Average Reward')
        axes[1, 0].set_xlabel('Step')
        axes[1, 0].set_ylabel('Reward')
        axes[1, 0].grid(True)
    else:
        plotted = False
        if 'train_reward_diff' in df.columns:
            axes[1, 0].plot(x, df['train_reward_diff'], label='Train Reward Diff')
            plotted = True
        if 'val_reward_diff' in df.columns:
            axes[1, 0].plot(x, df['val_reward_diff'], label='Val Reward Diff')
            plotted = True
        if plotted:
            axes[1, 0].set_title('Reward Difference (chosen - rejected)')
            axes[1, 0].set_xlabel('Step' if 'step' in df.columns else 'Index')
            axes[1, 0].set_ylabel('Reward Diff')
            axes[1, 0].legend()
            axes[1, 0].grid(True)
    
    # Bottom-right: KL or reward means
    if 'kl_divergence' in df.columns:
        axes[1, 1].plot(x, df['kl_divergence'], label='KL Divergence')
        axes[1, 1].set_title('KL Divergence')
        axes[1, 1].set_xlabel('Step')
        axes[1, 1].set_ylabel('KL Divergence')
        axes[1, 1].grid(True)
    else:
        plotted = False
        for col, label in [
            ('train_chosen_reward_mean', 'Train Chosen'),
            ('train_rejected_reward_mean', 'Train Rejected'),
            ('val_chosen_reward_mean', 'Val Chosen'),
            ('val_rejected_reward_mean', 'Val Rejected'),
        ]:
            if col in df.columns:
                axes[1, 1].plot(x, df[col], label=label)
                plotted = True
        if plotted:
            axes[1, 1].set_title('Chosen/Rejected Reward Means')
            axes[1, 1].set_xlabel('Step' if 'step' in df.columns else 'Index')
            axes[1, 1].set_ylabel('Reward')
            axes[1, 1].legend()
            axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved training curves to {save_path}")


def plot_reward_distribution(
    rewards: List[float], 
    save_path: str,
    title: str = "Reward Distribution",
    bins: int = 50
) -> None:
    """
    Plot histogram of reward values.
    
    Args:
        rewards: List of reward values
        save_path: Path to save the plot
        title: Plot title
        bins: Number of histogram bins
    """
    plt.figure(figsize=(10, 6))
    
    plt.hist(rewards, bins=bins, alpha=0.7, edgecolor='black')
    plt.axvline(np.mean(rewards), color='red', linestyle='--', label=f'Mean: {np.mean(rewards):.3f}')
    plt.axvline(np.median(rewards), color='green', linestyle='--', label=f'Median: {np.median(rewards):.3f}')
    
    plt.title(title)
    plt.xlabel('Reward Value')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved reward distribution plot to {save_path}")


def compare_models_side_by_side(
    prompts: List[str],
    base_responses: List[str],
    rlhf_responses: List[str],
    base_rewards: List[float],
    rlhf_rewards: List[float],
    save_path: str,
    num_examples: int = 5
) -> None:
    """
    Create a side-by-side comparison of model outputs.
    
    Args:
        prompts: List of prompts
        base_responses: Responses from base model
        rlhf_responses: Responses from RLHF model
        base_rewards: Rewards for base responses
        rlhf_rewards: Rewards for RLHF responses
        save_path: Path to save the comparison
        num_examples: Number of examples to include
    """
    comparison_data = []
    
    # Select examples for comparison
    indices = random.sample(range(len(prompts)), min(num_examples, len(prompts)))
    
    for i in indices:
        comparison_data.append({
            'prompt': prompts[i],
            'base_response': base_responses[i],
            'base_reward': base_rewards[i],
            'rlhf_response': rlhf_responses[i],
            'rlhf_reward': rlhf_rewards[i],
            'improvement': rlhf_rewards[i] - base_rewards[i]
        })
    
    # Save as JSON for easy viewing
    save_json_data(comparison_data, save_path)
    
    logger.info(f"Saved model comparison to {save_path}")


def create_summary_report(
    base_metrics: Dict[str, float],
    rlhf_metrics: Dict[str, float],
    training_time: float,
    save_path: str
) -> None:
    """
    Create a summary report of the RLHF training results.
    
    Args:
        base_metrics: Metrics from base model evaluation
        rlhf_metrics: Metrics from RLHF model evaluation
        training_time: Total training time in seconds
        save_path: Path to save the report
    """
    report = {
        'training_summary': {
            'training_time_seconds': training_time,
            'training_time_minutes': training_time / 60,
            'training_time_hours': training_time / 3600
        },
        'base_model_metrics': base_metrics,
        'rlhf_model_metrics': rlhf_metrics,
        'improvements': {}
    }
    
    # Compute improvements
    for metric in base_metrics:
        if metric in rlhf_metrics:
            base_val = base_metrics[metric]
            rlhf_val = rlhf_metrics[metric]
            
            report['improvements'][f'{metric}_improvement_abs'] = rlhf_val - base_val
    
    # Save report
    save_json_data([report], save_path)
    
    logger.info(f"Saved summary report to {save_path}")


def validate_data_format(data: List[Dict], required_keys: List[str]) -> bool:
    """
    Validate that data has the required format.
    
    Args:
        data: Data to validate
        required_keys: List of required keys in each data entry
        
    Returns:
        True if data is valid, False otherwise
    """
    if not isinstance(data, list):
        logger.error("Data must be a list")
        return False
    
    for i, entry in enumerate(data):
        if not isinstance(entry, dict):
            logger.error(f"Entry {i} must be a dictionary")
            return False
        
        for key in required_keys:
            if key not in entry:
                logger.error(f"Entry {i} missing required key: {key}")
                return False
    
    logger.info(f"Data validation passed for {len(data)} entries")
    return True


def clean_text(text: str) -> str:
    """
    Clean and normalize text.
    
    Args:
        text: Text to clean
        
    Returns:
        Cleaned text
    """
    # Remove extra whitespace
    text = ' '.join(text.split())
    
    # Remove leading/trailing whitespace
    text = text.strip()
    
    return text


def truncate_text(text: str, max_length: int, tokenizer: PreTrainedTokenizer) -> str:
    """
    Truncate text to a maximum number of tokens.
    
    Args:
        text: Text to truncate
        max_length: Maximum number of tokens
        tokenizer: Tokenizer to use
        
    Returns:
        Truncated text
    """
    tokens = tokenizer.encode(text, add_special_tokens=False)
    
    if len(tokens) <= max_length:
        return text
    
    # Truncate tokens and decode back to text
    truncated_tokens = tokens[:max_length]
    truncated_text = tokenizer.decode(truncated_tokens, skip_special_tokens=True)
    
    return truncated_text


def estimate_gpu_memory_usage(model, batch_size: int = 1, sequence_length: int = 512) -> Dict[str, float]:
    """
    Estimate GPU memory usage for a model.
    
    Args:
        model: PyTorch model
        batch_size: Batch size for estimation
        sequence_length: Sequence length for estimation
        
    Returns:
        Dictionary with memory estimates in GB
    """
    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())
    
    # Estimate memory usage (rough approximation)
    # Parameters: 4 bytes per parameter (float32)
    param_memory = num_params * 4 / (1024**3)  # GB
    
    # Activations: estimate based on batch size and sequence length
    # This is a rough estimate and can vary significantly
    activation_memory = batch_size * sequence_length * model.config.hidden_size * 4 / (1024**3)
    
    # Gradients: same as parameters
    gradient_memory = param_memory
    
    # Optimizer states (Adam): ~2x parameter memory
    optimizer_memory = param_memory * 2
    
    total_memory = param_memory + activation_memory + gradient_memory + optimizer_memory
    
    return {
        'parameters_gb': param_memory,
        'activations_gb': activation_memory,
        'gradients_gb': gradient_memory,
        'optimizer_gb': optimizer_memory,
        'total_gb': total_memory,
        'recommended_gpu_gb': total_memory * 1.5  # Add 50% buffer
    }


def format_time(seconds: float) -> str:
    """
    Format time in seconds to human-readable string.
    
    Args:
        seconds: Time in seconds
        
    Returns:
        Formatted time string
    """
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}h"


def get_device_info() -> Dict[str, Any]:
    """
    Get information about available compute devices.
    
    Returns:
        Dictionary with device information
    """
    device_info = {
        'cuda_available': torch.cuda.is_available(),
        'cuda_device_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
        'current_device': torch.cuda.current_device() if torch.cuda.is_available() else None,
    }
    
    if torch.cuda.is_available():
        device_info['cuda_version'] = torch.version.cuda
        device_info['gpu_names'] = [
            torch.cuda.get_device_name(i) 
            for i in range(torch.cuda.device_count())
        ]
        device_info['gpu_memory'] = [
            {
                'total_gb': torch.cuda.get_device_properties(i).total_memory / (1024**3),
                'allocated_gb': torch.cuda.memory_allocated(i) / (1024**3),
                'cached_gb': torch.cuda.memory_reserved(i) / (1024**3)
            }
            for i in range(torch.cuda.device_count())
        ]
    
    return device_info


def save_model_outputs(
    prompts: List[str],
    responses: List[str],
    rewards: List[float],
    save_path: str,
    model_name: str = "model"
) -> None:
    """
    Save model outputs for analysis.
    
    Args:
        prompts: List of input prompts
        responses: List of model responses
        rewards: List of reward scores
        save_path: Path to save outputs
        model_name: Name of the model
    """
    outputs = []
    
    for prompt, response, reward in zip(prompts, responses, rewards):
        outputs.append({
            'prompt': prompt,
            'response': response,
            'reward': reward,
            'model': model_name,
            'timestamp': datetime.now().isoformat()
        })
    
    save_json_data(outputs, save_path)
    logger.info(f"Saved {len(outputs)} outputs to {save_path}")


class MetricsTracker:
    """
    Class for tracking and managing training metrics.
    """
    
    def __init__(self):
        """Initialize metrics tracker."""
        self.metrics_history = []
        self.best_metrics = {}
        self.current_step = 0
    
    def update(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        """
        Update metrics for the current step.
        
        Args:
            metrics: Dictionary of metric values
            step: Step number (auto-increment if None)
        """
        if step is None:
            step = self.current_step
            self.current_step += 1
        
        # Add step number to metrics
        metrics_with_step = {'step': step, **metrics}
        self.metrics_history.append(metrics_with_step)
        
        # Update best metrics
        for key, value in metrics.items():
            if key not in self.best_metrics:
                self.best_metrics[key] = {'value': value, 'step': step}
            elif ('loss' in key.lower() and value < self.best_metrics[key]['value']) or \
                 ('loss' not in key.lower() and value > self.best_metrics[key]['value']):
                self.best_metrics[key] = {'value': value, 'step': step}
    
    def get_latest_metrics(self) -> Dict[str, float]:
        """Get the latest metrics."""
        return self.metrics_history[-1] if self.metrics_history else {}
    
    def get_best_metrics(self) -> Dict[str, Dict[str, float]]:
        """Get the best metrics achieved so far."""
        return self.best_metrics
    
    def save_metrics(self, save_path: str) -> None:
        """Save metrics history to file."""
        save_json_data(self.metrics_history, save_path)
        logger.info(f"Saved metrics history to {save_path}")
    
    def load_metrics(self, load_path: str) -> None:
        """Load metrics history from file."""
        self.metrics_history = load_json_data(load_path)
        
        # Rebuild best metrics
        self.best_metrics = {}
        for metrics in self.metrics_history:
            step = metrics.get('step', 0)
            for key, value in metrics.items():
                if key == 'step':
                    continue
                
                if key not in self.best_metrics:
                    self.best_metrics[key] = {'value': value, 'step': step}
                elif ('loss' in key.lower() and value < self.best_metrics[key]['value']) or \
                     ('loss' not in key.lower() and value > self.best_metrics[key]['value']):
                    self.best_metrics[key] = {'value': value, 'step': step}
        
        # Update current step
        if self.metrics_history:
            self.current_step = max(m.get('step', 0) for m in self.metrics_history) + 1
        
        logger.info(f"Loaded metrics history from {load_path}")
