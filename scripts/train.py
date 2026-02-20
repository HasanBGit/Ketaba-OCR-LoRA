#!/usr/bin/env python3
"""
Training Script for Ketaba-OCR
Run QLoRA fine-tuning of Arabic HTR model.

Usage:
    python scripts/train.py
    python scripts/train.py --config configs/training_config.yaml
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.training import train_model, TrainingConfig


def main():
    parser = argparse.ArgumentParser(description='Train Ketaba-OCR model')
    parser.add_argument(
        '--config', '-c', type=str,
        help='Path to YAML config file'
    )
    parser.add_argument('--model', type=str, help='Base model ID')
    parser.add_argument('--output-dir', type=str, help='Output directory')
    parser.add_argument('--train-data', type=str, help='Training data path')
    parser.add_argument('--epochs', type=int, help='Number of epochs')
    parser.add_argument('--lr', type=float, help='Learning rate')
    parser.add_argument('--batch-size', type=int, help='Batch size')
    parser.add_argument('--lora-r', type=int, help='LoRA rank')
    parser.add_argument('--lora-alpha', type=int, help='LoRA alpha')

    args = parser.parse_args()

    # Create config
    config = TrainingConfig()

    # Override with command line args
    if args.model:
        config.model_id = args.model
    if args.output_dir:
        config.output_dir = args.output_dir
    if args.train_data:
        config.train_dataset_path = args.train_data
    if args.epochs:
        config.num_epochs = args.epochs
    if args.lr:
        config.learning_rate = args.lr
    if args.batch_size:
        config.batch_size = args.batch_size
    if args.lora_r:
        config.lora_r = args.lora_r
    if args.lora_alpha:
        config.lora_alpha = args.lora_alpha

    print("=" * 60)
    print("KETABA-OCR TRAINING")
    print("=" * 60)
    print(f"\nConfiguration:")
    for key, value in config.to_dict().items():
        print(f"  {key}: {value}")

    # Run training
    train_model(config)


if __name__ == "__main__":
    main()
