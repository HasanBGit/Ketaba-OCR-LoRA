"""
Training Script for Ketaba-OCR
Fine-tune Arabic-English HTR model using QLoRA
"""

import os
import sys
import gc
import json
import glob
import random
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any, List

import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from tqdm.auto import tqdm
from datasets import load_from_disk

from transformers import (
    AutoProcessor,
    Qwen2_5_VLForConditionalGeneration,
    BitsAndBytesConfig,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    default_data_collator,
    TrainerCallback,
)
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
from qwen_vl_utils import process_vision_info

from .config import TrainingConfig


def clear_memory():
    """Clear GPU and CPU memory."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def load_model_and_processor(config: TrainingConfig):
    """
    Load the base model and processor with quantization.

    Args:
        config: Training configuration

    Returns:
        Tuple of (model, processor)
    """
    print(f"Loading model: {config.model_id}")

    # Load processor
    processor = AutoProcessor.from_pretrained(
        config.model_id,
        trust_remote_code=True
    )

    # Configure quantization
    compute_dtype = getattr(torch, config.bnb_4bit_compute_dtype)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=config.load_in_4bit,
        bnb_4bit_quant_type=config.bnb_4bit_quant_type,
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=config.bnb_4bit_use_double_quant,
    )

    # Model kwargs
    model_kwargs = {
        "quantization_config": bnb_config,
        "device_map": "auto",
        "trust_remote_code": True,
        "low_cpu_mem_usage": True,
    }

    # Flash Attention
    try:
        import flash_attn
        model_kwargs["attn_implementation"] = "flash_attention_2"
        print("Flash Attention 2 enabled")
    except ImportError:
        model_kwargs["attn_implementation"] = "sdpa"
        print("Using SDPA (Flash Attention not available)")

    # Load model
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        config.model_id,
        **model_kwargs
    )

    # Apply weight tying fix
    print("Applying weight tying fix...")
    try:
        model.lm_head.weight = model.model.language_model.embed_tokens.weight
        print("Weight tying applied successfully")
    except AttributeError:
        try:
            model.lm_head.weight = model.model.embed_tokens.weight
            print("Weight tying applied (fallback path)")
        except Exception as e:
            print(f"Warning: Could not apply weight tying: {e}")

    print(f"Model loaded! Parameters: {model.num_parameters() / 1e9:.2f}B")

    return model, processor


def apply_lora(model, config: TrainingConfig):
    """
    Apply LoRA adapters to the model.

    Args:
        model: Base model
        config: Training configuration

    Returns:
        Model with LoRA adapters
    """
    print("Applying LoRA/DoRA adapters...")

    # Prepare for k-bit training
    model = prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=True,
    )

    # LoRA configuration
    lora_config = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        target_modules=config.target_modules,
        lora_dropout=config.lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        use_dora=config.use_dora,
        use_rslora=config.use_rslora,
    )

    print(f"LoRA Config: r={config.lora_r}, alpha={config.lora_alpha}, "
          f"DoRA={config.use_dora}, RSLoRA={config.use_rslora}")

    # Apply LoRA
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    model.train()

    return model, lora_config


def preprocess_sample(sample: Dict, processor, max_image_size: int, max_seq_length: int):
    """
    Preprocess a single sample for training.

    Args:
        sample: Dictionary with 'image' and 'text' keys
        processor: Model processor
        max_image_size: Maximum image dimension
        max_seq_length: Maximum sequence length

    Returns:
        Preprocessed sample dictionary or None if failed
    """
    try:
        image = sample["image"]
        text = sample["text"]

        # Resize if too large
        if image.width > max_image_size or image.height > max_image_size:
            image.thumbnail((max_image_size, max_image_size), Image.Resampling.LANCZOS)

        # Ensure dimensions are multiples of 32
        width, height = image.size
        new_width = ((width + 31) // 32) * 32
        new_height = ((height + 31) // 32) * 32
        image = image.resize((new_width, new_height), Image.LANCZOS)

        # Create message format
        prompt = "ارجو استخراج النص العربي كاملاً من هذه الصورة من البداية الى النهاية بدون اي اختصار ودون ذيادة او حذف. اقرأ كل المحتوى النصي الموجود في الصورة:"

        messages_full = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt}
                ]
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": text}]
            }
        ]

        messages_question = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt}
                ]
            }
        ]

        # Apply chat template
        full_text = processor.apply_chat_template(
            messages_full, tokenize=False, add_generation_prompt=False
        )
        question_text = processor.apply_chat_template(
            messages_question, tokenize=False, add_generation_prompt=True
        )

        # Process vision info
        image_inputs, _ = process_vision_info(messages_full)

        # Tokenize
        inputs = processor(
            text=[full_text],
            images=image_inputs,
            padding="longest",
            truncation=True,
            max_length=max_seq_length,
            return_tensors="pt"
        )

        question_inputs = processor(
            text=[question_text],
            images=image_inputs,
            return_tensors="pt"
        )
        question_len = question_inputs.input_ids.shape[1]

        # Create labels (mask the question part)
        labels = inputs.input_ids.clone()
        labels[:, :question_len] = -100

        return {
            "pixel_values": inputs.pixel_values.squeeze(0),
            "image_grid_thw": inputs.image_grid_thw.squeeze(0),
            "input_ids": inputs.input_ids.squeeze(0),
            "attention_mask": inputs.attention_mask.squeeze(0),
            "labels": labels.squeeze(0)
        }

    except Exception as e:
        print(f"Error preprocessing sample: {e}")
        return None


class OCRDataset(Dataset):
    """Dataset class for OCR training."""

    def __init__(self, processed_samples: List[Dict]):
        self.samples = [s for s in processed_samples if s is not None]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


class ProgressCallback(TrainerCallback):
    """Callback for progress tracking."""

    def __init__(self, results_dir: str):
        self.results_dir = Path(results_dir)
        self.loss_history = []

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs and "loss" in logs:
            self.loss_history.append({
                "step": state.global_step,
                "loss": logs["loss"],
                "timestamp": datetime.now().isoformat()
            })
            print(f"Step {state.global_step} - Loss: {logs['loss']:.4f}")

    def on_save(self, args, state, control, **kwargs):
        state_info = {
            "global_step": state.global_step,
            "epoch": state.epoch,
            "loss_history": self.loss_history[-100:],
            "timestamp": datetime.now().isoformat(),
        }
        with open(self.results_dir / "training_state.json", "w") as f:
            json.dump(state_info, f, indent=2)


def train_model(
    config: Optional[TrainingConfig] = None,
    train_samples: Optional[List[Dict]] = None,
    eval_samples: Optional[List[Dict]] = None,
):
    """
    Main training function.

    Args:
        config: Training configuration (uses default if None)
        train_samples: List of training samples with 'image' and 'text' keys
        eval_samples: List of evaluation samples

    Returns:
        Trained model and processor
    """
    if config is None:
        config = TrainingConfig()

    # Create directories
    for dir_path in [config.output_dir, config.results_dir, config.logs_dir]:
        Path(dir_path).mkdir(parents=True, exist_ok=True)

    # Set seed
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)

    # Load model
    clear_memory()
    model, processor = load_model_and_processor(config)

    # Apply LoRA
    model, lora_config = apply_lora(model, config)

    # Save LoRA config
    lora_config.save_pretrained(config.output_dir)

    # Load data if not provided
    if train_samples is None:
        print(f"Loading training data from {config.train_dataset_path}")
        train_dataset = load_from_disk(config.train_dataset_path)

        train_samples = []
        for i in tqdm(range(len(train_dataset)), desc="Converting dataset"):
            sample = train_dataset[i]
            messages = sample['messages']
            user_msg = messages[0]
            images = user_msg.get('images', [])
            image = images[0] if images else None
            assistant_msg = messages[1] if len(messages) > 1 else None
            text = assistant_msg.get('content', '') if assistant_msg else ''

            if image is not None and text:
                train_samples.append({"image": image, "text": text})

    # Shuffle and split
    random.shuffle(train_samples)

    if eval_samples is None:
        split_idx = int(0.9 * len(train_samples))
        eval_samples = train_samples[split_idx:]
        train_samples = train_samples[:split_idx]

    print(f"Train samples: {len(train_samples)}, Eval samples: {len(eval_samples)}")

    # Preprocess
    print("Preprocessing training samples...")
    train_processed = []
    for i, s in enumerate(tqdm(train_samples, desc="Train")):
        result = preprocess_sample(s, processor, config.max_image_size, config.max_seq_length)
        if result is not None:
            train_processed.append(result)
        if (i + 1) % 50 == 0:
            clear_memory()

    print("Preprocessing eval samples...")
    eval_processed = []
    for s in tqdm(eval_samples, desc="Eval"):
        result = preprocess_sample(s, processor, config.max_image_size, config.max_seq_length)
        if result is not None:
            eval_processed.append(result)

    # Create datasets
    train_dataset = OCRDataset(train_processed)
    eval_dataset = OCRDataset(eval_processed)

    # Training arguments
    training_args = Seq2SeqTrainingArguments(
        output_dir=config.output_dir,
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        num_train_epochs=config.num_epochs,
        learning_rate=config.learning_rate,
        lr_scheduler_type=config.lr_scheduler_type,
        warmup_steps=config.warmup_steps,
        weight_decay=config.weight_decay,
        max_grad_norm=config.max_grad_norm,
        optim="adamw_torch_fused" if torch.cuda.is_available() else "adamw_torch",
        bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        fp16=torch.cuda.is_available() and not torch.cuda.is_bf16_supported(),
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=5,
        logging_first_step=True,
        report_to="none",
        save_strategy="steps",
        save_steps=config.save_steps,
        save_total_limit=config.save_total_limit,
        eval_strategy="steps",
        eval_steps=config.eval_steps,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        dataloader_num_workers=config.dataloader_num_workers,
        dataloader_pin_memory=False,
        remove_unused_columns=False,
        seed=config.seed,
    )

    # Create trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=default_data_collator,
        processing_class=processor,
        callbacks=[ProgressCallback(config.results_dir)],
    )

    # Train
    print("\nStarting training...")
    trainer.train()

    # Save final model
    final_path = Path(config.output_dir) / "final_model"
    final_path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(final_path)
    processor.save_pretrained(final_path)

    # Save training info
    training_info = config.to_dict()
    training_info["completed_at"] = datetime.now().isoformat()
    training_info["train_samples"] = len(train_samples)
    training_info["eval_samples"] = len(eval_samples)

    with open(final_path / "training_info.json", "w") as f:
        json.dump(training_info, f, indent=2)

    print(f"\nTraining complete! Model saved to: {final_path}")

    return model, processor


if __name__ == "__main__":
    config = TrainingConfig()
    print("Training Configuration:")
    print(json.dumps(config.to_dict(), indent=2))
    train_model(config)
