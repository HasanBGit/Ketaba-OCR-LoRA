"""
Training Configuration for Ketaba-OCR
"""

from dataclasses import dataclass, field
from typing import List, Optional
from pathlib import Path


@dataclass
class TrainingConfig:
    """Configuration for QLoRA fine-tuning of Ketaba-OCR."""

    # Model Configuration
    model_id: str = "sherif1313/Arabic-English-handwritten-OCR-v3"
    model_tag: str = "ketaba_ocr_lora"

    # LoRA Configuration
    lora_r: int = 32
    lora_alpha: int = 64
    lora_dropout: float = 0.05
    use_dora: bool = True
    use_rslora: bool = True
    target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ])

    # Quantization
    load_in_4bit: bool = True
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_compute_dtype: str = "bfloat16"
    bnb_4bit_use_double_quant: bool = True

    # Training Hyperparameters
    batch_size: int = 1
    gradient_accumulation_steps: int = 4
    num_epochs: int = 1
    learning_rate: float = 2e-5
    lr_scheduler_type: str = "cosine"
    warmup_steps: int = 200
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0

    # Sequence Settings
    max_seq_length: int = 2048
    max_image_size: int = 1024

    # Chunked Training
    files_per_chunk: int = 1000
    start_chunk: int = 0
    num_chunks: Optional[int] = None

    # Evaluation
    eval_steps: int = 500
    save_steps: int = 500
    save_total_limit: int = 3

    # Paths
    train_dataset_path: Optional[str] = None
    test_data_path: Optional[str] = None
    output_dir: Optional[str] = None
    results_dir: Optional[str] = None
    logs_dir: Optional[str] = None

    # Misc
    seed: int = 42
    dataloader_num_workers: int = 0

    def __post_init__(self):
        """Set default paths based on environment."""
        import os

        is_runpod = os.path.exists("/workspace")

        if is_runpod:
            base_dir = Path("/workspace/project")
        else:
            base_dir = Path.cwd()

        if self.train_dataset_path is None:
            self.train_dataset_path = str(base_dir / "ARABIC_OCR_FINAL")
        if self.test_data_path is None:
            self.test_data_path = str(base_dir / "data" / "test")
        if self.output_dir is None:
            self.output_dir = str(base_dir / "checkpoints" / "hrt_lora")
        if self.results_dir is None:
            self.results_dir = str(base_dir / "results" / "hrt_lora")
        if self.logs_dir is None:
            self.logs_dir = str(base_dir / "logs" / "hrt_lora")

    @property
    def effective_batch_size(self) -> int:
        return self.batch_size * self.gradient_accumulation_steps

    def to_dict(self) -> dict:
        """Convert config to dictionary."""
        return {
            "model_id": self.model_id,
            "model_tag": self.model_tag,
            "lora_r": self.lora_r,
            "lora_alpha": self.lora_alpha,
            "lora_dropout": self.lora_dropout,
            "use_dora": self.use_dora,
            "use_rslora": self.use_rslora,
            "target_modules": self.target_modules,
            "load_in_4bit": self.load_in_4bit,
            "batch_size": self.batch_size,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "effective_batch_size": self.effective_batch_size,
            "num_epochs": self.num_epochs,
            "learning_rate": self.learning_rate,
            "max_seq_length": self.max_seq_length,
            "max_image_size": self.max_image_size,
        }
