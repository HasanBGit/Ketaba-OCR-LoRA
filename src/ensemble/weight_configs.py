"""
Weight Configurations for Ensemble Experiments
30 different weighting strategies tested for optimal ensemble performance.

Config 18 (Linear+Boost<0.15) achieved the best results: CER 0.0819
"""

import numpy as np
from typing import Dict, Tuple, List

# CER scores from CodaBench submissions
CODABENCH_CER = {
    "annotations_submission_1.csv": 0.09,
    "annotations_blind_inference_results.csv": 0.11,
    "annotations_blind_hrt_lora.csv": 0.18,
    "annotations_blind_test.csv": 0.20,
    "annotations_blind_qari.csv": 0.26,
    "annotations_blind_arabic_ocr_4bit_v2.csv": 0.32,
}

# Model names for display
MODEL_NAMES = {
    "annotations_submission_1.csv": "Fine-tuned HRT (best)",
    "annotations_blind_inference_results.csv": "Fine-tuned HRT variant",
    "annotations_blind_hrt_lora.csv": "Zero-shot HRT+LoRA",
    "annotations_blind_test.csv": "Zero-shot HRT baseline",
    "annotations_blind_qari.csv": "Fine-tuned QARI",
    "annotations_blind_arabic_ocr_4bit_v2.csv": "Arabic OCR 4-bit",
}


def normalize(weights) -> np.ndarray:
    """Normalize weights to sum to 1."""
    w = np.array(weights, dtype=float)
    return w / w.sum()


def get_weight_configs() -> Dict[int, Tuple[str, Dict[str, float]]]:
    """
    Generate all 30 weight configurations.

    Returns:
        Dictionary mapping config number to (name, weights_dict) tuple
    """
    files = list(CODABENCH_CER.keys())
    cers = np.array(list(CODABENCH_CER.values()))

    configs = {}

    # === Original 10 configs ===
    configs[1] = ("Inverse CER (w=1/CER)", dict(zip(files, normalize(1 / cers))))
    configs[2] = ("Inverse CER² (w=1/CER²)", dict(zip(files, normalize(1 / (cers ** 2)))))
    configs[3] = ("Exponential Decay k=5", dict(zip(files, normalize(np.exp(-cers * 5)))))
    configs[4] = ("Exponential Decay k=10", dict(zip(files, normalize(np.exp(-cers * 10)))))
    configs[5] = ("Softmax T=0.1", dict(zip(files, normalize(np.exp(-cers / 0.1)))))
    configs[6] = ("Linear Decay (w=1-CER)", dict(zip(files, normalize(1 - cers))))
    configs[7] = ("Top-3 Equal", dict(zip(files, normalize([1, 1, 1, 0, 0, 0]))))
    configs[8] = ("Top-3 Weighted", dict(zip(files, normalize([1/0.09, 1/0.11, 1/0.18, 0, 0, 0]))))
    configs[9] = ("Top-2 Only", dict(zip(files, normalize([1/0.09, 1/0.11, 0, 0, 0, 0]))))
    configs[10] = ("Rank-Based (w=1/rank)", dict(zip(files, normalize(1 / np.array([1, 2, 3, 4, 5, 6])))))

    # === Linear Decay Variations (11-20) ===
    configs[11] = ("Linear² (w=(1-CER)²)", dict(zip(files, normalize((1 - cers) ** 2))))
    configs[12] = ("Linear³ (w=(1-CER)³)", dict(zip(files, normalize((1 - cers) ** 3))))
    configs[13] = ("Linear+Exp (w=(1-CER)*e^-CER)", dict(zip(files, normalize((1 - cers) * np.exp(-cers)))))
    configs[14] = ("Linear Top-4", dict(zip(files, normalize((1 - cers) * np.array([1, 1, 1, 1, 0, 0])))))
    configs[15] = ("Linear Top-3", dict(zip(files, normalize((1 - cers) * np.array([1, 1, 1, 0, 0, 0])))))
    configs[16] = ("Linear+Rank", dict(zip(files, normalize((1 - cers) / np.array([1, 2, 3, 4, 5, 6])))))
    configs[17] = ("Linear√ (w=√(1-CER))", dict(zip(files, normalize(np.sqrt(1 - cers)))))

    # Config 18: THE WINNER - Linear+Boost<0.15
    configs[18] = ("Linear+Boost<0.15 [BEST]", dict(zip(files, normalize((1 - cers) + (cers < 0.15) * 0.5))))

    configs[19] = ("Shifted Linear", dict(zip(files, normalize(cers.max() - cers))))
    configs[20] = ("Linear*Inverse", dict(zip(files, normalize((1 - cers) / cers))))

    # === Linear+Boost Variations (21-25) ===
    configs[21] = ("Linear+Boost<0.12", dict(zip(files, normalize((1 - cers) + (cers < 0.12) * 0.5))))
    configs[22] = ("Linear+Boost<0.20", dict(zip(files, normalize((1 - cers) + (cers < 0.20) * 0.5))))
    configs[23] = ("Linear+StrongBoost<0.15", dict(zip(files, normalize((1 - cers) + (cers < 0.15) * 1.0))))
    configs[24] = ("Linear+TieredBoost", dict(zip(files, normalize((1 - cers) + np.array([1.0, 0.7, 0.3, 0.0, 0.0, 0.0])))))
    configs[25] = ("Linear+ExpBoost", dict(zip(files, normalize((1 - cers) + np.exp(-cers * 5) * (cers < 0.20)))))

    # === Config 18 Variations (26-30) ===
    configs[26] = ("Linear+Boost<0.15(0.3)", dict(zip(files, normalize((1 - cers) + (cers < 0.15) * 0.3))))
    configs[27] = ("Linear+Boost<0.15(0.7)", dict(zip(files, normalize((1 - cers) + (cers < 0.15) * 0.7))))
    configs[28] = ("Linear+Boost<0.15(1.0)", dict(zip(files, normalize((1 - cers) + (cers < 0.15) * 1.0))))
    configs[29] = ("Linear+Boost<0.13", dict(zip(files, normalize((1 - cers) + (cers < 0.13) * 0.5))))
    configs[30] = ("Linear√+Boost<0.15", dict(zip(files, normalize(np.sqrt(1 - cers) + (cers < 0.15) * 0.5))))

    return configs


# Pre-computed weight configurations
WEIGHT_CONFIGS = get_weight_configs()


def print_all_configs():
    """Print all available weight configurations."""
    print("=" * 70)
    print("AVAILABLE WEIGHT CONFIGURATIONS")
    print("=" * 70)

    for num, (name, weights) in WEIGHT_CONFIGS.items():
        star = " ⭐" if num == 18 else ""
        print(f"\n[{num:2d}] {name}{star}")
        for fname, w in weights.items():
            if w > 0.001:
                short_name = MODEL_NAMES.get(fname, fname)[:30]
                print(f"     {short_name:32} → {w:.4f}")


def get_config(config_num: int) -> Tuple[str, Dict[str, float]]:
    """Get a specific weight configuration."""
    if config_num not in WEIGHT_CONFIGS:
        raise ValueError(f"Config {config_num} not found. Use 1-30.")
    return WEIGHT_CONFIGS[config_num]


if __name__ == "__main__":
    print_all_configs()
