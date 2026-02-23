# Ketaba-OCR at NakbaNLP 2026 Shared Task: Efficient Adaptation of Vision-Language Models for Handwritten Text Recognition

<p align="center">
<img src="https://placehold.co/800x200/fef3c7/d97706?text=Arabic-Manuscript-OCR" alt="Arabic Manuscript OCR">
</p>

This repository contains the official models and results for **Ketaba-OCR**, the **1st place winning** submission to the **[NakbaNLP 2026 Shared Task](https://acrps.ai/nakba-nlp-manu-understanding-2026)** on Arabic Manuscript Understanding (Subtask 2: Systems Track).

#### By: [Hassan Barmandah](https://scholar.google.com/citations?user=2VzOr0kAAAAJ&hl=en), [Fatimah Emad Eldin](https://scholar.google.com/citations?user=CfX6eA8AAAAJ&hl=ar), [Khloud Al Jallad](https://scholar.google.com/citations?user=A0EvL6cAAAAJ&hl=ar), [Omar Nacer](https://scholar.google.com/citations?user=pezf5FYAAAAJ&hl=en)

[![Paper](https://img.shields.io/badge/arXiv-26XX.XXXXX-b31b1b.svg)](https://arxiv.org/abs/26XX.XXXXX)
[![Code](https://img.shields.io/badge/GitHub-Code-blue)](https://github.com/HasanBGit/Ketab-OCR-LoRA)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-Page-F9D371)](https://huggingface.co/collections/HassanB4/nakba-nlp-2026-arabic-manuscript-understanding-shared-task)
[![License](https://img.shields.io/badge/License-Apache%202.0-lightgrey)](LICENSE)

---

## Model Description

This project introduces a **parameter-efficient approach** for Arabic handwritten text recognition (HTR) on historical manuscripts. The system is built upon Sherif's pretrained Arabic-English HTR model, which leverages prior training on diverse handwritten datasets including Kitab and IAM. Rather than training from scratch, we fine-tune the HTR backbone using **Low-Rank Adaptation (LoRA)** with 4-bit quantization (QLoRA), along with DoRA and RSLoRA for improved training stability.

A key element of this system is its ensemble strategy using a novel **Linear+Boost weighted voting scheme**. This approach proved to be highly effective, achieving **1st place** on the official leaderboard with a Character Error Rate (CER) of **0.0819** and Word Error Rate (WER) of **0.2588** on the blind test set.

The model transcribes cropped line images from Arabic manuscripts into machine-readable text, specifically optimized for the **Omar Al-Saleh Memoir Collection** (1951-1965) written in Ruq'ah and Naskh script variants.

### Key Contributions

* **Ranking & Performance**: Secured **1st place** on the official leaderboard with CER 0.082 and WER 0.259
* **HTR vs. Generalist VLMs**: Demonstrated that specialized fine-tuned HTR models drastically outperform zero-shot generalist VLMs
* **Parameter Efficiency**: QLoRA efficiently bridged the domain gap, reducing CER from 0.58 to 0.08 with minimal computational overhead (~8GB VRAM)
* **Ensemble Innovation**: Linear+Boost weighting strategy improved CER by 7.4% over standard inverse-CER weighting

---

## 🚀 How to Use

You can use the fine-tuned model directly with the `transformers` and `peft` libraries. The following example demonstrates inference on a manuscript line image.

```python
import torch
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration, BitsAndBytesConfig
from peft import PeftModel
from PIL import Image
from qwen_vl_utils import process_vision_info

# Configure 4-bit quantization
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

# Load base model
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    "sherif1313/Arabic-English-handwritten-OCR-v3",
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
)

# Load LoRA adapter
model = PeftModel.from_pretrained(model, "HassanB4/Ketab-OCR-LoRA")

# Apply weight tying fix (critical for correct output)
model.lm_head.weight = model.model.language_model.embed_tokens.weight

# Load processor
processor = AutoProcessor.from_pretrained(
    "sherif1313/Arabic-English-handwritten-OCR-v3",
    trust_remote_code=True
)

# Example inference
image = Image.open("manuscript_line.png").convert("RGB")

messages = [{
    "role": "user",
    "content": [
        {"type": "image", "image": image},
        {"type": "text", "text": "اقرأ النص الموجود في الصورة:"}
    ]
}]

text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
image_inputs, _ = process_vision_info(messages)
inputs = processor(text=[text], images=image_inputs, return_tensors="pt")
inputs = {k: v.to(model.device) for k, v in inputs.items()}

with torch.no_grad():
    output_ids = model.generate(**inputs, max_new_tokens=512, do_sample=False)

transcription = processor.decode(output_ids[0][len(inputs['input_ids'][0]):], skip_special_tokens=True)
print(transcription)
```

---

## ⚙️ Training Procedure

The system employs QLoRA fine-tuning of a specialized pretrained HTR model, rather than training a general-purpose VLM from scratch.

### Training Data

The model was fine-tuned on the official NakbaNLP 2026 dataset from the Omar Al-Saleh Memoir Collection:

| Split | Samples | Description |
| :--- | :---: | :--- |
| Training | 15,962 | Line images with gold transcriptions |
| Development | 1,774 | Line images for validation |
| Test (Blind) | 2,095 | Held-out for official evaluation |

### Hyperparameters

| Parameter | Value | Parameter | Value |
| :--- | :--- | :--- | :--- |
| **Base Model** | sherif1313/Arabic-English-OCR-v3 | **Architecture** | Vision-Language Transformer |
| **Model Size** | ~4.07B parameters | **Pretraining Data** | Kitab, IAM, Custom |
| **Quantization** | 4-bit NF4 (QLoRA) | **Compute Dtype** | bfloat16 |
| **Double Quant** | True | | |
| **LoRA Rank (r)** | 32 | **LoRA Alpha (α)** | 64 |
| **Target Modules** | q, k, v, o, gate, up, down | **LoRA Dropout** | 0.05 |
| **DoRA** | True | **RSLoRA** | True |
| **Learning Rate** | 2×10⁻⁵ | **Optimizer** | AdamW (fused) |
| **LR Scheduler** | Cosine | **Warmup Steps** | 200 |
| **Batch Size** | 1 (per GPU) | **Gradient Accumulation** | 4 |
| **Effective Batch** | 4 | **Number of Epochs** | 1 |
| **Max Gradient Norm** | 1.0 | **Weight Decay** | 0.01 |
| **Max Sequence Length** | 2048 | **Max Image Size** | 1024 |

### Ensemble Strategy

Our final submission employs a **Linear+Boost** weighted ensemble combining predictions from six model configurations:

```
w_i = normalize((1 - CER_i) + 1[CER_i < 0.15] × 0.5)
```

This applies a linear decay based on CER, plus a bonus of 0.5 for models with CER below 0.15. The ensemble algorithm uses:

1. **Weighted Majority Voting**: Predictions exceeding 50% weighted consensus are selected directly
2. **Arabic Normalization**: For disagreements, normalize alef variants and teh marbuta before voting
3. **N-gram Consistency**: Score predictions by 3-gram overlap with other models
4. **Edit Distance Consensus**: Final tie-breaking uses minimum average edit distance

### Frameworks

* PyTorch 2.5.0
* Hugging Face Transformers ≥4.45.0
* PEFT ≥0.14.0
* bitsandbytes ≥0.43.0
* Flash Attention 2.8.3

---

## 📊 Evaluation Results

The models were evaluated on the blind test set provided by the NakbaNLP 2026 organizers. The primary metric is **Character Error Rate (CER)**, computed as normalized Levenshtein distance.

### Final Test Set Scores

| System | Test CER | Test WER | Blind CER | Blind WER |
| :--- | :---: | :---: | :---: | :---: |
| Organizer Baseline | 0.584 | 0.881 | 0.591 | 0.885 |
| Zero-Shot HRT (Qwen2.5-VL) | 0.169 | 0.499 | 0.203 | 0.503 |
| Fine-Tuned HRT (Single Model) | 0.081 | 0.115 | 0.088 | 0.270 |
| **Ketaba-OCR + Ensemble (Ours)** | — | — | **0.0819** | **0.2588** |

### Comparison with Other Models

| Model | Blind CER | Blind WER |
| :--- | :---: | :---: |
| **Ketaba-OCR (Ours)** | **0.0819** | **0.2588** |
| Fine-Tuned QARI-3 | 0.2635 | 0.5521 |
| Arabic OCR 4-bit (Sherif) | 0.3234 | 0.6203 |
| Qwen2.5-VL-7B (Zero-Shot) | 0.6808 | 0.9198 |
| Qwen2.5-VL-3B (Zero-Shot) | 0.6213 | 0.8628 |

---

## ⚠️ Limitations

* **Domain Specificity**: Optimized for 1950s Ruq'ah/Naskh manuscripts; requires adaptation for other periods/styles
* **Agglutination Gap**: WER (0.26) is disproportionately higher than CER (0.08) due to Arabic's agglutinative structure
* **Degraded Images**: Performance degrades on severely faded or damaged manuscript regions
* **Generalization**: Not tested on other historical Arabic manuscript collections

---

## 🙏 Acknowledgements

We thank the **NakbaNLP 2026 organizers** (Fadi Zaraket, Bilal Shalash, Hadi Hamoud, Ahmad Chamseddine, Firas Ben Abid, Mustafa Jarrar, Chadi Abou Chakra, Bernard Ghanem) for access to the Omar Al-Saleh Memoir Collection. We acknowledge **Sherif** for the pretrained Arabic-English OCR model, and the **Hugging Face** community for PEFT and bitsandbytes libraries.

### Related Links

* [NakbaNLP 2026 Workshop](https://sina.birzeit.edu/nakba-nlp/2026/)
* [AR-MS Shared Task Website](https://acrps.ai/nakba-nlp-manu-understanding-2026)
* [Base Model (Sherif's HTR)](https://huggingface.co/sherif1313/Arabic-English-handwritten-OCR-v3)

---

## 📜 Citation

If you use this work, please cite the paper:

```bibtex
@inproceedings{barmandah2026ketaba,
    title={{Ketaba-OCR at NakbaNLP 2026 Shared Task: Efficient Adaptation of Vision-Language Models for Handwritten Text Recognition}},
    author={Barmandah, Hassan and Eldin, Fatimah Emad and Al Jallad, Khloud and Nacer, Omar},
    year={2026},
    booktitle={Proceedings of the 2nd International Workshop on Nakba Narratives as Language Resources (NakbaNLP 2026)},
    publisher={RASD}
}
```

---

## 📄 License

This project is licensed under the Apache 2.0 License.
