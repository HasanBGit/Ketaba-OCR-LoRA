"""
Inference Module for Ketaba-OCR
Load model and transcribe Arabic manuscript images.
"""

import torch
from pathlib import Path
from typing import Optional, List, Union
from PIL import Image

from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration, BitsAndBytesConfig
from peft import PeftModel
from qwen_vl_utils import process_vision_info


# Default Arabic OCR prompt
DEFAULT_PROMPT = "اقرأ النص الموجود في الصورة:"

# Full prompt used during training
FULL_PROMPT = "ارجو استخراج النص العربي كاملاً من هذه الصورة من البداية الى النهاية بدون اي اختصار ودون ذيادة او حذف. اقرأ كل المحتوى النصي الموجود في الصورة:"


def load_model(
    adapter_path: str = "HassanB4/Ketab-OCR-LoRA",
    base_model: str = "sherif1313/Arabic-English-handwritten-OCR-v3",
    load_in_4bit: bool = True,
    device_map: str = "auto",
):
    """
    Load the Ketaba-OCR model with LoRA adapter.

    Args:
        adapter_path: Path to LoRA adapter (local or HuggingFace Hub)
        base_model: Base model ID
        load_in_4bit: Whether to use 4-bit quantization
        device_map: Device mapping strategy

    Returns:
        Tuple of (model, processor)
    """
    print(f"Loading base model: {base_model}")

    # Load processor
    processor = AutoProcessor.from_pretrained(
        base_model,
        trust_remote_code=True
    )

    # Model kwargs
    model_kwargs = {
        "device_map": device_map,
        "trust_remote_code": True,
        "low_cpu_mem_usage": True,
    }

    # Configure quantization if requested
    if load_in_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        model_kwargs["quantization_config"] = bnb_config
    else:
        model_kwargs["torch_dtype"] = torch.bfloat16

    # Flash Attention
    try:
        import flash_attn
        model_kwargs["attn_implementation"] = "flash_attention_2"
        print("Flash Attention 2 enabled")
    except ImportError:
        model_kwargs["attn_implementation"] = "sdpa"
        print("Using SDPA")

    # Load base model
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        base_model,
        **model_kwargs
    )

    # Load LoRA adapter
    print(f"Loading LoRA adapter: {adapter_path}")
    model = PeftModel.from_pretrained(model, adapter_path)

    # Apply weight tying fix (CRITICAL)
    print("Applying weight tying fix...")
    try:
        model.lm_head.weight = model.model.language_model.embed_tokens.weight
    except AttributeError:
        model.lm_head.weight = model.model.embed_tokens.weight

    model.eval()
    print("Model loaded successfully!")

    return model, processor


def transcribe_image(
    image: Union[str, Path, Image.Image],
    model,
    processor,
    prompt: str = DEFAULT_PROMPT,
    max_new_tokens: int = 512,
    do_sample: bool = False,
    repetition_penalty: float = 1.1,
) -> str:
    """
    Transcribe a single image.

    Args:
        image: Image path or PIL Image
        model: Loaded model
        processor: Loaded processor
        prompt: OCR prompt (Arabic)
        max_new_tokens: Maximum tokens to generate
        do_sample: Whether to use sampling
        repetition_penalty: Repetition penalty

    Returns:
        Transcribed text
    """
    # Load image if path provided
    if isinstance(image, (str, Path)):
        image = Image.open(image).convert("RGB")

    # Create message format
    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": prompt}
        ]
    }]

    # Process inputs
    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    image_inputs, _ = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        return_tensors="pt"
    )
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    # Generate
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            repetition_penalty=repetition_penalty,
            pad_token_id=processor.tokenizer.eos_token_id,
        )

    # Decode (skip input tokens)
    input_len = inputs['input_ids'].shape[1]
    transcription = processor.decode(
        output_ids[0][input_len:],
        skip_special_tokens=True
    ).strip()

    return transcription


def transcribe_batch(
    images: List[Union[str, Path, Image.Image]],
    model,
    processor,
    prompt: str = DEFAULT_PROMPT,
    max_new_tokens: int = 512,
    show_progress: bool = True,
) -> List[str]:
    """
    Transcribe a batch of images.

    Args:
        images: List of image paths or PIL Images
        model: Loaded model
        processor: Loaded processor
        prompt: OCR prompt (Arabic)
        max_new_tokens: Maximum tokens to generate
        show_progress: Whether to show progress bar

    Returns:
        List of transcribed texts
    """
    from tqdm.auto import tqdm

    results = []
    iterator = tqdm(images, desc="Transcribing") if show_progress else images

    for image in iterator:
        try:
            text = transcribe_image(
                image=image,
                model=model,
                processor=processor,
                prompt=prompt,
                max_new_tokens=max_new_tokens,
            )
            results.append(text)
        except Exception as e:
            print(f"Error processing image: {e}")
            results.append("")

    return results


def main():
    """Example usage."""
    import argparse

    parser = argparse.ArgumentParser(description='Transcribe Arabic manuscript images')
    parser.add_argument('images', nargs='+', help='Image paths to transcribe')
    parser.add_argument('--adapter', default='HassanB4/Ketab-OCR-LoRA', help='LoRA adapter path')
    parser.add_argument('--no-4bit', action='store_true', help='Disable 4-bit quantization')
    parser.add_argument('--prompt', default=DEFAULT_PROMPT, help='OCR prompt')

    args = parser.parse_args()

    # Load model
    model, processor = load_model(
        adapter_path=args.adapter,
        load_in_4bit=not args.no_4bit,
    )

    # Transcribe
    for image_path in args.images:
        print(f"\n{'='*50}")
        print(f"Image: {image_path}")
        print(f"{'='*50}")

        transcription = transcribe_image(
            image=image_path,
            model=model,
            processor=processor,
            prompt=args.prompt,
        )

        print(f"Transcription:\n{transcription}")


if __name__ == "__main__":
    main()
