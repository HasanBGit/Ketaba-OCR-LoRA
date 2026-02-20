#!/usr/bin/env python3
"""
Inference Script for Ketaba-OCR
Transcribe Arabic manuscript images.

Usage:
    python scripts/inference.py image1.png image2.png
    python scripts/inference.py --adapter HassanB4/Ketab-OCR-LoRA image.png
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.inference import load_model, transcribe_image, transcribe_batch


def main():
    parser = argparse.ArgumentParser(description='Transcribe Arabic manuscript images')
    parser.add_argument('images', nargs='+', help='Image paths to transcribe')
    parser.add_argument(
        '--adapter', '-a',
        default='HassanB4/Ketab-OCR-LoRA',
        help='LoRA adapter path (local or HuggingFace Hub)'
    )
    parser.add_argument(
        '--base-model', '-b',
        default='sherif1313/Arabic-English-handwritten-OCR-v3',
        help='Base model ID'
    )
    parser.add_argument(
        '--no-4bit',
        action='store_true',
        help='Disable 4-bit quantization (uses more VRAM)'
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        help='Output file for results (CSV format)'
    )
    parser.add_argument(
        '--max-tokens',
        type=int,
        default=512,
        help='Maximum tokens to generate'
    )

    args = parser.parse_args()

    print("=" * 60)
    print("KETABA-OCR INFERENCE")
    print("=" * 60)

    # Load model
    print(f"\nLoading model...")
    print(f"  Adapter: {args.adapter}")
    print(f"  Base: {args.base_model}")
    print(f"  4-bit: {not args.no_4bit}")

    model, processor = load_model(
        adapter_path=args.adapter,
        base_model=args.base_model,
        load_in_4bit=not args.no_4bit,
    )

    # Transcribe images
    print(f"\nTranscribing {len(args.images)} images...")

    results = []
    for image_path in args.images:
        print(f"\n{'='*50}")
        print(f"Image: {image_path}")
        print(f"{'='*50}")

        transcription = transcribe_image(
            image=image_path,
            model=model,
            processor=processor,
            max_new_tokens=args.max_tokens,
        )

        print(f"Transcription:\n{transcription}")
        results.append({'image': image_path, 'text': transcription})

    # Save results
    if args.output:
        import pandas as pd
        df = pd.DataFrame(results)
        df.to_csv(args.output, index=False, encoding='utf-8')
        print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()
