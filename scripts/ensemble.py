#!/usr/bin/env python3
"""
Ensemble Script for Ketaba-OCR
Run ensemble with configurable weights.

Usage:
    python scripts/ensemble.py --config 18  # Use Linear+Boost (best)
    python scripts/ensemble.py --config 0   # Run all 30 configs
    python scripts/ensemble.py --list       # List all configs
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ensemble import run_ensemble, run_all_configs
from src.ensemble.weight_configs import print_all_configs


def main():
    parser = argparse.ArgumentParser(description='Run ensemble with configurable weights')
    parser.add_argument(
        '--config', '-c',
        type=int,
        default=18,
        help='Weight configuration (1-30), 0 for all. Default: 18 (Linear+Boost)'
    )
    parser.add_argument(
        '--list', '-l',
        action='store_true',
        help='List all weight configurations'
    )
    parser.add_argument(
        '--submissions-dir', '-s',
        type=str,
        help='Directory containing submission CSVs'
    )
    parser.add_argument(
        '--output-dir', '-o',
        type=str,
        help='Output directory'
    )
    parser.add_argument(
        '--no-save',
        action='store_true',
        help='Do not save output files'
    )

    args = parser.parse_args()

    if args.list:
        print_all_configs()
        return

    print("=" * 60)
    print("KETABA-OCR ENSEMBLE")
    print("=" * 60)

    submissions_dir = Path(args.submissions_dir) if args.submissions_dir else None
    output_dir = Path(args.output_dir) if args.output_dir else None

    if args.config == 0:
        print("\nRunning all 30 configurations...")
        run_all_configs(submissions_dir, output_dir)
    else:
        print(f"\nRunning configuration {args.config}...")
        run_ensemble(
            config_num=args.config,
            submissions_dir=submissions_dir,
            output_dir=output_dir,
            save_output=not args.no_save,
        )


if __name__ == "__main__":
    main()
