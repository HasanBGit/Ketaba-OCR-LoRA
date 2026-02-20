"""
Run Ensemble with Configurable Weights
Generate ensemble predictions using different weighting strategies.
"""

import argparse
import zipfile
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict

import pandas as pd

from .weight_configs import WEIGHT_CONFIGS, CODABENCH_CER, get_config, print_all_configs
from .advanced_ensemble import advanced_ensemble


def load_submissions(
    submissions_dir: Path,
    weights: Dict[str, float]
) -> Dict[str, Dict]:
    """
    Load submission CSV files with their weights.

    Args:
        submissions_dir: Directory containing submission CSVs
        weights: Dict mapping filename to weight

    Returns:
        Dict mapping filename to {'data': {...}, 'weight': float}
    """
    submissions = {}

    for fname, weight in weights.items():
        if weight < 0.001:  # Skip zero-weight models
            continue

        path = submissions_dir / fname
        if path.exists():
            df = pd.read_csv(path)
            df['image'] = df['image'].astype(str).str.strip()
            df['text'] = df['text'].fillna('').astype(str).str.strip()
            submissions[fname] = {
                'data': df.set_index('image')['text'].to_dict(),
                'weight': weight
            }
            print(f"  Loaded: {fname} ({len(df)} rows, weight={weight:.4f})")
        else:
            print(f"  Warning: {fname} not found")

    return submissions


def run_ensemble(
    config_num: int = 18,
    submissions_dir: Optional[Path] = None,
    output_dir: Optional[Path] = None,
    save_output: bool = True,
) -> pd.DataFrame:
    """
    Run ensemble with specified weight configuration.

    Args:
        config_num: Weight configuration number (1-30)
        submissions_dir: Directory containing submission CSVs
        output_dir: Directory for output files
        save_output: Whether to save CSV and ZIP files

    Returns:
        DataFrame with ensemble results
    """
    # Get configuration
    config_name, weights = get_config(config_num)

    print("=" * 70)
    print(f"ENSEMBLE - Config [{config_num}]: {config_name}")
    print("=" * 70)

    # Setup paths
    if submissions_dir is None:
        submissions_dir = Path.cwd() / "submissions"
    if output_dir is None:
        output_dir = submissions_dir

    submissions_dir = Path(submissions_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Print weights
    print("\nModel Weights:")
    print("-" * 50)
    for fname, weight in sorted(weights.items(), key=lambda x: -x[1]):
        if weight > 0.001:
            name = fname.replace('annotations_', '').replace('.csv', '')
            cer = CODABENCH_CER.get(fname, 0)
            print(f"  {name:35} CER={cer:.2f} → weight={weight:.4f}")

    # Load submissions
    print("\nLoading submissions...")
    submissions = load_submissions(submissions_dir, weights)

    if not submissions:
        raise ValueError("No submissions loaded! Check submissions_dir path.")

    # Get all images
    all_images = set()
    for sub in submissions.values():
        all_images.update(sub['data'].keys())
    all_images = sorted(all_images)
    print(f"\nTotal images: {len(all_images)}")
    print(f"Active models: {len(submissions)}")

    # Run ensemble
    print("\nRunning ensemble...")
    results = []
    for i, image in enumerate(all_images):
        ensemble_text = advanced_ensemble(image, submissions)
        results.append({'image': image, 'text': ensemble_text})

        if (i + 1) % 500 == 0:
            print(f"  Processed {i+1}/{len(all_images)} images...")

    final_df = pd.DataFrame(results)
    print(f"\nEnsemble complete: {len(final_df)} rows")

    # Save outputs
    if save_output:
        print("\n" + "=" * 70)
        print("SAVING FILES")
        print("=" * 70)

        # Save CSV
        csv_name = f"annotations_ensemble_config{config_num}.csv"
        csv_path = output_dir / csv_name
        final_df.to_csv(csv_path, index=False, encoding='utf-8')
        print(f"CSV: {csv_path}")

        # Create ZIP for submission
        ts = datetime.now().strftime("%Y%m%d")
        zip_name = f"submission_ensemble_config{config_num}_{ts}.zip"
        zip_path = output_dir / zip_name
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
            zf.write(csv_path, "annotations.csv")
        print(f"ZIP: {zip_path}")

        print(f"\n>>> Submit: {zip_path}")

    return final_df


def run_all_configs(
    submissions_dir: Optional[Path] = None,
    output_dir: Optional[Path] = None,
) -> Dict[int, pd.DataFrame]:
    """
    Run all 30 weight configurations.

    Args:
        submissions_dir: Directory containing submission CSVs
        output_dir: Directory for output files

    Returns:
        Dict mapping config number to results DataFrame
    """
    print("=" * 70)
    print("RUNNING ALL 30 WEIGHT CONFIGURATIONS")
    print("=" * 70)

    results = {}
    for config_num in range(1, 31):
        print(f"\n{'#' * 70}")
        print(f"# CONFIG {config_num}")
        print(f"{'#' * 70}")

        try:
            df = run_ensemble(
                config_num=config_num,
                submissions_dir=submissions_dir,
                output_dir=output_dir,
                save_output=True,
            )
            results[config_num] = df
        except Exception as e:
            print(f"Error in config {config_num}: {e}")
            results[config_num] = None

    print("\n" + "=" * 70)
    print("ALL CONFIGS COMPLETE!")
    print("=" * 70)

    # Summary
    print("\nGenerated files:")
    for i in range(1, 31):
        status = "OK" if results.get(i) is not None else "FAILED"
        print(f"  [{i:2d}] submission_ensemble_config{i}_*.zip - {status}")

    return results


def main():
    """Command line interface."""
    parser = argparse.ArgumentParser(
        description='Run ensemble with configurable weights'
    )
    parser.add_argument(
        '--config', '-c', type=int, default=18,
        help='Weight configuration number (1-30), or 0 for all configs'
    )
    parser.add_argument(
        '--list', '-l', action='store_true',
        help='List all weight configurations'
    )
    parser.add_argument(
        '--submissions-dir', '-s', type=str,
        help='Directory containing submission CSVs'
    )
    parser.add_argument(
        '--output-dir', '-o', type=str,
        help='Output directory'
    )
    parser.add_argument(
        '--no-save', action='store_true',
        help='Do not save output files'
    )

    args = parser.parse_args()

    if args.list:
        print_all_configs()
        return

    submissions_dir = Path(args.submissions_dir) if args.submissions_dir else None
    output_dir = Path(args.output_dir) if args.output_dir else None

    if args.config == 0:
        run_all_configs(submissions_dir, output_dir)
    else:
        run_ensemble(
            config_num=args.config,
            submissions_dir=submissions_dir,
            output_dir=output_dir,
            save_output=not args.no_save,
        )


if __name__ == "__main__":
    main()
