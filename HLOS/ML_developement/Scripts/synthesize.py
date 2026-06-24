import argparse
import json
from pathlib import Path
import pandas as pd
import numpy as np
import random

from sdv.metadata import SingleTableMetadata
from sdv.single_table import CTGANSynthesizer, TVAESynthesizer
from sdmetrics.reports.single_table import QualityReport

try:
    from sdmetrics.reports.single_table import PrivacyReport
    _HAS_PRIVACY = True
except ImportError:
    _HAS_PRIVACY = False


def read_csv_with_fallback(path: Path):
    encodings = ["utf-8", "utf-8-sig", "cp1252", "latin1"]
    for enc in encodings:
        try:
            df = pd.read_csv(path, low_memory=False, encoding=enc)
            return df, enc
        except Exception:
            continue
    raise RuntimeError(f"Failed to read CSV: {path}")


def drop_high_card_text(df: pd.DataFrame, threshold_ratio: float = 0.9, min_len: int = 80):
    """Drop columns that are almost unique or long free-text."""
    drops = []
    n = len(df)
    for col in df.columns:
        s = df[col]
        if s.dtype == "object":
            nunq = s.nunique(dropna=True)
            if n > 0 and (nunq / n) >= threshold_ratio:
                drops.append(col)
                continue
            try:
                avg_len = s.dropna().astype(str).str.len().mean()
                if avg_len and avg_len >= min_len:
                    drops.append(col)
            except Exception:
                pass
    return drops


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic data using CTGAN or TVAE")
    parser.add_argument("--input", required=True, help="Path to real CSV")
    parser.add_argument("--output_csv", default="synthetic_data.csv")
    parser.add_argument("--synthesizer", choices=["ctgan", "tvae"], default="ctgan",
                        help="Which synthesizer to use")
    parser.add_argument("--rows", type=int, default=0, help="Number of synthetic rows (0 = same as real)")
    parser.add_argument("--epochs", type=int, default=400, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=500)
    parser.add_argument("--random_seed", type=int, default=42)
    parser.add_argument("--drop_text_high_card", action="store_true")
    parser.add_argument("--high_card_ratio", type=float, default=0.9)
    parser.add_argument("--avg_text_len", type=int, default=80)
    parser.add_argument("--exclude_columns", nargs="*", default=None)
    parser.add_argument("--limit_rows_real", type=int, default=0, help="Limit real data for faster testing")
    parser.add_argument("--skip_quality", action="store_true")
    args = parser.parse_args()

    # Reproducibility
    if args.random_seed is not None:
        random.seed(args.random_seed)
        np.random.seed(args.random_seed)

    # Load data
    input_path = Path(args.input)
    df, enc = read_csv_with_fallback(input_path)
    print(f"Loaded {len(df)} rows with encoding: {enc}")

    # Preprocessing
    excluded = args.exclude_columns or []
    if args.drop_text_high_card:
        drops = drop_high_card_text(df, args.high_card_ratio, args.avg_text_len)
        excluded.extend([c for c in drops if c not in excluded])

    if excluded:
        df_synth = df[[c for c in df.columns if c not in excluded]].copy()
        print(f"Excluded {len(excluded)} columns. Keeping {df_synth.shape[1]} columns.")
    else:
        df_synth = df.copy()

    if args.limit_rows_real > 0:
        df_synth = df_synth.head(args.limit_rows_real).copy()
        print(f"Limited real data to {len(df_synth)} rows for testing.")

    # Create Metadata
    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(df_synth)
    print("Metadata auto-detected successfully.")

    # Choose synthesizer
    if args.synthesizer == "ctgan":
        print(f"Training CTGAN for {args.epochs} epochs...")
        synthesizer = CTGANSynthesizer(
            metadata,
            enforce_min_max_values=True,
            enforce_rounding=False,
            epochs=args.epochs,
            batch_size=args.batch_size,
            verbose=True
        )
    else:  # tvae
        print(f"Training TVAE for {args.epochs} epochs...")
        synthesizer = TVAESynthesizer(
            metadata,
            enforce_min_max_values=True,
            enforce_rounding=False,
            epochs=args.epochs,
            verbose=True
        )

    # Train
    synthesizer.fit(df_synth)

    # Generate synthetic data
    n_rows = args.rows if args.rows > 0 else len(df_synth)
    print(f"Generating {n_rows} synthetic rows...")
    synthetic = synthesizer.sample(num_rows=n_rows)

    # Save synthetic data
    output_path = output_dir / args.output_csv
    synthetic.to_csv(output_path, index=False)
    print(f"Synthetic data saved to: {output_path}")

    # Quality Report
    if not args.skip_quality:
        print("Generating Quality Report...")
        qrep = QualityReport()
        qrep.generate(real_data=df_synth, synthetic_data=synthetic, metadata=metadata.to_dict())
        
        quality = {
            "overall_score": qrep.get_score(),
            "properties": qrep.get_properties().reset_index().to_dict(orient="records")
        }
        quality_path = output_dir/"quality_report.json"
        quality_path.write_text(json.dumps(quality, indent=2), encoding="utf-8")

        # Privacy Report
        if _HAS_PRIVACY:
            prep = PrivacyReport()
            prep.generate(real_data=df_synth, synthetic_data=synthetic, metadata=metadata.to_dict())
            privacy = {"overall_score": prep.get_score(), "details": prep.get_details()}
            privacy_path = output_dir/"privacy_report.json"
            privacy_path.write_text(json.dumps(privacy, indent=2), encoding="utf-8")
            print(f"Quality Score: {quality['overall_score']:.3f} | Privacy Score: {privacy['overall_score']:.3f}")
        else:
            print(f"Quality Score: {quality['overall_score']:.3f} (Privacy report not available)")

    print("Done!")

output_dir = Path("/home/kyomukama/ACE/camonet/ML_Rerun/Outputs/Synthetic_data")
output_dir.mkdir(parents=True, exist_ok=True)

if __name__ == "__main__":
    main()