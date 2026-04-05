from pathlib import Path
import argparse
import pandas as pd


def normalize_series(s):
    s = s.astype(float)
    if s.max() == s.min():
        return s * 0.0
    return (s - s.min()) / (s.max() - s.min())


def main(args):
    input_csv = Path(args.input_csv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_csv)

    required_cols = ["image_path", "mean_confidence", "mean_entropy", "embedding_cluster"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    df["entropy_norm"] = normalize_series(df["mean_entropy"])
    df["confidence_norm"] = normalize_series(df["mean_confidence"])
    df["uncertainty_score"] = args.w_entropy * df["entropy_norm"] + args.w_lowconf * (1.0 - df["confidence_norm"])

    df = df.sort_values("uncertainty_score", ascending=False).reset_index(drop=True)

    selected_rows = []
    used_paths = set()

    cluster_groups = {
        cluster: subdf.sort_values("uncertainty_score", ascending=False).reset_index(drop=True)
        for cluster, subdf in df.groupby("embedding_cluster")
    }

    # round-robin selection across clusters for diversity
    while len(selected_rows) < args.budget:
        progress = False
        for cluster, subdf in cluster_groups.items():
            for _, row in subdf.iterrows():
                path = row["image_path"]
                if path not in used_paths:
                    selected_rows.append(row.to_dict())
                    used_paths.add(path)
                    progress = True
                    break
            if len(selected_rows) >= args.budget:
                break
        if not progress:
            break

    selected_df = pd.DataFrame(selected_rows)

    # fallback if budget not reached
    if len(selected_df) < args.budget:
        remaining = df[~df["image_path"].isin(used_paths)].head(args.budget - len(selected_df))
        selected_df = pd.concat([selected_df, remaining], ignore_index=True)

    selected_df = selected_df.reset_index(drop=True)
    selected_df["selection_rank"] = range(1, len(selected_df) + 1)

    cluster_summary = (
        selected_df.groupby("embedding_cluster")
        .agg(
            selected_count=("image_path", "count"),
            avg_uncertainty=("uncertainty_score", "mean"),
            avg_entropy=("mean_entropy", "mean"),
            avg_confidence=("mean_confidence", "mean"),
        )
        .reset_index()
        .sort_values("selected_count", ascending=False)
    )

    selected_df.to_csv(output_dir / "annotation_candidates.csv", index=False)
    cluster_summary.to_csv(output_dir / "annotation_candidates_cluster_summary.csv", index=False)

    print(f"[ok] saved {output_dir / 'annotation_candidates.csv'}")
    print(f"[ok] saved {output_dir / 'annotation_candidates_cluster_summary.csv'}")
    print(f"[ok] selected {len(selected_df)} frames")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-csv", type=str, default="outputs/reports/robot_analysis/robot_prediction_analysis.csv")
    parser.add_argument("--output-dir", type=str, default="outputs/reports/annotation_selection")
    parser.add_argument("--budget", type=int, default=50)
    parser.add_argument("--w-entropy", type=float, default=0.7)
    parser.add_argument("--w-lowconf", type=float, default=0.3)
    args = parser.parse_args()
    main(args)