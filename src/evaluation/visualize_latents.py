from pathlib import Path
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


def infer_group_from_path(path_str, mode="parent"):
    p = Path(path_str)
    if mode == "parent":
        return p.parent.name
    if mode == "stem_prefix":
        return p.stem.split("_")[0]
    return "unknown"


def project_embeddings(embeddings, method="pca", seed=42):
    if method == "pca":
        projector = PCA(n_components=2, random_state=seed)
        coords = projector.fit_transform(embeddings)
        extra = {"explained_variance_ratio": projector.explained_variance_ratio_.tolist()}
    elif method == "tsne":
        projector = TSNE(
            n_components=2,
            perplexity=30,
            learning_rate="auto",
            init="pca",
            random_state=seed
        )
        coords = projector.fit_transform(embeddings)
        extra = {}
    else:
        raise ValueError(f"Unsupported method: {method}")
    return coords, extra


def run(args):
    feature_dir = Path(args.feature_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    embeddings = np.load(feature_dir / "embeddings.npy")
    index_df = pd.read_csv(feature_dir / "embedding_index.csv")

    if len(index_df) != len(embeddings):
        raise ValueError("embedding_index.csv and embeddings.npy size mismatch")

    if args.group_col in index_df.columns:
        groups = index_df[args.group_col].astype(str).values
    else:
        groups = np.array([infer_group_from_path(p, mode=args.infer_mode) for p in index_df["img_path"]])

    coords, extra = project_embeddings(embeddings, method=args.method, seed=args.seed)

    plot_df = pd.DataFrame({
        "x": coords[:, 0],
        "y": coords[:, 1],
        "group": groups,
        "img_path": index_df["img_path"]
    })

    plt.figure(figsize=(10, 8))
    unique_groups = sorted(plot_df["group"].unique())

    cmap = plt.cm.get_cmap("tab20", len(unique_groups))
    for i, group in enumerate(unique_groups):
        subset = plot_df[plot_df["group"] == group]
        plt.scatter(
            subset["x"],
            subset["y"],
            s=20,
            alpha=0.75,
            color=cmap(i),
            label=group
        )

    title = f"Latent space visualization ({args.method.upper()})"
    if args.method == "pca" and "explained_variance_ratio" in extra:
        evr = extra["explained_variance_ratio"]
        title += f"\nPC1={evr[0]:.3f}, PC2={evr[1]:.3f}"

    plt.title(title)
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.legend(loc="best", fontsize=8)
    plt.tight_layout()

    save_path = out_dir / f"latent_{args.method}.png"
    plt.savefig(save_path, dpi=160, bbox_inches="tight")
    plt.close()

    plot_df.to_csv(out_dir / f"latent_{args.method}.csv", index=False)

    print(f"[ok] saved plot: {save_path}")
    print(f"[ok] saved coordinates: {out_dir / f'latent_{args.method}.csv'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature-dir", type=str, default="outputs/features/autoencoder")
    parser.add_argument("--output-dir", type=str, default="outputs/figures/latents")
    parser.add_argument("--method", type=str, default="pca", choices=["pca", "tsne"])
    parser.add_argument("--group-col", type=str, default="group")
    parser.add_argument("--infer-mode", type=str, default="parent", choices=["parent", "stem_prefix"])
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    run(args)