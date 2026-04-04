from pathlib import Path
import argparse
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances


def load_image(path, image_size=None):
    img = Image.open(path).convert("RGB")
    if image_size is not None:
        img = img.resize(image_size)
    return np.array(img)


def get_neighbors(embeddings, query_idx, top_k=5, metric="cosine"):
    query = embeddings[query_idx:query_idx + 1]

    if metric == "cosine":
        sims = cosine_similarity(query, embeddings)[0]
        order = np.argsort(-sims)
        scores = sims
    elif metric == "euclidean":
        dists = euclidean_distances(query, embeddings)[0]
        order = np.argsort(dists)
        scores = -dists
    else:
        raise ValueError(f"Unsupported metric: {metric}")

    order = [idx for idx in order if idx != query_idx]
    return order[:top_k], scores


def run(args):
    feature_dir = Path(args.feature_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    embeddings = np.load(feature_dir / "embeddings.npy")
    index_df = pd.read_csv(feature_dir / "embedding_index.csv")

    if len(index_df) != len(embeddings):
        raise ValueError("Mismatch between embedding_index.csv and embeddings.npy")

    if args.query_idx is not None:
        query_idx = args.query_idx
    else:
        random.seed(args.seed)
        query_idx = random.randint(0, len(index_df) - 1)

    neighbor_indices, scores = get_neighbors(
        embeddings=embeddings,
        query_idx=query_idx,
        top_k=args.top_k,
        metric=args.metric
    )

    query_path = index_df.iloc[query_idx]["img_path"]
    query_img = load_image(query_path, image_size=(args.thumb_w, args.thumb_h))

    fig, axes = plt.subplots(1, args.top_k + 1, figsize=(3 * (args.top_k + 1), 4))

    axes[0].imshow(query_img)
    axes[0].set_title(f"Query\nidx={query_idx}")
    axes[0].axis("off")

    rows = [{
        "rank": 0,
        "idx": query_idx,
        "img_path": query_path,
        "score": 1.0 if args.metric == "cosine" else 0.0
    }]

    for rank, nn_idx in enumerate(neighbor_indices, start=1):
        nn_path = index_df.iloc[nn_idx]["img_path"]
        nn_img = load_image(nn_path, image_size=(args.thumb_w, args.thumb_h))

        axes[rank].imshow(nn_img)
        axes[rank].set_title(f"Top-{rank}\nidx={nn_idx}\nscore={scores[nn_idx]:.4f}")
        axes[rank].axis("off")

        rows.append({
            "rank": rank,
            "idx": int(nn_idx),
            "img_path": nn_path,
            "score": float(scores[nn_idx])
        })

    plt.tight_layout()

    img_save = out_dir / f"nn_query_{query_idx:05d}_{args.metric}.png"
    csv_save = out_dir / f"nn_query_{query_idx:05d}_{args.metric}.csv"

    plt.savefig(img_save, dpi=160, bbox_inches="tight")
    plt.close(fig)

    pd.DataFrame(rows).to_csv(csv_save, index=False)

    print(f"[ok] saved panel: {img_save}")
    print(f"[ok] saved neighbors: {csv_save}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature-dir", type=str, default="outputs/features/autoencoder")
    parser.add_argument("--output-dir", type=str, default="outputs/figures/neighbors")
    parser.add_argument("--query-idx", type=int, default=None)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--metric", type=str, default="cosine", choices=["cosine", "euclidean"])
    parser.add_argument("--thumb-w", type=int, default=256)
    parser.add_argument("--thumb-h", type=int, default=160)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    run(args)