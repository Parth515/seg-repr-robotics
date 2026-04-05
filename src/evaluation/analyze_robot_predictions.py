# src/evaluation/analyze_robot_predictions.py

from pathlib import Path
import argparse
import json

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.cluster import KMeans
from torch.utils.data import DataLoader
from torchvision.models.segmentation import deeplabv3_resnet50

from src.datasets.image_only_dataset import ImageOnlyDataset
from src.datasets.transforms import get_val_transforms, get_autoencoder_val_transforms
from src.models.representation.autoencoder import ConvAutoencoder


NUM_CLASSES = 19
IGNORE_INDEX = 255


def load_seg_model(checkpoint_path, device):
    model = deeplabv3_resnet50(weights=None, num_classes=NUM_CLASSES, aux_loss=True)
    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


def load_ae_model(checkpoint_path, latent_dim, device):
    model = ConvAutoencoder(latent_dim=latent_dim)
    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


@torch.no_grad()
def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    seg_dataset = ImageOnlyDataset(
        image_dir=args.image_dir,
        transform=get_val_transforms((args.height, args.width)),
        suffix=args.suffix,
    )
    ae_dataset = ImageOnlyDataset(
        image_dir=args.image_dir,
        transform=get_autoencoder_val_transforms((args.ae_height, args.ae_width)),
        suffix=args.suffix,
    )

    seg_loader = DataLoader(
        seg_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    ae_loader = DataLoader(
        ae_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    seg_model = load_seg_model(args.seg_checkpoint, device)
    ae_model = load_ae_model(args.ae_checkpoint, args.latent_dim, device)

    seg_rows = []
    for batch in seg_loader:
        images = batch["image"].to(device)
        paths = batch["img_path"]

        logits = seg_model(images)["out"]
        probs = F.softmax(logits, dim=1)

        max_probs, preds = torch.max(probs, dim=1)
        entropy = -(probs * torch.log(probs.clamp_min(1e-8))).sum(dim=1)

        for i in range(images.shape[0]):
            pred_np = preds[i].cpu().numpy().astype(np.uint8)
            unique_classes = np.unique(pred_np).tolist()

            seg_rows.append({
                "image_path": paths[i],
                "mean_confidence": float(max_probs[i].mean().item()),
                "mean_entropy": float(entropy[i].mean().item()),
                "num_predicted_classes": int(len(unique_classes)),
                "predicted_classes": json.dumps(unique_classes),
            })

    seg_df = pd.DataFrame(seg_rows).sort_values("image_path").reset_index(drop=True)

    emb_rows = []
    emb_list = []
    for batch in ae_loader:
        images = batch["image"].to(device)
        paths = batch["img_path"]

        z = ae_model.encode(images)
        z = torch.mean(z, dim=(2, 3))
        z = z.cpu().numpy().astype(np.float32)

        emb_list.append(z)
        for i in range(len(paths)):
            emb_rows.append({"image_path": paths[i]})

    emb_df = pd.DataFrame(emb_rows).sort_values("image_path").reset_index(drop=True)
    embeddings = np.concatenate(emb_list, axis=0)

    if len(seg_df) != len(emb_df):
        raise ValueError("Segmentation rows and embedding rows do not match in length.")

    if not np.all(seg_df["image_path"].values == emb_df["image_path"].values):
        raise ValueError("Image path ordering mismatch between segmentation and embedding passes.")

    kmeans = KMeans(n_clusters=args.num_clusters, random_state=args.seed, n_init=10)
    clusters = kmeans.fit_predict(embeddings)

    out_df = seg_df.copy()
    out_df["embedding_cluster"] = clusters

    cluster_stats = (
        out_df.groupby("embedding_cluster")
        .agg(
            num_samples=("image_path", "count"),
            mean_confidence=("mean_confidence", "mean"),
            mean_entropy=("mean_entropy", "mean"),
            mean_num_classes=("num_predicted_classes", "mean"),
        )
        .reset_index()
        .sort_values("mean_entropy", ascending=False)
    )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    out_df.to_csv(out_dir / "robot_prediction_analysis.csv", index=False)
    cluster_stats.to_csv(out_dir / "robot_cluster_summary.csv", index=False)

    hardest = out_df.sort_values(["mean_entropy", "mean_confidence"], ascending=[False, True]).head(args.top_k)
    easiest = out_df.sort_values(["mean_entropy", "mean_confidence"], ascending=[True, False]).head(args.top_k)

    hardest.to_csv(out_dir / "hardest_frames.csv", index=False)
    easiest.to_csv(out_dir / "easiest_frames.csv", index=False)

    print(f"[ok] saved: {out_dir / 'robot_prediction_analysis.csv'}")
    print(f"[ok] saved: {out_dir / 'robot_cluster_summary.csv'}")
    print(f"[ok] saved: {out_dir / 'hardest_frames.csv'}")
    print(f"[ok] saved: {out_dir / 'easiest_frames.csv'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-dir", type=str, default="data/interim/robot_frames")
    parser.add_argument("--seg-checkpoint", type=str, default="outputs/checkpoints/deeplabv3_cityscapes/best_model.pt")
    parser.add_argument("--ae-checkpoint", type=str, default="outputs/checkpoints/autoencoder/best_autoencoder.pt")
    parser.add_argument("--output-dir", type=str, default="outputs/reports/robot_analysis")
    parser.add_argument("--suffix", type=str, default=".png")
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--ae-height", type=int, default=256)
    parser.add_argument("--ae-width", type=int, default=512)
    parser.add_argument("--latent-dim", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--num-clusters", type=int, default=8)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    main(args)