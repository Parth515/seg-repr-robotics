from pathlib import Path
import argparse
import csv

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.datasets.image_only_dataset import ImageOnlyDataset
from src.datasets.transforms import get_autoencoder_val_transforms
from src.models.representation.autoencoder import ConvAutoencoder


def load_model(checkpoint_path, latent_dim, device):
    model = ConvAutoencoder(latent_dim=latent_dim)
    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


@torch.no_grad()
def run(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = ImageOnlyDataset(
        image_dir=args.image_dir,
        transform=get_autoencoder_val_transforms((args.height, args.width)),
        suffix=args.suffix,
    )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    model = load_model(args.checkpoint, args.latent_dim, device)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_paths = []
    all_embeddings = []

    for batch in loader:
        images = batch["image"].to(device)
        img_paths = batch["img_path"]

        z = model.encode(images)                  # [B, C, H, W]
        z = torch.mean(z, dim=(2, 3))            # global average pooling -> [B, C]
        z = z.cpu().numpy().astype(np.float32)

        all_embeddings.append(z)
        all_paths.extend(img_paths)

    all_embeddings = np.concatenate(all_embeddings, axis=0)

    np.save(out_dir / "embeddings.npy", all_embeddings)

    with open(out_dir / "embedding_index.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["idx", "img_path"])
        for idx, path in enumerate(all_paths):
            writer.writerow([idx, path])

    print(f"[ok] saved embeddings: {all_embeddings.shape}")
    print(f"[ok] saved index: {out_dir / 'embedding_index.csv'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-dir", type=str, default="data/interim/robot_frames")
    parser.add_argument("--checkpoint", type=str, default="outputs/checkpoints/autoencoder/best_autoencoder.pt")
    parser.add_argument("--output-dir", type=str, default="outputs/features/autoencoder")
    parser.add_argument("--suffix", type=str, default=".png")
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--latent-dim", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    args = parser.parse_args()
    run(args)