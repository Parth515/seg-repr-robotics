# src/training/train_autoencoder.py

from pathlib import Path
import argparse
import json
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from src.datasets.image_only_dataset import ImageOnlyDataset
from src.datasets.transforms import (
    get_autoencoder_train_transforms,
    get_autoencoder_val_transforms,
)
from src.models.representation.autoencoder import ConvAutoencoder


def seed_everything(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0

    for batch in loader:
        images = batch["image"].to(device)
        recons, _ = model(images)
        loss = criterion(recons, images)
        total_loss += loss.item()

    return total_loss / len(loader)


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0

    for batch in loader:
        images = batch["image"].to(device)

        optimizer.zero_grad()
        recons, _ = model(images)
        loss = criterion(recons, images)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


def main(args):
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    full_ds = ImageOnlyDataset(
        image_dir=args.image_dir,
        transform=get_autoencoder_train_transforms((args.height, args.width)),
        suffix=args.suffix,
    )

    val_size = int(len(full_ds) * args.val_ratio)
    train_size = len(full_ds) - val_size
    train_ds, val_ds = random_split(
        full_ds,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(args.seed)
    )

    # override val transform
    val_ds.dataset.transform = get_autoencoder_val_transforms((args.height, args.width))

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )

    model = ConvAutoencoder(latent_dim=args.latent_dim).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    history = []
    best_val = float("inf")

    for epoch in range(1, args.epochs + 1):
        start = time.time()

        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss = validate(model, val_loader, criterion, device)
        dt = time.time() - start

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "time_sec": dt,
        }
        history.append(row)

        print(
            f"Epoch {epoch:03d} | "
            f"train_loss={train_loss:.6f} | "
            f"val_loss={val_loss:.6f} | "
            f"time={dt:.1f}s"
        )

        torch.save(model.state_dict(), out_dir / "last_autoencoder.pt")
        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), out_dir / "best_autoencoder.pt")

        with open(out_dir / "history.json", "w") as f:
            json.dump(history, f, indent=2)

    print(f"Best val loss: {best_val:.6f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-dir", type=str, default="data/interim/robot_frames")
    parser.add_argument("--output-dir", type=str, default="outputs/checkpoints/autoencoder")
    parser.add_argument("--suffix", type=str, default=".png")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--latent-dim", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    main(args)