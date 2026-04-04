from pathlib import Path
import argparse
import random
import matplotlib
# Use 'Agg' for non-interactive environments (saves to file)
# or 'module://matplotlib_inline.backend_inline' for VS Code
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import numpy as np
import torch

from src.datasets.image_only_dataset import ImageOnlyDataset
from src.datasets.transforms import get_autoencoder_val_transforms
from src.models.representation.autoencoder import ConvAutoencoder


def to_display_image(tensor_img):
    img = tensor_img.detach().cpu().numpy().transpose(1, 2, 0)
    img = np.clip(img, 0.0, 1.0)
    return img


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

    model = load_model(args.checkpoint, args.latent_dim, device)

    indices = list(range(len(dataset)))
    random.seed(args.seed)
    random.shuffle(indices)
    indices = indices[:args.num_samples]

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(len(indices), 2, figsize=(10, 4 * len(indices)))
    if len(indices) == 1:
        axes = np.array([axes])

    for row, idx in enumerate(indices):
        sample = dataset[idx]
        image = sample["image"].unsqueeze(0).to(device)

        recon, z = model(image)

        image_np = to_display_image(sample["image"])
        recon_np = to_display_image(recon.squeeze(0))

        axes[row, 0].imshow(image_np)
        axes[row, 0].set_title(f"Original\n{Path(sample['img_path']).name}")
        axes[row, 0].axis("off")

        axes[row, 1].imshow(recon_np)
        axes[row, 1].set_title(f"Reconstruction\nlatent: {tuple(z.shape)}")
        axes[row, 1].axis("off")

    plt.tight_layout()
    save_path = out_dir / "reconstructions.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"[ok] saved {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-dir", type=str, default="data/interim/robot_frames")
    parser.add_argument("--checkpoint", type=str, default="outputs/checkpoints/autoencoder/best_autoencoder.pt")
    parser.add_argument("--output-dir", type=str, default="outputs/figures/autoencoder")
    parser.add_argument("--suffix", type=str, default=".png")
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--latent-dim", type=int, default=256)
    parser.add_argument("--num-samples", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    run(args)