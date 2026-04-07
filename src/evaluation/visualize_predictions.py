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
from torch.utils.data import DataLoader
from torchvision.models.segmentation import deeplabv3_resnet50

from src.datasets.cityscapes_dataset import CityscapesSegDataset
from src.datasets.transforms import get_val_transforms


NUM_CLASSES = 19
IGNORE_INDEX = 255

CITYSCAPES_COLORS = np.array([
    [128,  64, 128],  # road
    [244,  35, 232],  # sidewalk
    [ 70,  70,  70],  # building
    [102, 102, 156],  # wall
    [190, 153, 153],  # fence
    [153, 153, 153],  # pole
    [250, 170,  30],  # traffic light
    [220, 220,   0],  # traffic sign
    [107, 142,  35],  # vegetation
    [152, 251, 152],  # terrain
    [ 70, 130, 180],  # sky
    [220,  20,  60],  # person
    [255,   0,   0],  # rider
    [  0,   0, 142],  # car
    [  0,   0,  70],  # truck
    [  0,  60, 100],  # bus
    [  0,  80, 100],  # train
    [  0,   0, 230],  # motorcycle
    [119,  11,  32],  # bicycle
], dtype=np.uint8)


def denormalize(img):
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img = img.transpose(1, 2, 0)
    img = (img * std) + mean
    img = np.clip(img, 0, 1)
    return img


def mask_to_color(mask):
    color = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    valid = mask != IGNORE_INDEX
    color[valid] = CITYSCAPES_COLORS[mask[valid]]
    return color


def make_overlay(image_rgb, color_mask, alpha=0.5):
    image_u8 = (image_rgb * 255).astype(np.uint8)
    overlay = image_u8.copy()
    valid = np.any(color_mask != 0, axis=-1)
    overlay[valid] = (
        alpha * color_mask[valid] + (1 - alpha) * image_u8[valid]
    ).astype(np.uint8)
    return overlay


def build_model(ckpt_path, device):
    model = deeplabv3_resnet50(weights=None, num_classes=NUM_CLASSES, aux_loss=True)
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


@torch.no_grad()
def run(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = CityscapesSegDataset(
        root=args.data_root,
        split=args.split,
        transform=get_val_transforms((args.height, args.width)),
    )

    model = build_model(args.checkpoint, device)

    indices = list(range(len(dataset)))
    random.seed(args.seed)
    random.shuffle(indices)
    indices = indices[:args.num_samples]

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for i, idx in enumerate(indices):
        sample = dataset[idx]
        image = sample["image"].unsqueeze(0).to(device)
        gt_mask = sample["mask"].cpu().numpy()

        output = model(image)["out"]
        pred_mask = torch.argmax(output, dim=1).squeeze(0).cpu().numpy().astype(np.uint8)

        image_rgb = denormalize(sample["image"].cpu().numpy())
        gt_color = mask_to_color(gt_mask)
        pred_color = mask_to_color(pred_mask)
        overlay = make_overlay(image_rgb, pred_color, alpha=0.55)

        fig, axes = plt.subplots(1, 4, figsize=(20, 5))
        axes[0].imshow(image_rgb)  # type: ignore
        axes[0].set_title("RGB")
        axes[1].imshow(gt_color)  # type: ignore
        axes[1].set_title("Ground Truth")
        axes[2].imshow(pred_color)  # type: ignore
        axes[2].set_title("Prediction")
        axes[3].imshow(overlay)  # type: ignore
        axes[3].set_title("Overlay")

        for ax in axes:
            ax.axis("off")

        fig.suptitle(Path(sample["img_path"]).name, fontsize=12)
        plt.tight_layout()

        save_path = out_dir / f"sample_r_ro_{i:02d}.png"
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

        print(f"[ok] saved {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=str, default="data/robot")
    parser.add_argument("--checkpoint", type=str, default="outputs/checkpoints/robot_only_ft/best_model.pt")
    parser.add_argument("--output-dir", type=str, default="outputs/figures/predictions")
    parser.add_argument("--split", type=str, default="val")
    parser.add_argument("--num-samples", type=int, default=8)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    run(args)