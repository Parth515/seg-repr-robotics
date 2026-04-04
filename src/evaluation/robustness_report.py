from pathlib import Path
import argparse
import pandas as pd
import numpy as np
import torch
import matplotlib
# Use 'Agg' for non-interactive environments (saves to file)
# or 'module://matplotlib_inline.backend_inline' for VS Code
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision.models.segmentation import deeplabv3_resnet50

from src.datasets.cityscapes_dataset import CityscapesSegDataset
from src.datasets.transforms import get_val_transforms
from src.evaluation.metrics_segmentation import (
    fast_confusion_matrix,
    compute_iou_from_confusion_matrix,
    compute_pixel_accuracy,
    CLASS_NAMES,
    NUM_CLASSES,
    IGNORE_INDEX,
)


def build_model(checkpoint_path, device):
    model = deeplabv3_resnet50(weights=None, num_classes=NUM_CLASSES, aux_loss=True)
    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


def infer_group_from_path(img_path, mode="sequence"):
    p = Path(img_path)
    if mode == "sequence":
        return p.parent.name
    return "all"


@torch.no_grad()
def run(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = CityscapesSegDataset(
        root=args.data_root,
        split=args.split,
        transform=get_val_transforms((args.height, args.width)),
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    model = build_model(args.checkpoint, device)

    group_confmats = {}
    group_counts = {}

    for batch in loader:
        images = batch["image"].to(device)
        masks = batch["mask"].to(device)
        img_paths = batch["img_path"]

        logits = model(images)["out"]
        preds = torch.argmax(logits, dim=1)

        for i in range(images.shape[0]):
            group = infer_group_from_path(img_paths[i], mode=args.group_mode)

            if group not in group_confmats:
                group_confmats[group] = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
                group_counts[group] = 0

            cm = fast_confusion_matrix(
                preds[i],
                masks[i],
                num_classes=NUM_CLASSES,
                ignore_index=IGNORE_INDEX
            )
            group_confmats[group] += cm
            group_counts[group] += 1

    rows = []
    for group, cm in group_confmats.items():
        iou = compute_iou_from_confusion_matrix(cm)
        valid = (cm.sum(axis=0) + cm.sum(axis=1)) > 0
        miou = float(iou[valid].mean()) if valid.any() else 0.0
        pixel_acc = float(compute_pixel_accuracy(cm))

        rows.append({
            "group": group,
            "num_samples": group_counts[group],
            "pixel_accuracy": pixel_acc,
            "mean_iou": miou,
        })

    report_df = pd.DataFrame(rows).sort_values("mean_iou", ascending=False)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    report_csv = out_dir / f"robustness_{args.group_mode}.csv"
    report_df.to_csv(report_csv, index=False)

    plt.figure(figsize=(10, 5))
    plt.bar(report_df["group"], report_df["mean_iou"])
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("mIoU")
    plt.title(f"Robustness by {args.group_mode}")
    plt.tight_layout()

    chart_path = out_dir / f"robustness_{args.group_mode}.png"
    plt.savefig(chart_path, dpi=160, bbox_inches="tight")
    plt.close()

    print(f"[ok] saved report: {report_csv}")
    print(f"[ok] saved chart: {chart_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=str, default="/media/parth/My Passport/Cityspaces")
    parser.add_argument("--checkpoint", type=str, default="outputs/checkpoints/deeplabv3_cityscapes/best_model.pt")
    parser.add_argument("--output-dir", type=str, default="outputs/reports")
    parser.add_argument("--split", type=str, default="val")
    parser.add_argument("--group-mode", type=str, default="sequence", choices=["sequence"])
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=4)
    args = parser.parse_args()
    run(args)