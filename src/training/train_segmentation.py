from pathlib import Path
import argparse
import json
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.models.segmentation import deeplabv3_resnet50

from src.datasets.cityscapes_dataset import CityscapesSegDataset
from src.datasets.transforms import get_train_transforms, get_val_transforms


NUM_CLASSES = 19
IGNORE_INDEX = 255


def seed_everything(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def build_model(num_classes=NUM_CLASSES):
    model = deeplabv3_resnet50(weights=None, num_classes=num_classes, aux_loss=True)
    return model


def compute_miou(logits, targets, num_classes=NUM_CLASSES, ignore_index=IGNORE_INDEX):
    preds = torch.argmax(logits, dim=1)

    preds = preds.view(-1)
    targets = targets.view(-1)

    valid = targets != ignore_index
    preds = preds[valid]
    targets = targets[valid]

    ious = []
    for cls in range(num_classes):
        pred_c = preds == cls
        target_c = targets == cls
        intersection = (pred_c & target_c).sum().item()
        union = (pred_c | target_c).sum().item()
        if union == 0:
            continue
        ious.append(intersection / union)

    if len(ious) == 0:
        return 0.0
    return float(np.mean(ious))


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    total_miou = 0.0

    for batch in loader:
        images = batch["image"].to(device)
        masks = batch["mask"].to(device)

        optimizer.zero_grad()
        outputs = model(images)

        loss_main = criterion(outputs["out"], masks)
        loss_aux = criterion(outputs["aux"], masks) if "aux" in outputs else 0.0
        loss = loss_main + 0.4 * loss_aux

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_miou += compute_miou(outputs["out"].detach(), masks)

    return total_loss / len(loader), total_miou / len(loader)

# Short version
@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_miou = 0.0

    for batch in loader:
        images = batch["image"].to(device)
        masks = batch["mask"].to(device)

        outputs = model(images)
        loss_main = criterion(outputs["out"], masks)
        loss_aux = criterion(outputs["aux"], masks) if "aux" in outputs else 0.0
        loss = loss_main + 0.4 * loss_aux

        total_loss += loss.item()
        total_miou += compute_miou(outputs["out"], masks)

    return total_loss / len(loader), total_miou / len(loader)

# Better Validation loop 
# accumulate one confusion matrix over the whole validation set and compute metrics once at the end.
# @torch.no_grad()
# def validate(model, loader, criterion, device):
#     from src.evaluation.metrics_segmentation import (
#         fast_confusion_matrix,
#         compute_iou_from_confusion_matrix,
#         compute_pixel_accuracy,
#         NUM_CLASSES,
#         IGNORE_INDEX,
#     )

#     model.eval()
#     total_loss = 0.0
#     conf_mat = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)

#     for batch in loader:
#         images = batch["image"].to(device)
#         masks = batch["mask"].to(device)

#         outputs = model(images)
#         loss_main = criterion(outputs["out"], masks)
#         loss_aux = criterion(outputs["aux"], masks) if "aux" in outputs else 0.0
#         loss = loss_main + 0.4 * loss_aux
#         total_loss += loss.item()

#         preds = torch.argmax(outputs["out"], dim=1)
#         conf_mat += fast_confusion_matrix(preds, masks, NUM_CLASSES, IGNORE_INDEX)

#     iou = compute_iou_from_confusion_matrix(conf_mat)
#     valid_classes = conf_mat.sum(axis=1) + conf_mat.sum(axis=0) > 0
#     miou = float(iou[valid_classes].mean()) if valid_classes.any() else 0.0
#     pixel_acc = float(compute_pixel_accuracy(conf_mat))

#     return total_loss / len(loader), miou, pixel_acc, conf_mat


def main(args):
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_ds = CityscapesSegDataset(
        root=args.data_root,
        split="train",
        transform=get_train_transforms((args.height, args.width)),
    )
    val_ds = CityscapesSegDataset(
        root=args.data_root,
        split="val",
        transform=get_val_transforms((args.height, args.width)),
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        #drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    model = build_model().to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    best_miou = -1.0
    history = []

    for epoch in range(1, args.epochs + 1):
        start = time.time()

        train_loss, train_miou = train_one_epoch(model, train_loader, optimizer, criterion, device)
        # If you use short version
        val_loss, val_miou = validate(model, val_loader, criterion, device)

        # If you use better validation
        # val_loss, val_miou, val_pixel_acc, conf_mat = validate(model, val_loader, criterion, device)

        epoch_time = time.time() - start

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_miou": train_miou,
            "val_loss": val_loss,
            "val_miou": val_miou,
            "time_sec": epoch_time,
        }
        history.append(row)

        print(
            f"Epoch {epoch:03d} | "
            f"train_loss={train_loss:.4f} train_mIoU={train_miou:.4f} | "
            f"val_loss={val_loss:.4f} val_mIoU={val_miou:.4f} | "
            f"time={epoch_time:.1f}s"
        )

        # For better validation usage, you can visualize by adding it into print above
        # print(f"val_loss={val_loss:.4f} val_mIoU={val_miou:.4f} val_pixel_acc={val_pixel_acc:.4f}")

        torch.save(model.state_dict(), out_dir / "last_model.pt")

        if val_miou > best_miou:
            best_miou = val_miou
            torch.save(model.state_dict(), out_dir / "best_model.pt")

        with open(out_dir / "history.json", "w") as f:
            json.dump(history, f, indent=2)

    print(f"Best val mIoU: {best_miou:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=str, default="/media/parth/My Passport/Cityspaces")
    parser.add_argument("--output-dir", type=str, default="outputs/checkpoints/deeplabv3_cityscapes")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    main(args)