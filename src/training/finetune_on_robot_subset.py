from pathlib import Path
import argparse
import json
import time

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torchvision.models.segmentation import deeplabv3_resnet50

from src.datasets.transforms import get_train_transforms, get_val_transforms
from src.evaluation.metrics_segmentation import (
    fast_confusion_matrix,
    compute_iou_from_confusion_matrix,
    compute_pixel_accuracy,
    NUM_CLASSES,
    IGNORE_INDEX,
)


def seed_everything(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


class CityscapesLikeSegDataset(Dataset):
    def __init__(
        self,
        root,
        split="train",
        transform=None,
        image_suffix="_leftImg8bit.png",
        mask_suffix="_gtFine_labelTrainIds.png",
    ):
        self.root = Path(root)
        self.split = split
        self.transform = transform
        self.image_suffix = image_suffix
        self.mask_suffix = mask_suffix

        self.image_root = self.root / "images" / split
        self.mask_root = self.root / "gtFine" / split

        if not self.image_root.exists():
            raise FileNotFoundError(f"Image split folder not found: {self.image_root}")
        if not self.mask_root.exists():
            raise FileNotFoundError(f"Mask split folder not found: {self.mask_root}")

        self.samples = self._collect_samples()
        if len(self.samples) == 0:
            raise RuntimeError(f"No samples found in split={split} under {self.root}")

    def _collect_samples(self):
        samples = []
        image_paths = sorted(self.image_root.rglob(f"*{self.image_suffix}"))

        for img_path in image_paths:
            rel = img_path.relative_to(self.image_root)
            city_or_seq = rel.parent
            stem = img_path.name.replace(self.image_suffix, "")
            mask_name = stem + self.mask_suffix
            mask_path = self.mask_root / city_or_seq / mask_name

            if mask_path.exists():
                samples.append((img_path, mask_path))

        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image_path, mask_path = self.samples[idx]

        image = np.array(Image.open(image_path).convert("RGB"))
        mask = np.array(Image.open(mask_path), dtype=np.uint8)

        if self.transform is not None:
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"].long()
        else:
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
            mask = torch.from_numpy(mask).long()

        return {
            "image": image,
            "mask": mask,
            "image_path": str(image_path),
            "mask_path": str(mask_path),
        }


def build_model(checkpoint_path=None, device="cpu"):
    model = deeplabv3_resnet50(weights=None, num_classes=NUM_CLASSES, aux_loss=True)
    if checkpoint_path is not None:
        state = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(state, strict=True)
    return model


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0

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

    return total_loss / max(len(loader), 1)


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    conf_mat = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)

    for batch in loader:
        images = batch["image"].to(device)
        masks = batch["mask"].to(device)

        outputs = model(images)
        loss_main = criterion(outputs["out"], masks)
        loss_aux = criterion(outputs["aux"], masks) if "aux" in outputs else 0.0
        loss = loss_main + 0.4 * loss_aux
        total_loss += loss.item()

        preds = torch.argmax(outputs["out"], dim=1)
        conf_mat += fast_confusion_matrix(
            preds, masks, num_classes=NUM_CLASSES, ignore_index=IGNORE_INDEX
        )

    iou = compute_iou_from_confusion_matrix(conf_mat)
    valid = (conf_mat.sum(axis=0) + conf_mat.sum(axis=1)) > 0
    miou = float(iou[valid].mean()) if valid.any() else 0.0
    pixel_acc = float(compute_pixel_accuracy(conf_mat))

    return total_loss / max(len(loader), 1), miou, pixel_acc


def make_robot_dataset(robot_root, split, image_size, train_mode=False):
    transform = get_train_transforms(image_size) if train_mode else get_val_transforms(image_size)
    return CityscapesLikeSegDataset(
        root=robot_root,
        split=split,
        transform=transform,
    )


def make_cityscapes_dataset(cityscapes_root, split, image_size, train_mode=True):
    transform = get_train_transforms(image_size) if train_mode else get_val_transforms(image_size)
    return CityscapesLikeSegDataset(
        root=cityscapes_root,
        split=split,
        transform=transform,
    )


def make_datasets(args):
    image_size = (args.height, args.width)

    if args.mode == "robot_only":
        train_ds = make_robot_dataset(args.robot_root, "train", image_size, train_mode=True)

    elif args.mode == "mixed":
        train_ds = ConcatDataset([
            make_cityscapes_dataset(args.cityscapes_root, "train", image_size, train_mode=True),
            make_robot_dataset(args.robot_root, "train", image_size, train_mode=True),
        ])

    elif args.mode == "final_trainval_robot_only":
        train_ds = ConcatDataset([
            make_robot_dataset(args.robot_root, "train", image_size, train_mode=True),
            make_robot_dataset(args.robot_root, "val", image_size, train_mode=True),
        ])

    elif args.mode == "final_trainval_mixed":
        train_ds = ConcatDataset([
            make_cityscapes_dataset(args.cityscapes_root, "train", image_size, train_mode=True),
            make_robot_dataset(args.robot_root, "train", image_size, train_mode=True),
            make_robot_dataset(args.robot_root, "val", image_size, train_mode=True),
        ])

    else:
        raise ValueError(f"Unknown mode: {args.mode}")

    eval_ds = None
    if args.eval_split is not None:
        eval_ds = make_robot_dataset(args.robot_root, args.eval_split, image_size, train_mode=False)

    return train_ds, eval_ds


def main(args):
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_ds, eval_ds = make_datasets(args)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    eval_loader = None
    if eval_ds is not None:
        eval_loader = DataLoader(
            eval_ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
        )

    model = build_model(
        checkpoint_path=args.init_checkpoint,
        device=device,
    ).to(device)

    criterion = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    history = []
    best_score = -1.0

    for epoch in range(1, args.epochs + 1):
        start = time.time()

        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)

        row = {"epoch": epoch, "train_loss": train_loss}

        if eval_loader is not None:
            eval_loss, eval_miou, eval_pixel_acc = evaluate(model, eval_loader, criterion, device)
            row.update({
                "eval_loss": eval_loss,
                "eval_miou": eval_miou,
                "eval_pixel_acc": eval_pixel_acc,
            })

        row["time_sec"] = time.time() - start
        history.append(row)

        if eval_loader is not None:
            print(
                f"Epoch {epoch:03d} | "
                f"train_loss={train_loss:.4f} | "
                f"eval_loss={eval_loss:.4f} | "
                f"eval_mIoU={eval_miou:.4f} | "
                f"eval_pixel_acc={eval_pixel_acc:.4f} | "
                f"time={row['time_sec']:.1f}s"
            )
        else:
            print(
                f"Epoch {epoch:03d} | "
                f"train_loss={train_loss:.4f} | "
                f"time={row['time_sec']:.1f}s"
            )

        torch.save(model.state_dict(), out_dir / "last_model.pt")

        if eval_loader is not None:
            if eval_miou > best_score:
                best_score = eval_miou
                torch.save(model.state_dict(), out_dir / "best_model.pt")
        else:
            torch.save(model.state_dict(), out_dir / "final_model.pt")

        with open(out_dir / "history.json", "w") as f:
            json.dump(history, f, indent=2)

    if eval_loader is not None:
        print(f"Best eval mIoU: {best_score:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--cityscapes-root", type=str, default="/media/parth/My Passport/Cityspaces")
    parser.add_argument("--robot-root", type=str, default="data/robot")
    parser.add_argument("--init-checkpoint", type=str, default="outputs/checkpoints/deeplabv3_cityscapes/best_model.pt")

    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=[
            "robot_only",
            "mixed",
            "final_trainval_robot_only",
            "final_trainval_mixed",
        ],
    )

    parser.add_argument(
        "--eval-split",
        type=str,
        default="val",
        choices=["train", "val", "test"],
    )

    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    main(args)