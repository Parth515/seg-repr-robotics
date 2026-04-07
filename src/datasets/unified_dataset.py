from pathlib import Path
import pandas as pd
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset


class UnifiedSegmentationDataset(Dataset):
    def __init__(
        self,
        csv_path,
        split=None,
        source_dataset=None,
        transform=None,
        require_mask=True,
    ):
        self.csv_path = Path(csv_path)
        self.transform = transform
        self.require_mask = require_mask

        df = pd.read_csv(self.csv_path)

        if split is not None:
            df = df[df["split"] == split]

        if source_dataset is not None:
            df = df[df["source_dataset"] == source_dataset]

        if require_mask and "has_mask" in df.columns:
            df = df[df["has_mask"] == 1]

        df = df.reset_index(drop=True)

        if len(df) == 0:
            raise RuntimeError("No samples found after filtering.")

        self.df = df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        image_path = Path(row["image_path"])
        image = np.array(Image.open(image_path).convert("RGB"))

        sample = {
            "sample_id": row.get("sample_id", str(idx)),
            "image_path": str(image_path),
            "split": row.get("split", ""),
            "source_dataset": row.get("source_dataset", ""),
            "sequence": row.get("sequence", ""),
            "camera": row.get("camera", ""),
            "lighting": row.get("lighting", ""),
            "environment": row.get("environment", ""),
        }

        if self.require_mask:
            mask_path = Path(row["mask_path"])
            mask = np.array(Image.open(mask_path), dtype=np.uint8)

            if self.transform is not None:
                augmented = self.transform(image=image, mask=mask)
                image = augmented["image"]
                mask = augmented["mask"].long()
            else:
                image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
                mask = torch.from_numpy(mask).long()

            sample["mask"] = mask
            sample["mask_path"] = str(mask_path)

        else:
            if self.transform is not None:
                augmented = self.transform(image=image)
                image = augmented["image"]
            else:
                image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0

        sample["image"] = image
        return sample