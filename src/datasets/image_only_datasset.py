from pathlib import Path
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset

class ImageOnlyDataset(Dataset):
    def __init__(self, image_dir, transform=None, suffix=".png"):
        self.image_dir = Path(image_dir)
        self.transform = transform
        self.samples = sorted(self.image_dir.rglob(f"*{suffix}"))
        if len(self.samples) == 0:
            raise RuntimeError(f"{self.image_dir} is Empty.")
        
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path = self.samples[idx]
        image = np.array(Image.open(img_path).convert("RGB"))

        if self.transform is not None:
            image = self.transform(image=image)["image"]

        else:
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0

        return {
            "image": image,
            "img_path": str(img_path)
        }