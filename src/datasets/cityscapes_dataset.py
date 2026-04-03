from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset
import numpy as np


class CityscapesSegDataset(Dataset):
    CLASSES = (
        "road", "sidewalk", "building", "wall", "fence", "pole",
        "traffic light", "traffic sign", "vegetation", "terrain",
        "sky", "person", "rider", "car", "truck", "bus", "train",
        "motorcycle", "bicycle"
    )
    IGNORE_INDEX = 255

    def __init__(self, root, split="train", transform=None):
        self.root = Path(root)
        self.split = split
        self.transform = transform

        self.img_dir = self.root / "images" / split
        self.mask_dir = self.root / "gtFine" / split

        self.samples = sorted(self.img_dir.rglob("*_leftImg8bit.png"))
        if len(self.samples) == 0:
            raise RuntimeError(f"No images found in {self.img_dir}")

    def __len__(self):
        return len(self.samples)

    def _get_mask_path(self, img_path):
        city = img_path.parent.name
        mask_name = img_path.name.replace("_leftImg8bit.png", "_gtFine_labelTrainIds.png")
        return self.mask_dir / city / mask_name

    def __getitem__(self, idx):
        img_path = self.samples[idx]
        mask_path = self._get_mask_path(img_path)

        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path)

        image = np.array(image)
        mask = np.array(mask, dtype=np.uint8)

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
            "img_path": str(img_path),
            "mask_path": str(mask_path),
        }
    

# def main():
#     ds = CityscapesSegDataset(root="/media/parth/My Passport/Cityspaces", split="val")
#     print("num samples:", len(ds))

#     sample = ds[0]
#     print(sample["image"].shape)
#     print(sample["mask"].shape)
#     print(sample["img_path"])
#     print(sample["mask_path"])
#     print("unique labels:", sample["mask"].unique()[:30])

# if __name__ == "__main__":
#     main()