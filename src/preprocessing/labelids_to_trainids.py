# src/preprocessing/labelids_to_trainids.py

from pathlib import Path
import argparse
import numpy as np
from PIL import Image

# Cityscapes id -> trainId mapping from labels.py
ID_TO_TRAINID = {
    -1: 255,   # license plate
    0: 255,    # unlabeled
    1: 255,    # ego vehicle
    2: 255,    # rectification border
    3: 255,    # out of roi
    4: 255,    # static
    5: 255,    # dynamic
    6: 255,    # ground
    7: 0,      # road
    8: 1,      # sidewalk
    9: 255,    # parking
    10: 255,   # rail track
    11: 2,     # building
    12: 3,     # wall
    13: 4,     # fence
    14: 255,   # guard rail
    15: 255,   # bridge
    16: 255,   # tunnel
    17: 5,     # pole
    18: 255,   # polegroup
    19: 6,     # traffic light
    20: 7,     # traffic sign
    21: 8,     # vegetation
    22: 9,     # terrain
    23: 10,    # sky
    24: 11,    # person
    25: 12,    # rider
    26: 13,    # car
    27: 14,    # truck
    28: 15,    # bus
    29: 255,   # caravan
    30: 255,   # trailer
    31: 16,    # train
    32: 17,    # motorcycle
    33: 18,    # bicycle
}

def convert_one(labelids_path: Path, overwrite=False):
    labelids_path = Path(labelids_path)

    if not labelids_path.name.endswith("_gtFine_labelIds.png"):
        raise ValueError(f"Expected *_gtFine_labelIds.png, got: {labelids_path.name}")

    out_path = labelids_path.with_name(
        labelids_path.name.replace("_gtFine_labelIds.png", "_gtFine_labelTrainIds.png")
    )

    if out_path.exists() and not overwrite:
        print(f"[skip] exists: {out_path}")
        return

    mask = np.array(Image.open(labelids_path), dtype=np.int32)

    train_mask = np.full(mask.shape, 255, dtype=np.uint8)

    unique_ids = np.unique(mask)
    unknown_ids = []

    for class_id in unique_ids:
        if int(class_id) in ID_TO_TRAINID:
            train_mask[mask == class_id] = ID_TO_TRAINID[int(class_id)]
        else:
            unknown_ids.append(int(class_id))

    Image.fromarray(train_mask, mode="L").save(out_path)

    print(f"[ok] {labelids_path.name} -> {out_path.name}")
    print(f"  unique labelIds: {unique_ids.tolist()}")
    print(f"  unique trainIds: {np.unique(train_mask).tolist()}")
    if unknown_ids:
        print(f"  unknown ids mapped to 255 by default: {unknown_ids}")


def convert_dir(root_dir: Path, overwrite=False):
    root_dir = Path(root_dir)
    files = sorted(root_dir.rglob("*_gtFine_labelIds.png"))
    print(f"found {len(files)} labelIds masks")

    for fp in files:
        convert_one(fp, overwrite=overwrite)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="data/robot/gtFine")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    p = Path(args.input)
    if p.is_file():
        convert_one(p, overwrite=args.overwrite)
    else:
        convert_dir(p, overwrite=args.overwrite)