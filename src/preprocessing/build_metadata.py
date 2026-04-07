from pathlib import Path
import argparse
import hashlib
import pandas as pd


def md5_file(path, chunk_size=1024 * 1024):
    h = hashlib.md5()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def build_cityscapes_manifest(root_dir: Path, include_hash=False):
    rows = []

    left_root = root_dir / "images"
    gt_root = root_dir / "gtFine"

    for split in ["train", "val", "test"]:
        split_dir = left_root / split
        if not split_dir.exists():
            continue

        for img_path in sorted(split_dir.rglob("*_leftImg8bit.png")):
            city = img_path.parent.name
            stem = img_path.name.replace("_leftImg8bit.png", "")

            mask_path = gt_root / split / city / f"{stem}_gtFine_labelTrainIds.png"
            sample_id = stem

            row = {
                "sample_id": sample_id,
                "source_dataset": "cityscapes",
                "split": split,
                "sequence": city,
                "camera": "front_rgb",
                "lighting": "unknown",
                "environment": "urban",
                "image_path": str(img_path),
                "mask_path": str(mask_path) if mask_path.exists() else "",
                "has_mask": int(mask_path.exists()),
            }

            if include_hash:
                row["image_md5"] = md5_file(img_path)
                row["mask_md5"] = md5_file(mask_path) if mask_path.exists() else ""

            rows.append(row)

    return pd.DataFrame(rows)


def main(args):
    root_dir = Path(args.data_root)
    output_path = Path(args.output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = build_cityscapes_manifest(root_dir, include_hash=args.include_hash)
    df.to_csv(output_path, index=False)

    print(f"[ok] wrote {len(df)} rows to {output_path}")
    print(df.head())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=str, default="data/robot")
    parser.add_argument("--output-csv", type=str, default="data/processed/metadata/samples_robot.csv")
    parser.add_argument("--include-hash", action="store_true", help="Add MD5 hashes for reproducibility")
    args = parser.parse_args()
    main(args)