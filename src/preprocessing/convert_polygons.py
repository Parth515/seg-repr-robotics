import json
from pathlib import Path
from PIL import Image, ImageDraw

IGNORE_INDEX = 255

NAME_TO_TRAINID = {
    "unlabeled": 255,
    "ego vehicle": 255,
    "rectification border": 255,
    "out of roi": 255,
    "static": 255,
    "dynamic": 255,
    "ground": 255,
    "road": 0,
    "sidewalk": 1,
    "parking": 255,
    "rail track": 255,
    "building": 2,
    "wall": 3,
    "fence": 4,
    "guard rail": 255,
    "bridge": 255,
    "tunnel": 255,
    "pole": 5,
    "polegroup": 255,
    "traffic light": 6,
    "traffic sign": 7,
    "vegetation": 8,
    "terrain": 9,
    "sky": 10,
    "person": 11,
    "rider": 12,
    "car": 13,
    "truck": 14,
    "bus": 15,
    "caravan": 255,
    "trailer": 255,
    "train": 16,
    "motorcycle": 17,
    "bicycle": 18,
    "license plate": 255,
}

TRAINID_TO_COLOR = {
    0: (128, 64, 128),
    1: (244, 35, 232),
    2: (70, 70, 70),
    3: (102, 102, 156),
    4: (190, 153, 153),
    5: (153, 153, 153),
    6: (250, 170, 30),
    7: (220, 220, 0),
    8: (107, 142, 35),
    9: (152, 251, 152),
    10: (70, 130, 180),
    11: (220, 20, 60),
    12: (255, 0, 0),
    13: (0, 0, 142),
    14: (0, 0, 70),
    15: (0, 60, 100),
    16: (0, 80, 100),
    17: (0, 0, 230),
    18: (119, 11, 32),
    255: (0, 0, 0),
}

def normalize_label(label: str) -> str:
    if label in NAME_TO_TRAINID:
        return label
    if label.endswith("group"):
        base = label[:-5]
        if base in NAME_TO_TRAINID:
            return base
    return label

def polygon_valid(poly):
    return isinstance(poly, list) and len(poly) >= 3

def convert_one(json_path: Path, save_color=False, overwrite=False):
    json_path = Path(json_path)
    out_mask = json_path.with_name(json_path.name.replace("_polygons.json", "_labelTrainIds.png"))
    out_color = json_path.with_name(json_path.name.replace("_polygons.json", "_color.png"))

    if out_mask.exists() and not overwrite:
        print(f"[skip] exists: {out_mask}")
        return

    with open(json_path, "r") as f:
        data = json.load(f)

    width = data["imgWidth"]
    height = data["imgHeight"]
    objects = data.get("objects", [])

    mask = Image.new("L", (width, height), IGNORE_INDEX)
    draw = ImageDraw.Draw(mask)

    unknown_labels = set()
    skipped_invalid = 0

    for obj in objects:
        label = normalize_label(obj.get("label", ""))
        polygon = obj.get("polygon", [])

        if not polygon_valid(polygon):
            skipped_invalid += 1
            continue

        if label not in NAME_TO_TRAINID:
            unknown_labels.add(label)
            continue

        train_id = NAME_TO_TRAINID[label]
        points = [tuple(p) for p in polygon]
        draw.polygon(points, fill=train_id)

    mask.save(out_mask)

    if save_color:
        color_img = Image.new("RGB", (width, height), (0, 0, 0))
        mask_pixels = mask.load()
        color_pixels = color_img.load()
        for y in range(height):
            for x in range(width):
                color_pixels[x, y] = TRAINID_TO_COLOR.get(mask_pixels[x, y], (0, 0, 0))
        color_img.save(out_color)

    print(f"[ok] {json_path.name} -> {out_mask.name}")
    if save_color:
        print(f"[ok] color -> {out_color.name}")
    if skipped_invalid:
        print(f"  invalid polygons skipped: {skipped_invalid}")
    if unknown_labels:
        print(f"  unknown labels: {sorted(unknown_labels)}")

def convert_dir(root_dir: Path, save_color=False, overwrite=False):
    root_dir = Path(root_dir)
    json_files = sorted(root_dir.rglob("*_gtFine_polygons.json"))
    print(f"found {len(json_files)} json files")

    for jp in json_files:
        convert_one(jp, save_color=save_color, overwrite=overwrite)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="JSON file or root folder")
    parser.add_argument("--save-color", action="store_true", help="Save color visualization PNG")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    p = Path(args.input)
    if p.is_file():
        convert_one(p, save_color=args.save_color, overwrite=args.overwrite)
    else:
        convert_dir(p, save_color=args.save_color, overwrite=args.overwrite)