import numpy as np
import torch


IGNORE_INDEX = 255
NUM_CLASSES = 19

CLASS_NAMES = [
    "road", "sidewalk", "building", "wall", "fence", "pole",
    "traffic light", "traffic sign", "vegetation", "terrain",
    "sky", "person", "rider", "car", "truck", "bus", "train",
    "motorcycle", "bicycle"
]


def _to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return x


def fast_confusion_matrix(pred, target, num_classes=NUM_CLASSES, ignore_index=IGNORE_INDEX):
    pred = _to_numpy(pred).astype(np.int64).reshape(-1)
    target = _to_numpy(target).astype(np.int64).reshape(-1)

    valid = target != ignore_index
    pred = pred[valid]
    target = target[valid]

    mask = (target >= 0) & (target < num_classes)
    pred = pred[mask]
    target = target[mask]

    cm = np.bincount(
        num_classes * target + pred,
        minlength=num_classes ** 2
    ).reshape(num_classes, num_classes)

    return cm


def compute_iou_from_confusion_matrix(conf_mat):
    tp = np.diag(conf_mat)
    fp = conf_mat.sum(axis=0) - tp
    fn = conf_mat.sum(axis=1) - tp

    denom = tp + fp + fn
    iou = np.divide(
        tp,
        denom,
        out=np.zeros_like(tp, dtype=np.float64),
        where=denom != 0
    )
    return iou


def compute_pixel_accuracy(conf_mat):
    correct = np.diag(conf_mat).sum()
    total = conf_mat.sum()
    if total == 0:
        return 0.0
    return correct / total


def compute_metrics(pred, target, num_classes=NUM_CLASSES, ignore_index=IGNORE_INDEX, class_names=None):
    conf_mat = fast_confusion_matrix(
        pred=pred,
        target=target,
        num_classes=num_classes,
        ignore_index=ignore_index,
    )

    iou = compute_iou_from_confusion_matrix(conf_mat)
    valid_classes = conf_mat.sum(axis=1) + conf_mat.sum(axis=0) > 0
    miou = iou[valid_classes].mean() if valid_classes.any() else 0.0
    pixel_acc = compute_pixel_accuracy(conf_mat)

    results = {
        "pixel_accuracy": float(pixel_acc),
        "mean_iou": float(miou),
        "confusion_matrix": conf_mat,
        "per_class_iou": {}
    }

    if class_names is None:
        class_names = [f"class_{i}" for i in range(num_classes)]

    for idx, name in enumerate(class_names):
        results["per_class_iou"][name] = float(iou[idx])

    return results


