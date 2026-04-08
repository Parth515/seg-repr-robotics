# Robot Perception: Segmentation + Representation Learning

This project combines:
- **Semantic segmentation** with DeepLabV3-ResNet50
- **Autoencoder-based representation learning** on robot images
- **Target-domain adaptation** using a small labeled robot subset

The goal is to study how a segmentation model trained on **Cityscapes** can be adapted to robot-collected imagery.

---

## Project Flow

1. Train segmentation on **Cityscapes**
2. Train autoencoder on **all robot frames**
3. Run segmentation on robot frames
4. Select hard/diverse robot frames
5. Manually label a small robot subset
6. Fine-tune the segmentation model on robot data
7. Compare:
   - Cityscapes-only
   - Robot-only fine-tuned
   - Mixed fine-tuned

---

## Folder Layout

```text
data/
├── cityscapes/
│   ├── images/
│   └── gtFine/
└── robot/
    ├── leftImg8bit/
    │   ├── train/
    │   ├── val/
    │   └── test/
    └── gtFine/
        ├── train/
        ├── val/
        └── test/
```

Robot masks should follow Cityscapes-style naming:

- `*_gtFine_labelIds.png`
- `*_gtFine_labelTrainIds.png`
- optional: `*_gtFine_color.png`, `*_gtFine_instanceIds.png`



---

## Prepare Robot Labels

If your robot annotations are exported in Cityscapes style and already contain `labelIds`, generate `labelTrainIds` with:

```bash
python3 -m src.preprocessing.labelids_to_trainids --input data/robot/gtFine
```

---

## Training

### 1) Train segmentation on Cityscapes
Use your Cityscapes training script to train the base DeepLabV3 model.

### 2) Train autoencoder on robot frames
Train the autoencoder on all robot images, labeled or unlabeled.

### 3) Fine-tune on labeled robot subset
Run robot-only or mixed fine-tuning:

```bash
python3 -m src.training.finetune_on_robot_subset \
  --cityscapes-root data/cityscapes \
  --robot-root data/robot \
  --init-checkpoint outputs/checkpoints/deeplabv3_cityscapes/best_model.pt \
  --mode mixed \
  --eval-split val \
  --output-dir outputs/checkpoints/mixed_ft
```

Final training on `train+val` and evaluation on `test`:

```bash
python3 -m src.training.finetune_on_robot_subset \
  --cityscapes-root data/cityscapes \
  --robot-root data/robot \
  --init-checkpoint outputs/checkpoints/deeplabv3_cityscapes/best_model.pt \
  --mode final_trainval_mixed \
  --eval-split test \
  --output-dir outputs/checkpoints/mixed_final
```

---

## Modes

- `robot_only` — fine-tune only on labeled robot train set
- `mixed` — fine-tune on Cityscapes train + robot train
- `final_trainval_robot_only` — final training on robot train+val
- `final_trainval_mixed` — final training on Cityscapes train + robot train+val

---

## Outputs

Typical outputs are saved in:

```text
outputs/
├── checkpoints/
├── reports/
└── visualizations/
```

Important checkpoints:
- `best_model.pt`
- `last_model.pt`
- `final_model.pt`

---

## Notes

- Use `labelTrainIds` masks for segmentation training.
- Keep robot `test` untouched until final evaluation.
- Start with a small labeled robot subset, then expand only if needed.

---

## Result Goal

This project evaluates whether robot-domain adaptation with a small labeled subset can improve semantic segmentation performance compared with a Cityscapes-only baseline.
