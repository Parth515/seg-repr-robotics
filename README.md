seg-repr-robotics/
├── configs/
│   ├── dataset/
│   │   ├── cityscapes.yaml
│   │   ├── robot.yaml
│   │   └── unified.yaml
│   ├── model/
│   │   ├── unet.yaml
│   │   ├── deeplabv3.yaml
│   │   ├── autoencoder.yaml
│   │   └── classical.yaml
│   └── train/
│       ├── seg_train.yaml
│       ├── ae_train.yaml
│       └── eval.yaml
├── data/
│   ├── raw/
│   │   ├── cityscapes/
│   │   └── robot_raw/
│   ├── interim/
│   │   ├── robot_frames/
│   │   ├── robot_masks/
│   │   └── manifests/
│   └── processed/
│       ├── unified/
│       │   ├── images/
│       │   │   ├── train/
│       │   │   ├── val/
│       │   │   └── test/
│       │   ├── masks/
│       │   │   ├── train/
│       │   │   ├── val/
│       │   │   └── test/
│       │   └── metadata/
│       │       ├── samples.csv
│       │       ├── classes.json
│       │       └── splits.json
├── src/
│   ├── datasets/
│   │   ├── cityscapes_dataset.py
│   │   ├── robot_dataset.py
│   │   ├── unified_dataset.py
│   │   ├── transforms.py
│   │   └── label_maps.py
│   ├── preprocessing/
│   │   ├── extract_frames.py
│   │   ├── convert_polygons.py
│   │   ├── remap_labels.py
│   │   ├── build_metadata.py
│   │   └── make_splits.py
│   ├── models/
│   │   ├── segmentation/
│   │   │   ├── unet.py
│   │   │   ├── deeplab.py
│   │   │   └── classical.py
│   │   └── representation/
│   │       ├── autoencoder.py
│   │       ├── encoder_head.py
│   │       └── latent_eval.py
│   ├── training/
│   │   ├── train_segmentation.py
│   │   ├── train_autoencoder.py
│   │   └── losses.py
│   ├── evaluation/
│   │   ├── metrics_segmentation.py
│   │   ├── metrics_representation.py
│   │   ├── visualize_masks.py
│   │   ├── visualize_latents.py
│   │   └── robustness_report.py
│   └── utils/
│       ├── io.py
│       ├── logger.py
│       └── seed.py
├── experiments/
├── outputs/
│   ├── checkpoints/
│   ├── figures/
│   └── reports/
└── README.md