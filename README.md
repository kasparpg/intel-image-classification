# Intel Image Classification

A deep learning project that classifies natural scene images into 6 categories using Transfer Learning with MobileNetV2.

## Categories

- 🏢 Buildings
- 🌲 Forest
- 🏔️ Glacier
- ⛰️ Mountain
- 🌊 Sea
- 🛣️ Street

## Dataset Structure

```
data/
├── seg_train/     # Training images (labeled, ~14k images)
├── seg_test/      # Test images (labeled, ~3k images)
└── seg_pred/      # Prediction images (unlabeled, ~7k images)
```

## Model

- **Architecture**: MobileNetV2 (pre-trained on ImageNet) + custom classification head
- **Transfer Learning**: Base model frozen, only top layers trained
- **Input Size**: 150x150 RGB images
- **Data Augmentation**: Random flip, rotation, zoom, contrast

## Quick Start

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Run the notebook `main.ipynb` cells in order

## Results

Training typically achieves **85-92% test accuracy** in ~10-15 minutes.

## Requirements

- TensorFlow 2.x
- NumPy
- Pandas
- Matplotlib
