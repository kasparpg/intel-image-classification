# Intel Image Classification

A deep learning project for classifying natural scene images into 6 categories using TensorFlow/Keras.

## 📋 Overview

This project implements image classification models to categorize images into:
- **Buildings** - Urban structures and architecture
- **Forest** - Dense vegetation and trees
- **Glacier** - Ice formations and snowy landscapes
- **Mountain** - Mountain ranges and peaks
- **Sea** - Ocean and water bodies
- **Street** - Urban roads and pathways

## 📁 Project Structure

```
intel-image-classification/
├── data/
│   ├── seg_train/          # Training images (organized by class)
│   ├── seg_test/           # Test images (organized by class)
│   └── seg_pred/           # Images for prediction
├── models/                  # Saved model files
├── intel_image_classification.ipynb  # Main Jupyter notebook
├── models.py               # Model architectures
├── utils.py                # Utility functions
├── train.py                # Training script
├── predict.py              # Prediction script
├── requirements.txt        # Python dependencies
└── README.md               # This file
```

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- pip or conda package manager

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/intel-image-classification.git
cd intel-image-classification
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

### Dataset

The dataset should be organized in the `data/` directory with the following structure:
- `seg_train/` - Training images with subfolders for each class
- `seg_test/` - Test images with subfolders for each class
- `seg_pred/` - Images for prediction (optional)

## 📓 Using the Notebook

The main Jupyter notebook (`intel_image_classification.ipynb`) provides an interactive environment for:

1. **Data Exploration** - Visualize dataset distribution and sample images
2. **Preprocessing** - Data augmentation and normalization
3. **Model Building** - Custom CNN and transfer learning models
4. **Training** - Model training with callbacks
5. **Evaluation** - Performance metrics and confusion matrix
6. **Predictions** - Make predictions on new images

To run the notebook:
```bash
jupyter notebook intel_image_classification.ipynb
```

## 💻 Command Line Usage

### Training

Train a model using the command line:

```bash
# Train with MobileNetV2 (default)
python train.py

# Train with custom CNN
python train.py --model custom --epochs 50 --batch_size 64

# Train with ResNet50
python train.py --model resnet --learning_rate 0.0001

# Full options
python train.py \
    --model mobilenet \
    --epochs 30 \
    --batch_size 32 \
    --img_size 150 \
    --learning_rate 0.001 \
    --data_dir data \
    --output_dir models
```

### Prediction

Make predictions on images:

```bash
# Predict single image
python predict.py --model models/best_mobilenet_model.keras --image path/to/image.jpg

# Predict folder of images
python predict.py --model models/best_mobilenet_model.keras --folder data/seg_pred

# Save predictions to CSV
python predict.py --model models/best_mobilenet_model.keras --folder data/seg_pred --output predictions.csv
```

## 🏗️ Model Architectures

### Custom CNN
A deep convolutional neural network built from scratch with:
- 4 convolutional blocks with BatchNormalization and Dropout
- Dense layers for classification
- ~2M parameters

### Transfer Learning Models
Pre-trained models with custom classification heads:
- **MobileNetV2** - Lightweight and efficient
- **ResNet50** - Deep residual network
- **EfficientNetB0** - Balanced accuracy and efficiency

## 📊 Results

| Model | Test Accuracy | Parameters |
|-------|---------------|------------|
| Custom CNN | ~85% | ~2M |
| MobileNetV2 | ~90% | ~3.5M |
| ResNet50 | ~91% | ~25M |
| EfficientNetB0 | ~92% | ~5M |

*Results may vary based on training configuration*

## 🔧 Configuration

### Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `img_size` | 150 | Input image dimensions |
| `batch_size` | 32 | Training batch size |
| `epochs` | 30 | Number of training epochs |
| `learning_rate` | 0.001 | Initial learning rate |
| `validation_split` | 0.2 | Fraction for validation |

### Data Augmentation

The training pipeline includes:
- Random rotation (±20°)
- Width/height shifts (±20%)
- Shear transformation
- Zoom (±20%)
- Horizontal flip

## 📝 License

This project is licensed under the MIT License.

## 🙏 Acknowledgments

- Intel for providing the image dataset
- TensorFlow/Keras team for the deep learning framework
- The open-source community for various tools and libraries

## 📧 Contact

For questions or feedback, please open an issue on GitHub.
