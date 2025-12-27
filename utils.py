"""
Utility functions for Intel Image Classification project.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
from typing import List, Tuple, Optional, Dict
import tensorflow as tf
from tensorflow import keras


# Constants
CLASS_NAMES = ['buildings', 'forest', 'glacier', 'mountain', 'sea', 'street']
DEFAULT_IMG_SIZE = 150


def count_images_in_directory(directory: Path) -> Dict[str, int]:
    """
    Count images in each class folder.
    
    Args:
        directory: Path to the dataset directory
        
    Returns:
        Dictionary with class names as keys and image counts as values
    """
    counts = {}
    for class_name in CLASS_NAMES:
        class_path = directory / class_name
        if class_path.exists():
            counts[class_name] = len(list(class_path.glob('*')))
        else:
            counts[class_name] = 0
    return counts


def load_and_preprocess_image(
    image_path: str,
    target_size: Tuple[int, int] = (DEFAULT_IMG_SIZE, DEFAULT_IMG_SIZE),
    normalize: bool = True
) -> np.ndarray:
    """
    Load and preprocess a single image.
    
    Args:
        image_path: Path to the image file
        target_size: Target size for resizing (height, width)
        normalize: Whether to normalize pixel values to [0, 1]
        
    Returns:
        Preprocessed image as numpy array
    """
    img = Image.open(image_path)
    img = img.convert('RGB')
    img = img.resize(target_size)
    img_array = np.array(img)
    
    if normalize:
        img_array = img_array / 255.0
    
    return img_array


def predict_single_image(
    model: keras.Model,
    image_path: str,
    class_names: List[str] = CLASS_NAMES,
    img_size: int = DEFAULT_IMG_SIZE
) -> Tuple[str, float, np.ndarray]:
    """
    Predict class for a single image.
    
    Args:
        model: Trained Keras model
        image_path: Path to the image file
        class_names: List of class names
        img_size: Image size expected by the model
        
    Returns:
        Tuple of (predicted_class, confidence, all_probabilities)
    """
    img_array = load_and_preprocess_image(image_path, (img_size, img_size))
    img_array = np.expand_dims(img_array, axis=0)
    
    prediction = model.predict(img_array, verbose=0)
    predicted_class = class_names[np.argmax(prediction)]
    confidence = np.max(prediction) * 100
    
    return predicted_class, confidence, prediction[0]


def display_sample_images(
    directory: Path,
    class_names: List[str] = CLASS_NAMES,
    num_samples: int = 3,
    figsize: Tuple[int, int] = (12, 16)
) -> None:
    """
    Display sample images from each class.
    
    Args:
        directory: Path to the dataset directory
        class_names: List of class names
        num_samples: Number of samples per class
        figsize: Figure size for matplotlib
    """
    import random
    
    fig, axes = plt.subplots(len(class_names), num_samples, figsize=figsize)
    
    for i, class_name in enumerate(class_names):
        class_path = directory / class_name
        images = list(class_path.glob('*'))
        samples = random.sample(images, min(num_samples, len(images)))
        
        for j, img_path in enumerate(samples):
            img = Image.open(img_path)
            axes[i, j].imshow(img)
            axes[i, j].axis('off')
            if j == 0:
                axes[i, j].set_title(class_name.upper(), fontsize=12, fontweight='bold')
    
    plt.suptitle('Sample Images from Each Class', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()


def plot_training_history(history: keras.callbacks.History) -> None:
    """
    Plot training and validation accuracy/loss.
    
    Args:
        history: Keras training history object
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Accuracy
    axes[0].plot(history.history['accuracy'], label='Training Accuracy', linewidth=2)
    axes[0].plot(history.history['val_accuracy'], label='Validation Accuracy', linewidth=2)
    axes[0].set_title('Model Accuracy', fontsize=14)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Loss
    axes[1].plot(history.history['loss'], label='Training Loss', linewidth=2)
    axes[1].plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
    axes[1].set_title('Model Loss', fontsize=14)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: List[str],
    figsize: Tuple[int, int] = (10, 8)
) -> None:
    """
    Plot confusion matrix using seaborn.
    
    Args:
        cm: Confusion matrix
        class_names: List of class names
        figsize: Figure size for matplotlib
    """
    import seaborn as sns
    
    plt.figure(figsize=figsize)
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names
    )
    plt.title('Confusion Matrix', fontsize=14, fontweight='bold')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    plt.show()


def create_data_generators(
    train_dir: Path,
    test_dir: Path,
    img_size: int = DEFAULT_IMG_SIZE,
    batch_size: int = 32,
    validation_split: float = 0.2
) -> Tuple:
    """
    Create training, validation, and test data generators.
    
    Args:
        train_dir: Path to training data directory
        test_dir: Path to test data directory
        img_size: Target image size
        batch_size: Batch size for generators
        validation_split: Fraction of training data for validation
        
    Returns:
        Tuple of (train_generator, validation_generator, test_generator)
    """
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    
    # Data augmentation for training
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest',
        validation_split=validation_split
    )
    
    # Only rescaling for test data
    test_datagen = ImageDataGenerator(rescale=1./255)
    
    # Create generators
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(img_size, img_size),
        batch_size=batch_size,
        class_mode='categorical',
        subset='training',
        shuffle=True
    )
    
    validation_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(img_size, img_size),
        batch_size=batch_size,
        class_mode='categorical',
        subset='validation',
        shuffle=False
    )
    
    test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=(img_size, img_size),
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False
    )
    
    return train_generator, validation_generator, test_generator


def get_callbacks(
    model_path: str = 'best_model.keras',
    patience_early_stop: int = 5,
    patience_lr: int = 3
) -> List[keras.callbacks.Callback]:
    """
    Get standard callbacks for training.
    
    Args:
        model_path: Path to save the best model
        patience_early_stop: Patience for early stopping
        patience_lr: Patience for learning rate reduction
        
    Returns:
        List of Keras callbacks
    """
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=patience_early_stop,
            restore_best_weights=True,
            verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=patience_lr,
            min_lr=1e-7,
            verbose=1
        ),
        keras.callbacks.ModelCheckpoint(
            model_path,
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        )
    ]
    return callbacks


if __name__ == '__main__':
    # Test the utility functions
    print("Intel Image Classification Utilities")
    print(f"Class names: {CLASS_NAMES}")
    print(f"Default image size: {DEFAULT_IMG_SIZE}")
