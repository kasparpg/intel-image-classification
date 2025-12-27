"""
Training script for Intel Image Classification.

Usage:
    python train.py --model mobilenet --epochs 30 --batch_size 32
"""

import argparse
from pathlib import Path
import tensorflow as tf
from tensorflow import keras

from models import get_model
from utils import (
    create_data_generators,
    get_callbacks,
    plot_training_history,
    CLASS_NAMES
)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train Intel Image Classifier')
    
    parser.add_argument(
        '--model', 
        type=str, 
        default='mobilenet',
        choices=['custom', 'mobilenet', 'resnet', 'efficientnet'],
        help='Model architecture to use'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=30,
        help='Number of training epochs'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=32,
        help='Batch size for training'
    )
    parser.add_argument(
        '--img_size',
        type=int,
        default=150,
        help='Image size (height and width)'
    )
    parser.add_argument(
        '--learning_rate',
        type=float,
        default=0.001,
        help='Initial learning rate'
    )
    parser.add_argument(
        '--data_dir',
        type=str,
        default='data',
        help='Path to data directory'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='models',
        help='Directory to save trained models'
    )
    
    return parser.parse_args()


def main():
    """Main training function."""
    args = parse_args()
    
    # Set up paths
    data_dir = Path(args.data_dir)
    train_dir = data_dir / 'seg_train'
    test_dir = data_dir / 'seg_test'
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Print configuration
    print("=" * 60)
    print("Intel Image Classification - Training")
    print("=" * 60)
    print(f"Model:         {args.model}")
    print(f"Epochs:        {args.epochs}")
    print(f"Batch Size:    {args.batch_size}")
    print(f"Image Size:    {args.img_size}x{args.img_size}")
    print(f"Learning Rate: {args.learning_rate}")
    print(f"Data Dir:      {data_dir}")
    print(f"Output Dir:    {output_dir}")
    print("=" * 60)
    
    # Check for GPU
    gpus = tf.config.list_physical_devices('GPU')
    print(f"\nGPUs Available: {len(gpus)}")
    if gpus:
        for gpu in gpus:
            print(f"  - {gpu}")
    
    # Create data generators
    print("\nCreating data generators...")
    train_gen, val_gen, test_gen = create_data_generators(
        train_dir=train_dir,
        test_dir=test_dir,
        img_size=args.img_size,
        batch_size=args.batch_size
    )
    
    print(f"Training samples:   {train_gen.samples}")
    print(f"Validation samples: {val_gen.samples}")
    print(f"Test samples:       {test_gen.samples}")
    
    # Build model
    print(f"\nBuilding {args.model} model...")
    model = get_model(
        model_type=args.model,
        input_shape=(args.img_size, args.img_size, 3),
        num_classes=len(CLASS_NAMES),
        learning_rate=args.learning_rate
    )
    
    print(f"Total parameters: {model.count_params():,}")
    
    # Set up callbacks
    model_path = output_dir / f'best_{args.model}_model.keras'
    callbacks = get_callbacks(str(model_path))
    
    # Train model
    print("\nStarting training...")
    print("-" * 60)
    
    history = model.fit(
        train_gen,
        epochs=args.epochs,
        validation_data=val_gen,
        callbacks=callbacks,
        verbose=1
    )
    
    # Evaluate on test set
    print("\n" + "=" * 60)
    print("Evaluation on Test Set")
    print("=" * 60)
    
    test_loss, test_accuracy = model.evaluate(test_gen, verbose=1)
    print(f"\nTest Loss:     {test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
    
    # Save final model
    final_model_path = output_dir / f'final_{args.model}_model.keras'
    model.save(str(final_model_path))
    print(f"\nFinal model saved to: {final_model_path}")
    
    # Plot training history
    print("\nGenerating training history plot...")
    plot_training_history(history)
    
    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)


if __name__ == '__main__':
    main()
