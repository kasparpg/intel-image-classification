"""
Prediction script for Intel Image Classification.

Usage:
    python predict.py --model models/best_mobilenet_model.keras --image path/to/image.jpg
    python predict.py --model models/best_mobilenet_model.keras --folder data/seg_pred
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
import tensorflow as tf
from tensorflow import keras

from utils import CLASS_NAMES, DEFAULT_IMG_SIZE, predict_single_image


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Predict image classes')
    
    parser.add_argument(
        '--model',
        type=str,
        required=True,
        help='Path to trained model file'
    )
    parser.add_argument(
        '--image',
        type=str,
        default=None,
        help='Path to single image file'
    )
    parser.add_argument(
        '--folder',
        type=str,
        default=None,
        help='Path to folder containing images'
    )
    parser.add_argument(
        '--img_size',
        type=int,
        default=DEFAULT_IMG_SIZE,
        help='Image size (must match training)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output CSV file for predictions (optional)'
    )
    
    return parser.parse_args()


def predict_folder(model, folder_path, class_names, img_size):
    """
    Make predictions on all images in a folder.
    
    Args:
        model: Trained Keras model
        folder_path: Path to folder containing images
        class_names: List of class names
        img_size: Image size for model
        
    Returns:
        DataFrame with predictions
    """
    folder = Path(folder_path)
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif'}
    
    results = []
    
    # Find all images in folder (including subfolders)
    for img_path in folder.rglob('*'):
        if img_path.suffix.lower() in image_extensions:
            try:
                predicted_class, confidence, probs = predict_single_image(
                    model, str(img_path), class_names, img_size
                )
                
                result = {
                    'filename': img_path.name,
                    'path': str(img_path),
                    'predicted_class': predicted_class,
                    'confidence': confidence
                }
                
                # Add individual class probabilities
                for i, name in enumerate(class_names):
                    result[f'prob_{name}'] = probs[i] * 100
                
                results.append(result)
                
            except Exception as e:
                print(f"Error processing {img_path}: {e}")
    
    return pd.DataFrame(results)


def main():
    """Main prediction function."""
    args = parse_args()
    
    if args.image is None and args.folder is None:
        print("Error: Please provide either --image or --folder")
        return
    
    # Load model
    print(f"Loading model from: {args.model}")
    model = keras.models.load_model(args.model)
    print("Model loaded successfully!")
    
    if args.image:
        # Single image prediction
        print(f"\nPredicting: {args.image}")
        
        predicted_class, confidence, probs = predict_single_image(
            model, args.image, CLASS_NAMES, args.img_size
        )
        
        print("\n" + "=" * 50)
        print("Prediction Results")
        print("=" * 50)
        print(f"Predicted Class: {predicted_class}")
        print(f"Confidence:      {confidence:.2f}%")
        print("\nClass Probabilities:")
        for i, name in enumerate(CLASS_NAMES):
            bar = "█" * int(probs[i] * 20)
            print(f"  {name:12s}: {probs[i]*100:5.2f}% {bar}")
    
    elif args.folder:
        # Folder prediction
        print(f"\nProcessing folder: {args.folder}")
        
        results_df = predict_folder(model, args.folder, CLASS_NAMES, args.img_size)
        
        if len(results_df) == 0:
            print("No images found in folder.")
            return
        
        print(f"\nProcessed {len(results_df)} images")
        print("\n" + "=" * 80)
        print("Prediction Summary")
        print("=" * 80)
        
        # Display summary
        class_counts = results_df['predicted_class'].value_counts()
        print("\nClass Distribution:")
        for class_name, count in class_counts.items():
            pct = count / len(results_df) * 100
            print(f"  {class_name:12s}: {count:4d} ({pct:.1f}%)")
        
        # Display detailed results
        print("\nDetailed Results:")
        print("-" * 80)
        display_cols = ['filename', 'predicted_class', 'confidence']
        print(results_df[display_cols].to_string(index=False))
        
        # Save to CSV if requested
        if args.output:
            results_df.to_csv(args.output, index=False)
            print(f"\nResults saved to: {args.output}")
        
        # Statistics
        print("\n" + "=" * 80)
        print("Statistics")
        print("=" * 80)
        print(f"Average Confidence: {results_df['confidence'].mean():.2f}%")
        print(f"Min Confidence:     {results_df['confidence'].min():.2f}%")
        print(f"Max Confidence:     {results_df['confidence'].max():.2f}%")


if __name__ == '__main__':
    main()
