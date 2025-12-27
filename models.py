"""
Model architectures for Intel Image Classification.
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2, ResNet50, VGG16, EfficientNetB0


def build_custom_cnn(
    input_shape: tuple = (150, 150, 3),
    num_classes: int = 6
) -> keras.Model:
    """
    Build a custom CNN model from scratch.
    
    Args:
        input_shape: Input image shape (height, width, channels)
        num_classes: Number of output classes
        
    Returns:
        Compiled Keras model
    """
    model = models.Sequential([
        # First Conv Block
        layers.Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=input_shape),
        layers.BatchNormalization(),
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Second Conv Block
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Third Conv Block
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Fourth Conv Block
        layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Fully Connected Layers
        layers.Flatten(),
        layers.Dense(512, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    return model


def build_mobilenet_model(
    input_shape: tuple = (150, 150, 3),
    num_classes: int = 6,
    trainable_base: bool = False
) -> keras.Model:
    """
    Build a transfer learning model using MobileNetV2.
    
    Args:
        input_shape: Input image shape (height, width, channels)
        num_classes: Number of output classes
        trainable_base: Whether to allow training of base model layers
        
    Returns:
        Keras model
    """
    base_model = MobileNetV2(
        weights='imagenet',
        include_top=False,
        input_shape=input_shape
    )
    base_model.trainable = trainable_base
    
    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(512, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    return model


def build_resnet_model(
    input_shape: tuple = (150, 150, 3),
    num_classes: int = 6,
    trainable_base: bool = False
) -> keras.Model:
    """
    Build a transfer learning model using ResNet50.
    
    Args:
        input_shape: Input image shape (height, width, channels)
        num_classes: Number of output classes
        trainable_base: Whether to allow training of base model layers
        
    Returns:
        Keras model
    """
    base_model = ResNet50(
        weights='imagenet',
        include_top=False,
        input_shape=input_shape
    )
    base_model.trainable = trainable_base
    
    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(512, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    return model


def build_efficientnet_model(
    input_shape: tuple = (150, 150, 3),
    num_classes: int = 6,
    trainable_base: bool = False
) -> keras.Model:
    """
    Build a transfer learning model using EfficientNetB0.
    
    Args:
        input_shape: Input image shape (height, width, channels)
        num_classes: Number of output classes
        trainable_base: Whether to allow training of base model layers
        
    Returns:
        Keras model
    """
    base_model = EfficientNetB0(
        weights='imagenet',
        include_top=False,
        input_shape=input_shape
    )
    base_model.trainable = trainable_base
    
    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(512, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    return model


def compile_model(
    model: keras.Model,
    learning_rate: float = 0.001,
    optimizer: str = 'adam'
) -> keras.Model:
    """
    Compile a Keras model with standard settings.
    
    Args:
        model: Keras model to compile
        learning_rate: Learning rate for optimizer
        optimizer: Optimizer type ('adam', 'sgd', 'rmsprop')
        
    Returns:
        Compiled Keras model
    """
    if optimizer == 'adam':
        opt = keras.optimizers.Adam(learning_rate=learning_rate)
    elif optimizer == 'sgd':
        opt = keras.optimizers.SGD(learning_rate=learning_rate, momentum=0.9)
    elif optimizer == 'rmsprop':
        opt = keras.optimizers.RMSprop(learning_rate=learning_rate)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer}")
    
    model.compile(
        optimizer=opt,
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model


def get_model(
    model_type: str = 'mobilenet',
    input_shape: tuple = (150, 150, 3),
    num_classes: int = 6,
    learning_rate: float = 0.001
) -> keras.Model:
    """
    Factory function to get a compiled model.
    
    Args:
        model_type: Type of model ('custom', 'mobilenet', 'resnet', 'efficientnet')
        input_shape: Input image shape
        num_classes: Number of output classes
        learning_rate: Learning rate for optimizer
        
    Returns:
        Compiled Keras model
    """
    model_builders = {
        'custom': build_custom_cnn,
        'mobilenet': build_mobilenet_model,
        'resnet': build_resnet_model,
        'efficientnet': build_efficientnet_model
    }
    
    if model_type not in model_builders:
        raise ValueError(f"Unknown model type: {model_type}. Choose from {list(model_builders.keys())}")
    
    model = model_builders[model_type](input_shape, num_classes)
    model = compile_model(model, learning_rate)
    
    return model


if __name__ == '__main__':
    # Test model creation
    print("Testing model architectures...")
    
    for model_type in ['custom', 'mobilenet', 'resnet', 'efficientnet']:
        print(f"\nBuilding {model_type} model...")
        model = get_model(model_type)
        print(f"  Total parameters: {model.count_params():,}")
