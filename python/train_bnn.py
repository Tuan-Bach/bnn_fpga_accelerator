#!/usr/bin/env python3
"""
Train Binary Neural Network on MNIST using Larq
"""

import os
import argparse
import numpy as np
import tensorflow as tf
import larq as lq
import larq_zoo as lqz

# Configure TensorFlow
tf.keras.backend.set_floatx('float32')

def build_bnn_model(input_shape=(28, 28, 1), num_classes=10):
    """Build a Binary Neural Network for MNIST"""
    
    # Input quantization to binary
    x_in = tf.keras.Input(shape=input_shape)
    x = lq.layers.QuantConv2D(
        32, (3, 3), padding='same', use_bias=False,
        input_quantizer="ste_sign", kernel_quantizer="ste_sign",
        kernel_constraint="weight_clip"
    )(x_in)
    x = tf.keras.layers.BatchNormalization(momentum=0.9, epsilon=1e-5)(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.MaxPool2D((2, 2))(x)
    
    x = lq.layers.QuantConv2D(
        64, (3, 3), padding='same', use_bias=False,
        kernel_quantizer="ste_sign",
        kernel_constraint="weight_clip"
    )(x)
    x = tf.keras.layers.BatchNormalization(momentum=0.9, epsilon=1e-5)(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.MaxPool2D((2, 2))(x)
    
    x = tf.keras.layers.Flatten()(x)
    x = lq.layers.QuantDense(
        512, use_bias=False,
        kernel_quantizer="ste_sign",
        kernel_constraint="weight_clip"
    )(x)
    x = tf.keras.layers.BatchNormalization(momentum=0.9, epsilon=1e-5)(x)
    x = tf.keras.layers.Activation('relu')(x)
    
    x = lq.layers.QuantDense(
        num_classes, use_bias=False,
        kernel_quantizer="ste_sign",
        kernel_constraint="weight_clip"
    )(x)
    x = tf.keras.layers.BatchNormalization(momentum=0.9, epsilon=1e-5)(x)
    x = tf.keras.layers.Activation('softmax')(x)
    
    model = tf.keras.Model(inputs=x_in, outputs=x)
    return model


def build_bnn_mlp(input_shape=(784,), num_classes=10):
    """Build a Binary MLP for MNIST (matches FPGA architecture)"""
    
    x_in = tf.keras.Input(shape=input_shape)
    
    # Layer 1: 784 -> 1024
    x = lq.layers.QuantDense(
        1024, use_bias=True,
        kernel_quantizer="ste_sign",
        kernel_constraint="weight_clip"
    )(x_in)
    x = tf.keras.layers.BatchNormalization(momentum=0.9, epsilon=1e-5)(x)
    x = tf.keras.layers.Activation('relu')(x)
    
    # Layer 2: 1024 -> 1024
    x = lq.layers.QuantDense(
        1024, use_bias=True,
        kernel_quantizer="ste_sign",
        kernel_constraint="weight_clip"
    )(x)
    x = tf.keras.layers.BatchNormalization(momentum=0.9, epsilon=1e-5)(x)
    x = tf.keras.layers.Activation('relu')(x)
    
    # Layer 3: 1024 -> 10
    x = lq.layers.QuantDense(
        num_classes, use_bias=True,
        kernel_quantizer="ste_sign",
        kernel_constraint="weight_clip"
    )(x)
    x = tf.keras.layers.BatchNormalization(momentum=0.9, epsilon=1e-5)(x)
    x = tf.keras.layers.Activation('softmax')(x)
    
    model = tf.keras.Model(inputs=x_in, outputs=x)
    return model


def load_mnist():
    """Load and preprocess MNIST dataset"""
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    
    # Normalize to [0, 1]
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0
    
    # Flatten for MLP
    x_train_flat = x_train.reshape(-1, 784)
    x_test_flat = x_test.reshape(-1, 784)
    
    # For CNN: add channel dimension
    x_train_cnn = x_train.reshape(-1, 28, 28, 1)
    x_test_cnn = x_test.reshape(-1, 28, 28, 1)
    
    # One-hot encode labels
    y_train_cat = tf.keras.utils.to_categorical(y_train, 10)
    y_test_cat = tf.keras.utils.to_categorical(y_test, 10)
    
    return (x_train_flat, y_train_cat), (x_test_flat, y_test_cat), \
           (x_train_cnn, y_train_cat), (x_test_cnn, y_test_cat)


def train_model(model, x_train, y_train, x_test, y_test, epochs=50, batch_size=128, lr=1e-3):
    """Train the BNN model"""
    
    # Custom learning rate schedule
    def lr_schedule(epoch):
        if epoch < 10:
            return lr
        elif epoch < 20:
            return lr * 0.5
        elif epoch < 30:
            return lr * 0.1
        else:
            return lr * 0.01
    
    callbacks = [
        tf.keras.callbacks.LearningRateScheduler(lr_schedule),
        tf.keras.callbacks.ModelCheckpoint(
            'bnn_best.h5', save_best_only=True, monitor='val_accuracy', mode='max'
        ),
        tf.keras.callbacks.EarlyStopping(patience=15, restore_best_weights=True),
        tf.keras.callbacks.TensorBoard(log_dir='./logs'),
        tf.keras.callbacks.CSVLogger('training_log.csv')
    ]
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print("Model Summary:")
    model.summary()
    lq.models.summary(model)
    
    history = model.fit(
        x_train, y_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=(x_test, y_test),
        callbacks=callbacks,
        verbose=1
    )
    
    return history


def evaluate_model(model, x_test, y_test):
    """Evaluate model on test set"""
    loss, acc = model.evaluate(x_test, y_test, verbose=0)
    print(f"Test Loss: {loss:.4f}, Test Accuracy: {acc:.4f}")
    return loss, acc


def main():
    parser = argparse.ArgumentParser(description='Train BNN on MNIST')
    parser.add_argument('--model', choices=['mlp', 'cnn'], default='mlp',
                        help='Model architecture')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--output', type=str, default='bnn_best.h5')
    args = parser.parse_args()
    
    # Load data
    print("Loading MNIST...")
    (x_train_flat, y_train), (x_test_flat, y_test), \
    (x_train_cnn, _), (x_test_cnn, _) = load_mnist()
    
    # Build model
    if args.model == 'mlp':
        model = build_bnn_mlp()
        x_train, x_test = x_train_flat, x_test_flat
    else:
        model = build_bnn_model()
        x_train, x_test = x_train_cnn, x_test_cnn
    
    # Train
    print(f"Training {args.model} for {args.epochs} epochs...")
    train_model(model, x_train, y_train, x_test, y_test, 
                epochs=args.epochs, batch_size=args.batch_size, lr=args.lr)
    
    # Load best weights
    model.load_weights('bnn_best.h5')
    
    # Evaluate
    evaluate_model(model, x_test, y_test)
    
    # Save final model
    model.save(args.output)
    print(f"Model saved to {args.output}")


if __name__ == '__main__':
    main()