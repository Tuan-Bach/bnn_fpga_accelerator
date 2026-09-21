#!/usr/bin/env python3
"""
Export BNN weights to hex format for FPGA weight ROM initialization
"""

import argparse
import numpy as np
import tensorflow as tf
import larq as lq


def binarize_weights(weights):
    """Convert float weights to binary {+1, -1} using sign function"""
    return np.sign(weights).astype(np.int8)


def pack_bits(binary_weights, bit_width=64):
    """
    Pack binary weights (0/1) into bit-packed words.
    Input: array of 0s and 1s
    Output: array of 64-bit integers
    """
    # Convert {+1, -1} to {1, 0}
    binary_01 = (binary_weights > 0).astype(np.uint8)
    
    # Pad to multiple of bit_width
    total_bits = binary_01.size
    padded_bits = ((total_bits + bit_width - 1) // bit_width) * bit_width
    if padded_bits > total_bits:
        binary_01 = np.pad(binary_01, (0, padded_bits - total_bits), mode='constant')
    
    # Reshape and pack
    binary_01 = binary_01.reshape(-1, bit_width)
    packed = np.zeros(binary_01.shape[0], dtype=np.uint64)
    
    for i in range(bit_width):
        packed = (packed << 1) | binary_01[:, i]
    
    return packed


def export_dense_layer(weights, biases, layer_name, output_dir, bit_width=64):
    """Export a dense layer's weights and biases"""
    
    # weights shape: (input_dim, output_dim)
    # biases shape: (output_dim,)
    
    print(f"\nExporting {layer_name}...")
    print(f"  Weight shape: {weights.shape}")
    print(f"  Bias shape: {biases.shape}")
    
    # Binarize weights
    bin_weights = binarize_weights(weights)
    print(f"  Weight stats: min={bin_weights.min()}, max={bin_weights.max()}")
    print(f"  Sparsity (zeros): {(bin_weights == 0).mean()*100:.1f}%")
    
    # Binarize biases
    bin_biases = binarize_weights(biases)
    print(f"  Bias stats: min={bin_biases.min()}, max={bin_biases.max()}")
    
    # Pack weights: each output neuron gets input_dim/bit_width words
    input_dim, output_dim = weights.shape
    words_per_neuron = (input_dim + bit_width - 1) // bit_width
    
    all_packed = []
    for neuron_idx in range(output_dim):
        neuron_weights = bin_weights[:, neuron_idx]
        packed = pack_bits(neuron_weights, bit_width)
        all_packed.extend(packed)
    
    # Save weights
    weight_file = f"{output_dir}/{layer_name}_weights.mem"
    with open(weight_file, 'w') as f:
        for w in all_packed:
            f.write(f"{w:016X}\n")
    print(f"  Saved {len(all_packed)} words to {weight_file}")
    
    # Save biases (one per neuron, Q4.12 format)
    bias_file = f"{output_dir}/{layer_name}_biases.mem"
    with open(bias_file, 'w') as f:
        for b in bin_biases:
            # Convert {-1, 1} to Q4.12: -1 -> 0xF000, 1 -> 0x1000
            q4_12 = 0x1000 if b > 0 else 0xF000
            f.write(f"{q4_12:04X}\n")
    print(f"  Saved {len(bin_biases)} biases to {bias_file}")
    
    return all_packed, bin_biases


def export_model(model_path, output_dir):
    """Export all layers from a trained BNN model"""
    
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Loading model from {model_path}...")
    model = tf.keras.models.load_model(model_path, compile=False)
    
    print("\nModel layers:")
    for i, layer in enumerate(model.layers):
        print(f"  [{i}] {layer.name}: {layer.__class__.__name__}")
        if hasattr(layer, 'weights') and layer.weights:
            for w in layer.weights:
                print(f"      {w.name}: {w.shape}")
    
    # Find QuantDense layers
    dense_layers = []
    for layer in model.layers:
        if isinstance(layer, lq.layers.QuantDense):
            dense_layers.append(layer)
    
    print(f"\nFound {len(dense_layers)} QuantDense layers")
    
    all_weights = []
    all_biases = []
    
    for i, layer in enumerate(dense_layers):
        weights = layer.get_weights()
        if len(weights) == 2:
            w, b = weights
            packed_w, packed_b = export_dense_layer(w, b, f"layer{i}", output_dir)
            all_weights.extend(packed_w)
            all_biases.extend(packed_b)
        elif len(weights) == 1:
            w = weights[0]
            packed_w, _ = export_dense_layer(w, np.zeros(w.shape[1]), f"layer{i}", output_dir)
            all_weights.extend(packed_w)
    
    # Create combined weight file for all layers
    combined_file = f"{output_dir}/weights_all.mem"
    with open(combined_file, 'w') as f:
        for w in all_weights:
            f.write(f"{w:016X}\n")
    print(f"\nCombined weights: {len(all_weights)} words -> {combined_file}")
    
    # Generate layer config for FPGA
    config_file = f"{output_dir}/layer_config.txt"
    with open(config_file, 'w') as f:
        f.write("# Layer configuration for FPGA\n")
        f.write("# Format: layer_idx input_features output_features weight_offset threshold\n")
        offset = 0
        for i, layer in enumerate(dense_layers):
            weights = layer.get_weights()[0]
            input_dim, output_dim = weights.shape
            words_per_neuron = (input_dim + 63) // 64
            total_words = words_per_neuron * output_dim
            
            # Threshold: 0 for now (will be calibrated)
            threshold = 0
            
            f.write(f"{i} {input_dim} {output_dim} {offset} {threshold}\n")
            offset += total_words
    
    print(f"Layer config saved to {config_file}")
    
    # Print summary
    print(f"\n=== Export Summary ===")
    print(f"Total weight words: {len(all_weights)}")
    print(f"Total biases: {len(all_biases)}")
    print(f"Estimated BRAM: {len(all_weights) * 64 / 8 / 1024:.1f} KB")


def generate_test_vectors(model_path, output_dir, num_samples=10):
    """Generate test input vectors and expected outputs"""
    
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\nGenerating {num_samples} test vectors...")
    
    model = tf.keras.models.load_model(model_path, compile=False)
    
    # Load MNIST test data
    (_, _), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_test = x_test.astype('float32') / 255.0
    x_test_flat = x_test.reshape(-1, 784)
    
    # Select random samples
    indices = np.random.choice(len(x_test), num_samples, replace=False)
    test_inputs = x_test_flat[indices]
    test_labels = np.argmax(y_test[indices], axis=1)
    
    # Get model predictions
    predictions = model.predict(test_inputs, verbose=0)
    pred_labels = np.argmax(predictions, axis=1)
    
    # Binarize inputs for FPGA (threshold at 0.5)
    bin_inputs = (test_inputs > 0.5).astype(np.uint8)
    
    # Save input vectors (packed)
    input_file = f"{output_dir}/test_inputs.mem"
    with open(input_file, 'w') as f:
        for sample_idx in range(num_samples):
            sample = bin_inputs[sample_idx]
            packed = pack_bits(sample, 64)
            for w in packed:
                f.write(f"{w:016X}\n")
    print(f"  Saved {num_samples} test inputs to {input_file}")
    
    # Save expected outputs
    expected_file = f"{output_dir}/test_expected.txt"
    with open(expected_file, 'w') as f:
        for i in range(num_samples):
            f.write(f"Sample {i}: label={test_labels[i]}, pred={pred_labels[i]}, match={test_labels[i]==pred_labels[i]}\n")
    print(f"  Saved expected outputs to {expected_file}")
    
    accuracy = (test_labels == pred_labels).mean()
    print(f"  Model accuracy on test samples: {accuracy*100:.1f}%")


def pack_bits(binary_array, bit_width=64):
    """Pack binary array into words"""
    total_bits = binary_array.size
    padded_bits = ((total_bits + bit_width - 1) // bit_width) * bit_width
    if padded_bits > total_bits:
        binary_array = np.pad(binary_array, (0, padded_bits - total_bits), mode='constant')
    
    binary_array = binary_array.reshape(-1, bit_width)
    packed = np.zeros(binary_array.shape[0], dtype=np.uint64)
    
    for i in range(bit_width):
        packed = (packed << 1) | binary_array[:, i]
    
    return packed


def main():
    parser = argparse.ArgumentParser(description='Export BNN weights for FPGA')
    parser.add_argument('--model', type=str, default='bnn_best.h5',
                        help='Path to trained model')
    parser.add_argument('--output-dir', type=str, default='../rtl/weights',
                        help='Output directory for weight files')
    parser.add_argument('--test-vectors', type=int, default=10,
                        help='Number of test vectors to generate')
    args = parser.parse_args()
    
    export_model(args.model, args.output_dir)
    generate_test_vectors(args.model, args.output_dir, args.test_vectors)


if __name__ == '__main__':
    main()