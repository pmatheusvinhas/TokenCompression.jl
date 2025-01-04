# TokenCompression.jl

A Julia package for efficient compression of token sequences using Byte Pair Encoding (BPE). Designed for compressing large sequences of tokens, particularly useful for LLM applications and token-based data processing.

## Features

- Efficient Byte Pair Encoding (BPE) implementation optimized for token sequences
- Support for UInt32 tokens (compatible with most LLM tokenizers)
- Advanced pattern detection and optimization
- GPU acceleration for large sequences when available
- Automatic fallback to CPU when GPU is unavailable or for small sequences
- Parallel processing for large sequences
- High-performance model serialization
- Thread-safe operations
- Configurable vocabulary size and frequency thresholds
- Trained models can be saved and reused

## Installation

```julia
using Pkg
Pkg.add("TokenCompression")
```

## Quick Start

```julia
using TokenCompression

# Create sample token sequence
tokens = UInt32[1000, 2000, 3000, 4000, 1000, 2000, 3000, 4000]

# Train BPE model and compress tokens in one step
compressed = optimize_tokens(tokens)

# Or train model separately for reuse
model = train_bpe(tokens)
compressed = optimize_tokens(tokens, model)

# Save model for later use
save_model(model, "trained_model.bson")

# Load model in another session
loaded_model = load_model("trained_model.bson")
```

## Detailed Usage

### Training a BPE Model

```julia
# Create token sequence
tokens = Vector{UInt32}([...])

# Train model with default parameters
model = train_bpe(tokens)

# Access model information
println("Vocabulary size: ", length(model.vocab))
println("Original size: ", model.original_size)
println("Compressed size: ", model.compressed_size)
println("Compression ratio: ", model.compressed_size/model.original_size)

# Model contains:
# - vocab: Dictionary mapping tokens to frequencies
# - merges: Vector of TokenPair containing merge operations
# - original_size: Size of input sequence
# - compressed_size: Size after compression
```

### Compressing Token Sequences

```julia
# Compress using trained model
compressed = optimize_tokens(tokens, model)

# Compress new sequence with same patterns
new_tokens = Vector{UInt32}([...])
compressed_new = optimize_tokens(new_tokens, model)

# Decompress back to original sequence
decompressed = decompress_tokens(compressed, model)
@assert decompressed == tokens  # Original sequence is perfectly reconstructed
```

### GPU Acceleration

The package automatically detects CUDA-capable GPUs and uses them for acceleration when available:

```julia
# Check if GPU is available
has_gpu()

# Process large sequences - automatically uses GPU if available
large_tokens = rand(UInt32, 1_000_000)
compressed = optimize_tokens(large_tokens)  # Uses GPU if available and beneficial

# Process batches of sequences with GPU acceleration
batch = rand(UInt32, (1000, 1000))  # 1000 sequences of length 1000
compressed_batch = compress_batch(batch)  # Uses GPU if available
```

The package automatically falls back to CPU processing in the following cases:
- No CUDA-capable GPU is available
- The input sequence is too small to benefit from GPU acceleration
- GPU processing fails for any reason

### Model Serialization

```julia
# Save trained model
save_model(model, "model.bson")

# Load model in another session
loaded_model = load_model("model.bson")

# Models can be shared and reused across different sessions
compressed = optimize_tokens(tokens, loaded_model)
```

## Configuration

The package provides several configuration constants that can be tuned for specific use cases:

- `VOCAB_SIZE`: Maximum size of the vocabulary (default: 1000)
  - Larger values allow more patterns but increase memory usage
  - Smaller values provide more aggressive compression
  
- `MIN_FREQUENCY`: Minimum frequency for pattern merging (default: 2)
  - Higher values focus on more common patterns
  - Lower values catch rare but potentially important patterns

- `BATCH_SIZE`: Size of batches for parallel processing (default: 10,000)
  - Can be adjusted based on available memory and CPU/GPU cores

- `MIN_PARALLEL_SIZE`: Minimum sequence length for parallel processing (default: 1,000)
  - Sequences shorter than this are processed sequentially

## Performance Optimization

The implementation includes several optimizations:

### Memory Efficiency
- In-place operations where possible
- Efficient data structures for pattern storage
- Minimal temporary allocations
- Smart GPU memory management

### Speed
- GPU acceleration for large sequences
- Parallel processing for CPU operations
- SIMD operations for token manipulation
- Efficient pattern detection algorithms
- Automatic device selection based on input size

### Compression Ratio
- Typically achieves 25-50% reduction in sequence length
- Better ratios for sequences with repeating patterns
- Lossless compression (perfect reconstruction)
- Consistent ratios across CPU and GPU processing

## Use Cases

### LLM Token Sequence Compression
- Compress token sequences from LLM outputs
- Reduce storage requirements for token datasets
- Efficient token pattern analysis
- High-throughput batch processing

### General Token Processing
- Compress any UInt32 token sequences
- Pattern detection in numerical sequences
- Token vocabulary optimization
- Large-scale sequence processing

### Model Sharing
- Train compression models on specific domains
- Share models across different applications
- Standardize compression patterns
- Cross-platform compatibility

## Thread Safety

The package is designed to be thread-safe:
- Parallel processing for large sequences
- Thread-safe model training and compression
- Concurrent model serialization
- Safe GPU resource management

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. Areas of interest include:

- Additional compression algorithms
- Performance optimizations
- New features and use cases
- Documentation improvements
- Test coverage expansion
- GPU optimization improvements

## License

This package is licensed under the MIT License - see the LICENSE file for details. 