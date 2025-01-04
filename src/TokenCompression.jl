"""
    TokenCompression

A Julia package for efficient compression of token sequences using Byte Pair Encoding (BPE) with optional GPU acceleration.

# Features
- Byte Pair Encoding (BPE) implementation
- Token sequence compression
- Pattern detection and optimization
- Support for UInt32 token sequences
- High-performance model serialization
- Thread-safe operations
- GPU acceleration when available

# Example
```julia
using TokenCompression

# Create sample token sequence
tokens = UInt32[1000, 2000, 3000, 4000, 1000, 2000, 3000, 4000]

# Train BPE model and compress tokens
compressed = optimize_tokens(tokens)

# Train model separately for reuse
model = train_bpe(tokens)
compressed = optimize_tokens(tokens, model)

# Save model for later use
save_model(model, "trained_model.bson")

# Load model in another session
model = load_model("trained_model.bson")
```
"""
module TokenCompression

using Statistics
using ThreadsX
using CUDA
using LinearAlgebra

export optimize_tokens, TokenPattern, train_bpe, TokenPair
export save_model, load_model, decompress_tokens
export compress_batch, has_gpu

include("types.jl")
include("bpe.jl")
include("compression.jl")
include("serialization.jl")

# Re-export BPE functions
using .BPE: count_pairs, find_best_pair, merge_pair!

"""
    has_gpu()

Check if a CUDA-capable GPU is available and functional.
"""
function has_gpu()
    return CUDA.functional()
end

"""
    parallel_countmap(tokens::Vector{UInt32})

Count frequencies of tokens in parallel for large sequences.
Uses GPU if available, otherwise uses CPU threads.
"""
function parallel_countmap(tokens::Vector{UInt32})
    if length(tokens) < BPE.MIN_PARALLEL_SIZE
        # Sequential for small sequences
        counts = Dict{UInt32, Int}()
        for token in tokens
            counts[token] = get(counts, token, 0) + 1
        end
        return counts
    end
    
    if has_gpu()
        try
            # Move to GPU and count frequencies
            d_tokens = CuArray(tokens)
            d_counts = CUDA.zeros(Int, maximum(tokens))
            
            # Use atomic operations for counting
            CUDA.@sync for token in d_tokens
                CUDA.@atomic d_counts[token] += 1
            end
            
            # Convert back to dictionary
            counts = Dict{UInt32, Int}()
            h_counts = Array(d_counts)
            for i in 1:length(h_counts)
                if h_counts[i] > 0
                    counts[UInt32(i)] = h_counts[i]
                end
            end
            return counts
        catch e
            @warn "GPU counting failed, falling back to CPU" exception=e
            return parallel_countmap_cpu(tokens)
        end
    else
        return parallel_countmap_cpu(tokens)
    end
end

"""
    parallel_countmap_cpu(tokens::Vector{UInt32})

CPU fallback for parallel token counting.
"""
function parallel_countmap_cpu(tokens::Vector{UInt32})
    # Split into batches for parallel processing
    n_batches = max(1, div(length(tokens), BPE.BATCH_SIZE))
    batch_size = div(length(tokens), n_batches)
    
    # Count frequencies in parallel
    batch_counts = ThreadsX.map(1:n_batches) do i
        start_idx = (i-1) * batch_size + 1
        end_idx = i == n_batches ? length(tokens) : i * batch_size
        counts = Dict{UInt32, Int}()
        for token in view(tokens, start_idx:end_idx)
            counts[token] = get(counts, token, 0) + 1
        end
        counts
    end
    
    # Merge results
    merged = Dict{UInt32, Int}()
    for counts in batch_counts
        for (token, freq) in counts
            merged[token] = get(merged, token, 0) + freq
        end
    end
    
    return merged
end

"""
    compress_batch(tokens::Matrix{UInt32})

Compress a batch of token sequences in parallel.
Uses GPU acceleration if available.

# Arguments
- `tokens::Matrix{UInt32}`: Input token sequences, where each row is a sequence

# Returns
- `Matrix{UInt32}`: Compressed token sequences, where each row may have a different length
"""
function compress_batch(tokens::Matrix{UInt32})
    if has_gpu()
        try
            # Process in smaller batches to avoid memory issues
            batch_size = 1000
            num_batches = ceil(Int, size(tokens, 1) / batch_size)
            compressed_sequences = Vector{Vector{UInt32}}()
            
            for i in 1:num_batches
                start_idx = (i-1) * batch_size + 1
                end_idx = min(i * batch_size, size(tokens, 1))
                batch = tokens[start_idx:end_idx, :]
                
                # Move batch to GPU
                d_batch = CuArray(batch)
                
                # Process each row
                batch_compressed = mapslices(row -> 
                    Array(optimize_tokens(Array(row))), 
                    d_batch, dims=2)
                
                # Store compressed sequences
                for row in eachrow(batch_compressed)
                    push!(compressed_sequences, collect(row))
                end
            end
            
            # Find maximum length for padding
            max_length = maximum(length.(compressed_sequences))
            
            # Create result matrix with padding
            result = Matrix{UInt32}(undef, length(compressed_sequences), max_length)
            fill!(result, UInt32(0))  # Fill with zeros for padding
            
            # Copy compressed sequences with padding
            for (i, seq) in enumerate(compressed_sequences)
                result[i, 1:length(seq)] = seq
            end
            
            return result
        catch e
            @warn "GPU batch compression failed, falling back to CPU" exception=e
            return compress_batch_cpu(tokens)
        end
    else
        return compress_batch_cpu(tokens)
    end
end

"""
    compress_batch_cpu(tokens::Matrix{UInt32})

CPU fallback for batch compression.
"""
function compress_batch_cpu(tokens::Matrix{UInt32})
    # Compress each sequence
    compressed_sequences = Vector{Vector{UInt32}}()
    for row in eachrow(tokens)
        push!(compressed_sequences, optimize_tokens(collect(row)))
    end
    
    # Find maximum length for padding
    max_length = maximum(length.(compressed_sequences))
    
    # Create result matrix with padding
    result = Matrix{UInt32}(undef, length(compressed_sequences), max_length)
    fill!(result, UInt32(0))  # Fill with zeros for padding
    
    # Copy compressed sequences with padding
    for (i, seq) in enumerate(compressed_sequences)
        result[i, 1:length(seq)] = seq
    end
    
    return result
end

"""
    train_bpe(data::Vector{UInt32})

Train a BPE model on the given token sequence.
Uses parallel processing for large sequences.

# Arguments
- `data::Vector{UInt32}`: Input token sequence

# Returns
- `TokenPattern`: Trained compression model
"""
function train_bpe(data::Vector{UInt32})
    tokens = copy(data)
    vocab = parallel_countmap(tokens)
    merges = TokenPair[]
    
    # Iteratively find and merge most frequent pairs
    next_token = UInt32(maximum(keys(vocab)) + 1)
    while length(vocab) < VOCAB_SIZE
        # Count pair frequencies
        pairs = count_pairs(tokens)
        isempty(pairs) && break
        
        # Find best pair
        best_pair, freq = find_best_pair(pairs)
        freq < MIN_FREQUENCY && break
        
        # Record merge operation
        push!(merges, TokenPair(best_pair[1], best_pair[2], freq))
        
        # Apply merge operation
        merge_pair!(tokens, best_pair, next_token)
        vocab[next_token] = freq
        
        next_token = UInt32(next_token + 1)
    end
    
    return TokenPattern(vocab, merges, length(data), length(tokens))
end

end # module 