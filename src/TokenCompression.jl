"""
    TokenCompression

A Julia package for efficient compression of token sequences using Byte Pair Encoding (BPE).

# Features
- Byte Pair Encoding (BPE) implementation
- Token sequence compression
- Pattern detection and optimization
- Support for UInt32 token sequences
- High-performance model serialization
- Thread-safe operations

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

export optimize_tokens, TokenPattern, train_bpe, TokenPair
export save_model, load_model, decompress_tokens

include("types.jl")
include("bpe.jl")
include("compression.jl")
include("serialization.jl")

# Re-export BPE functions
using .BPE: count_pairs, find_best_pair, merge_pair!

"""
    parallel_countmap(tokens::Vector{UInt32})

Count frequencies of tokens in parallel for large sequences.
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