"""
    optimize_tokens(tokens::Vector{UInt32}, pattern::TokenPattern)

Compress token sequence using a trained model.

# Arguments
- `tokens::Vector{UInt32}`: Input token sequence
- `pattern::TokenPattern`: Trained compression model

# Returns
- `Vector{UInt32}`: Compressed token sequence
"""
function optimize_tokens(tokens::Vector{UInt32}, pattern::TokenPattern)
    result = copy(tokens)
    next_token = UInt32(maximum(keys(pattern.vocab)) + UInt32(1))
    
    # Apply merge operations in order
    for merge in pattern.merges
        merge_pair!(result, (merge.first, merge.second), next_token)
        next_token = next_token + UInt32(1)
    end
    
    return result
end

"""
    optimize_tokens(tokens::Vector{UInt32})

Train model and compress token sequence in one step.

# Arguments
- `tokens::Vector{UInt32}`: Input token sequence

# Returns
- `Vector{UInt32}`: Compressed token sequence
"""
function optimize_tokens(tokens::Vector{UInt32})
    if length(tokens) < MIN_FREQUENCY
        return tokens
    end
    
    # Train model
    pattern = train_bpe(tokens)
    
    # Apply compression
    return optimize_tokens(tokens, pattern)
end

"""
    decompress_tokens(compressed::Vector{UInt32}, pattern::TokenPattern)

Reconstruct the original token sequence from compressed tokens using the trained model.

# Arguments
- `compressed::Vector{UInt32}`: Compressed token sequence
- `pattern::TokenPattern`: Trained compression model used for compression

# Returns
- `Vector{UInt32}`: Reconstructed original token sequence
"""
function decompress_tokens(compressed::Vector{UInt32}, pattern::TokenPattern)
    # Create reverse mapping for merged tokens
    next_token = UInt32(maximum(keys(pattern.vocab)) + UInt32(1))
    reverse_merges = Dict{UInt32, Tuple{UInt32, UInt32}}()
    
    # Build reverse mapping
    for merge in pattern.merges
        reverse_merges[next_token] = (merge.first, merge.second)
        next_token = next_token + UInt32(1)
    end
    
    # Reconstruct original sequence
    result = copy(compressed)
    
    # Start from the last token used in compression
    current_token = next_token - UInt32(1)
    
    # Apply reverse merges in reverse order
    for merge in reverse(pattern.merges)
        # Find all occurrences of the current token
        i = 1
        while i <= length(result)
            if result[i] == current_token
                # Replace merged token with original pair
                deleteat!(result, i)
                insert!(result, i, merge.first)
                insert!(result, i+1, merge.second)
                i += 2
            else
                i += 1
            end
        end
        current_token = current_token - UInt32(1)
    end
    
    return result
end

export decompress_tokens 