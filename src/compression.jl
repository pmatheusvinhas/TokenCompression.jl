"""
    optimize_tokens(tokens::Vector{UInt32}, pattern::TokenPattern)

Compress token sequence using a trained model.
Uses GPU acceleration if available.

# Arguments
- `tokens::Vector{UInt32}`: Input token sequence
- `pattern::TokenPattern`: Trained compression model

# Returns
- `Vector{UInt32}`: Compressed token sequence

# Throws
- `ArgumentError`: If input sequence is empty
"""
function optimize_tokens(tokens::Vector{UInt32}, pattern::TokenPattern)
    if isempty(tokens)
        throw(ArgumentError("Cannot compress empty token sequence"))
    end
    
    if has_gpu() && length(tokens) > BPE.MIN_PARALLEL_SIZE
        try
            # Move data to GPU
            d_tokens = CuArray(copy(tokens))
            next_token = UInt32(maximum(keys(pattern.vocab)) + UInt32(1))
            
            # Apply merge operations in order
            for merge in pattern.merges
                # Create kernel for parallel merge operation
                function merge_kernel(d_tokens)
                    idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
                    stride = blockDim().x * gridDim().x
                    
                    for i = idx:stride:length(d_tokens)-1
                        if d_tokens[i] == merge.first && d_tokens[i+1] == merge.second
                            d_tokens[i] = next_token
                            d_tokens[i+1] = UInt32(0)  # Mark for removal
                        end
                    end
                    return nothing
                end
                
                # Launch kernel
                threads = 256
                blocks = cld(length(d_tokens), threads)
                CUDA.@sync @cuda threads=threads blocks=blocks merge_kernel(d_tokens)
                
                # Remove marked tokens
                d_tokens = CUDA.filter(x -> x != UInt32(0), d_tokens)
                next_token = next_token + UInt32(1)
            end
            
            return Array(d_tokens)
        catch e
            @warn "GPU optimization failed, falling back to CPU" exception=e
            return optimize_tokens_cpu(tokens, pattern)
        end
    else
        return optimize_tokens_cpu(tokens, pattern)
    end
end

"""
    optimize_tokens_cpu(tokens::Vector{UInt32}, pattern::TokenPattern)

CPU fallback for token sequence compression.
"""
function optimize_tokens_cpu(tokens::Vector{UInt32}, pattern::TokenPattern)
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
Uses GPU acceleration if available.

# Arguments
- `tokens::Vector{UInt32}`: Input token sequence

# Returns
- `Vector{UInt32}`: Compressed token sequence

# Throws
- `ArgumentError`: If input sequence is empty or too short for compression
"""
function optimize_tokens(tokens::Vector{UInt32})
    if isempty(tokens)
        throw(ArgumentError("Cannot compress empty token sequence"))
    end
    
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
Uses GPU acceleration if available.

# Arguments
- `compressed::Vector{UInt32}`: Compressed token sequence
- `pattern::TokenPattern`: Trained compression model used for compression

# Returns
- `Vector{UInt32}`: Reconstructed original token sequence

# Throws
- `ArgumentError`: If input sequence is empty
"""
function decompress_tokens(compressed::Vector{UInt32}, pattern::TokenPattern)
    if isempty(compressed)
        throw(ArgumentError("Cannot decompress empty token sequence"))
    end
    
    if has_gpu() && length(compressed) > BPE.MIN_PARALLEL_SIZE
        try
            # Create reverse mapping for merged tokens
            next_token = UInt32(maximum(keys(pattern.vocab)) + UInt32(1))
            reverse_merges = Dict{UInt32, Tuple{UInt32, UInt32}}()
            
            for merge in pattern.merges
                reverse_merges[next_token] = (merge.first, merge.second)
                next_token = next_token + UInt32(1)
            end
            
            # Move data to GPU
            d_compressed = CuArray(copy(compressed))
            d_result = CuArray(zeros(UInt32, length(compressed) * 2))  # Pre-allocate space for expansion
            d_length = CuArray([length(compressed)])  # Current length counter
            
            # Start from the last token used in compression
            current_token = next_token - UInt32(1)
            
            # Apply reverse merges in reverse order
            for merge in reverse(pattern.merges)
                # Create kernel for parallel decompression
                function decompress_kernel(d_result, d_length)
                    idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
                    stride = blockDim().x * gridDim().x
                    
                    for i = idx:stride:d_length[1]
                        if d_result[i] == current_token
                            d_result[i] = merge.first
                            # Atomically increment length and insert second token
                            pos = CUDA.atomic_add!(pointer(d_length), UInt32(1))
                            d_result[pos+1] = merge.second
                        end
                    end
                    return nothing
                end
                
                # Launch kernel
                threads = 256
                blocks = cld(length(d_compressed), threads)
                CUDA.@sync @cuda threads=threads blocks=blocks decompress_kernel(d_result, d_length)
                
                current_token = current_token - UInt32(1)
            end
            
            # Get final length and copy result
            final_length = Array(d_length)[1]
            return Array(view(d_result, 1:final_length))
        catch e
            @warn "GPU decompression failed, falling back to CPU" exception=e
            return decompress_tokens_cpu(compressed, pattern)
        end
    else
        return decompress_tokens_cpu(compressed, pattern)
    end
end

"""
    decompress_tokens_cpu(compressed::Vector{UInt32}, pattern::TokenPattern)

CPU fallback for token sequence decompression.
"""
function decompress_tokens_cpu(compressed::Vector{UInt32}, pattern::TokenPattern)
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