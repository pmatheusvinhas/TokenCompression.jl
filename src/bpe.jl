"""
BPE implementation with parallel processing support.
"""
module BPE

using ThreadsX

# Constants for parallel processing
const BATCH_SIZE = 10_000
const MIN_PARALLEL_SIZE = 1_000

"""
    count_pairs(tokens::Vector{UInt32})

Count frequencies of adjacent token pairs in a sequence.
Uses parallel processing for large sequences.

# Arguments
- `tokens::Vector{UInt32}`: Input token sequence

# Returns
- `Dict{Tuple{UInt32, UInt32}, Int}`: Mapping of token pairs to their frequencies
"""
function count_pairs(tokens::Vector{UInt32})
    if length(tokens) < MIN_PARALLEL_SIZE
        return _count_pairs_sequential(tokens)
    end
    
    # Split into batches for parallel processing
    n_batches = max(1, div(length(tokens), BATCH_SIZE))
    batch_size = div(length(tokens), n_batches)
    
    # Count pairs in parallel
    batch_counts = ThreadsX.map(1:n_batches) do i
        start_idx = (i-1) * batch_size + 1
        end_idx = i == n_batches ? length(tokens) : i * batch_size
        _count_pairs_sequential(view(tokens, start_idx:end_idx))
    end
    
    # Merge results
    return _merge_pair_counts(batch_counts)
end

"""
    _count_pairs_sequential(tokens::AbstractVector{UInt32})

Sequential implementation of pair counting for smaller sequences.
"""
function _count_pairs_sequential(tokens::AbstractVector{UInt32})
    pairs = Dict{Tuple{UInt32, UInt32}, Int}()
    for i in 1:length(tokens)-1
        pair = (tokens[i], tokens[i+1])
        pairs[pair] = get(pairs, pair, 0) + 1
    end
    return pairs
end

"""
    _merge_pair_counts(counts::Vector{Dict{Tuple{UInt32, UInt32}, Int}})

Merge multiple pair count dictionaries efficiently.
"""
function _merge_pair_counts(counts::Vector{Dict{Tuple{UInt32, UInt32}, Int}})
    merged = Dict{Tuple{UInt32, UInt32}, Int}()
    for count_dict in counts
        for (pair, freq) in count_dict
            merged[pair] = get(merged, pair, 0) + freq
        end
    end
    return merged
end

"""
    find_best_pair(pairs::Dict{Tuple{UInt32, UInt32}, Int})

Find the most frequent pair of tokens using parallel reduction for large dictionaries.
"""
function find_best_pair(pairs::Dict{Tuple{UInt32, UInt32}, Int})
    if length(pairs) < MIN_PARALLEL_SIZE
        return _find_best_pair_sequential(pairs)
    end
    
    # Convert to array for parallel processing
    pairs_array = collect(pairs)
    
    # Find best pair in parallel
    best = ThreadsX.reduce(pairs_array; init=nothing) do x, y
        if x === nothing
            return y
        elseif y === nothing
            return x
        end
        return last(x) > last(y) ? x : y
    end
    
    return best === nothing ? (nothing, 0) : (best[1], best[2])
end

"""
    _find_best_pair_sequential(pairs::Dict{Tuple{UInt32, UInt32}, Int})

Sequential implementation of finding the best pair.
"""
function _find_best_pair_sequential(pairs::Dict{Tuple{UInt32, UInt32}, Int})
    best_freq = 0
    best_pair = nothing
    
    for (pair, freq) in pairs
        if freq > best_freq
            best_freq = freq
            best_pair = pair
        end
    end
    
    return best_pair, best_freq
end

"""
    merge_pair!(tokens::Vector{UInt32}, pair::Tuple{UInt32, UInt32}, new_token::UInt32)

Replace all occurrences of a token pair with a new token.
Uses parallel processing for large sequences.
"""
function merge_pair!(tokens::Vector{UInt32}, pair::Tuple{UInt32, UInt32}, new_token::UInt32)
    if length(tokens) < MIN_PARALLEL_SIZE
        return _merge_pair_sequential!(tokens, pair, new_token)
    end
    
    # Find all merge positions in parallel
    positions = ThreadsX.findall(1:length(tokens)-1) do i
        tokens[i] == pair[1] && tokens[i+1] == pair[2]
    end
    
    # Apply merges from end to start to maintain indices
    for i in reverse(positions)
        tokens[i] = new_token
        deleteat!(tokens, i+1)
    end
    
    return tokens
end

"""
    _merge_pair_sequential!(tokens::Vector{UInt32}, pair::Tuple{UInt32, UInt32}, new_token::UInt32)

Sequential implementation of pair merging.
"""
function _merge_pair_sequential!(tokens::Vector{UInt32}, pair::Tuple{UInt32, UInt32}, new_token::UInt32)
    i = 1
    while i < length(tokens)
        if i < length(tokens) && tokens[i] == pair[1] && tokens[i+1] == pair[2]
            tokens[i] = new_token
            deleteat!(tokens, i+1)
        else
            i += 1
        end
    end
    return tokens
end

# Export public functions
export count_pairs, find_best_pair, merge_pair!

end # module BPE 