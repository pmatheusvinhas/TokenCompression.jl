"""
    TokenPair

Represents a pair of tokens that can be merged during BPE compression.

# Fields
- `first::UInt32`: First token in the pair
- `second::UInt32`: Second token in the pair
- `frequency::Int`: Number of times this pair appears in the sequence
"""
struct TokenPair
    first::UInt32
    second::UInt32
    frequency::Int
end

"""
    TokenPattern

Represents a trained compression model with vocabulary and merge rules.

# Fields
- `vocab::Dict{UInt32, Int}`: Mapping of tokens to their frequencies
- `merges::Vector{TokenPair}`: Ordered list of merge operations
- `original_size::Int`: Size of original token sequence
- `compressed_size::Int`: Size after compression
"""
struct TokenPattern
    vocab::Dict{UInt32, Int}
    merges::Vector{TokenPair}
    original_size::Int
    compressed_size::Int
end

# Constants
const VOCAB_SIZE = 1000
const MIN_FREQUENCY = 2 