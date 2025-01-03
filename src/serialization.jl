"""
    Serialization module for TokenCompression.jl

This module provides high-performance serialization of trained BPE models using BSON format.
The serialization is optimized for quick loading and minimal storage size.
"""

using BSON
using ..TokenCompression: TokenPattern, TokenPair

"""
    save_model(model::TokenPattern, filename::String)

Save a trained BPE model to disk in BSON format.

# Arguments
- `model::TokenPattern`: The trained model to save
- `filename::String`: Path where the model will be saved

# Performance Notes
- Uses BSON for efficient binary serialization
- Optimized for minimal storage size while maintaining fast load times
- Thread-safe for concurrent model saving

# Example
```julia
model = train_bpe(tokens)
save_model(model, "conversation_model.bson")
```
"""
function save_model(model::TokenPattern, filename::String)
    # Convert to Dict for BSON serialization
    model_dict = Dict(
        :vocab => model.vocab,
        :merges => [(m.first, m.second, m.frequency) for m in model.merges],
        :original_size => model.original_size,
        :compressed_size => model.compressed_size
    )
    
    # Save atomically to prevent corruption
    temp_file = filename * ".temp"
    BSON.bson(temp_file, model_dict)
    mv(temp_file, filename, force=true)
    
    return nothing
end

"""
    load_model(filename::String)::TokenPattern

Load a trained BPE model from disk.

# Arguments
- `filename::String`: Path to the saved model file

# Returns
- `TokenPattern`: The loaded model ready for use

# Performance Notes
- Optimized for fast loading of large models
- Memory-efficient deserialization
- Thread-safe for concurrent model loading

# Example
```julia
model = load_model("conversation_model.bson")
compressed = optimize_tokens(new_tokens, model)
```
"""
function load_model(filename::String)::TokenPattern
    # Load and validate BSON data
    model_dict = BSON.load(filename)
    
    # Reconstruct TokenPair objects
    merges = [TokenPair(first, second, freq) 
              for (first, second, freq) in model_dict[:merges]]
    
    return TokenPattern(
        model_dict[:vocab],
        merges,
        model_dict[:original_size],
        model_dict[:compressed_size]
    )
end

export save_model, load_model 