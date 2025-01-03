using Test
using TokenCompression

@testset "TokenCompression.jl" begin
    @testset "Basic functionality" begin
        # Test data
        tokens = UInt32[1000, 2000, 3000, 4000, 1000, 2000, 3000, 4000]
        
        # Test compression
        compressed = optimize_tokens(tokens)
        @test length(compressed) < length(tokens)
        @test length(unique(compressed)) <= length(unique(tokens))
        
        # Test model training
        model = train_bpe(tokens)
        @test !isempty(model.vocab)
        @test !isempty(model.merges)
        @test model.original_size == length(tokens)
        @test model.compressed_size < model.original_size
        
        # Test compression with trained model
        compressed2 = optimize_tokens(tokens, model)
        @test length(compressed2) < length(tokens)

        # Test decompression
        decompressed = decompress_tokens(compressed2, model)
        @test length(decompressed) == length(tokens)
        @test decompressed == tokens
    end
    
    @testset "Edge cases" begin
        # Empty sequence
        @test optimize_tokens(UInt32[]) == UInt32[]
        
        # Single token
        single = UInt32[1000]
        @test optimize_tokens(single) == single
        
        # No repeating patterns
        unique_tokens = UInt32[1000, 2000, 3000, 4000]
        @test length(optimize_tokens(unique_tokens)) == length(unique_tokens)

        # Test decompression of uncompressed sequence
        model = train_bpe(unique_tokens)
        compressed = optimize_tokens(unique_tokens, model)
        decompressed = decompress_tokens(compressed, model)
        @test decompressed == unique_tokens
    end
    
    @testset "Pattern detection" begin
        # Create sequence with known patterns
        pattern1 = UInt32[1000, 2000]
        pattern2 = UInt32[3000, 4000]
        tokens = vcat(repeat(pattern1, 5), repeat(pattern2, 5))
        
        # Train model
        model = train_bpe(tokens)
        
        # Verify patterns were detected
        @test length(model.merges) >= 2
        @test model.compressed_size < model.original_size
        
        # Compress new sequence with same patterns
        new_tokens = vcat(pattern1, pattern2)
        compressed = optimize_tokens(new_tokens, model)
        @test length(compressed) < length(new_tokens)

        # Test pattern reconstruction
        decompressed = decompress_tokens(compressed, model)
        @test decompressed == new_tokens
    end

    @testset "Compression-Decompression cycle" begin
        # Test with various sequence lengths
        for len in [10, 100, 1000]
            # Generate test sequence with some patterns
            tokens = UInt32[]
            for i in 1:len
                push!(tokens, UInt32(rand(1:5) * 1000))
            end

            # Train model and compress
            model = train_bpe(tokens)
            compressed = optimize_tokens(tokens, model)

            # Decompress and verify
            decompressed = decompress_tokens(compressed, model)
            @test length(decompressed) == length(tokens)
            @test decompressed == tokens
        end
    end

    @testset "Model serialization" begin
        # Create and train model
        tokens = UInt32[1000, 2000, 3000, 4000, 1000, 2000, 3000, 4000]
        model = train_bpe(tokens)
        compressed = optimize_tokens(tokens, model)
        
        # Save model
        temp_file = tempname() * ".bson"
        save_model(model, temp_file)
        
        # Load model
        loaded_model = load_model(temp_file)
        
        # Verify loaded model produces same results
        compressed2 = optimize_tokens(tokens, loaded_model)
        @test compressed == compressed2
        
        # Test model attributes preserved
        @test model.vocab == loaded_model.vocab
        @test length(model.merges) == length(loaded_model.merges)
        @test model.original_size == loaded_model.original_size
        @test model.compressed_size == loaded_model.compressed_size
        
        # Clean up
        rm(temp_file)
        
        # Test concurrent access
        models = Vector{TokenPattern}()
        Threads.@threads for i in 1:10
            tokens = UInt32[UInt32(j * i * 1000) for j in 1:8]
            model = train_bpe(tokens)
            file = tempname() * ".bson"
            save_model(model, file)
            loaded = load_model(file)
            push!(models, loaded)
            rm(file)
        end
        @test length(models) == 10
    end

    @testset "Parallel processing" begin
        # Test with large sequence
        n_tokens = 100_000
        tokens = UInt32[]
        # Generate sequence with known patterns
        for i in 1:n_tokens
            # Add some repeating patterns
            if i % 2 == 0
                push!(tokens, UInt32[1000, 2000]...)
            else
                push!(tokens, UInt32[3000, 4000]...)
            end
        end

        # Time sequential vs parallel processing
        sequential_time = @elapsed begin
            TokenCompression.count_pairs(tokens[1:1000])  # Force sequential
        end

        parallel_time = @elapsed begin
            TokenCompression.count_pairs(tokens)  # Should use parallel
        end

        # Test parallel processing is faster for large sequences
        @test parallel_time < sequential_time * (n_tokens ÷ 1000)

        # Test results are consistent
        model_seq = train_bpe(tokens[1:1000])  # Sequential
        model_par = train_bpe(tokens)  # Parallel

        # Compress same sequence with both models
        test_seq = vcat(UInt32[1000, 2000], UInt32[3000, 4000])
        compressed_seq = optimize_tokens(test_seq, model_seq)
        compressed_par = optimize_tokens(test_seq, model_par)

        # Results should be consistent with patterns
        @test length(compressed_seq) < length(test_seq)
        @test length(compressed_par) < length(test_seq)

        # Test thread safety
        Threads.@threads for _ in 1:10
            local_tokens = copy(tokens)
            model = train_bpe(local_tokens)
            compressed = optimize_tokens(test_seq, model)
            @test length(compressed) < length(test_seq)
        end
    end
end 