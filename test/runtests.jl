using Test
using TokenCompression

# Inclui os testes de performance GPU vs CPU
include("gpu_vs_cpu_tests.jl")

@testset "TokenCompression.jl" begin
    @testset "Functional Tests" begin
        @testset "Basic Compression" begin
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
        
        @testset "Edge Cases" begin
            # Empty sequence
            @test_throws ArgumentError optimize_tokens(UInt32[])
            
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
        
        @testset "Pattern Detection" begin
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
        
        @testset "Model Serialization" begin
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
        end
        
        @testset "Thread Safety" begin
            # Test concurrent model training and serialization
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
        
        @testset "Batch Processing" begin
            # Test with various batch sizes
            for batch_size in [10, 100]
                tokens = Matrix{UInt32}(undef, batch_size, 8)
                for i in 1:batch_size
                    tokens[i, :] = UInt32[1000, 2000, 3000, 4000, 1000, 2000, 3000, 4000]
                end
                
                compressed = compress_batch(tokens)
                @test size(compressed, 1) == batch_size
                @test size(compressed, 2) <= size(tokens, 2)
                
                # Verify each row is properly compressed
                for i in 1:batch_size
                    @test length(filter(!iszero, compressed[i, :])) < length(filter(!iszero, tokens[i, :]))
                end
            end
        end
    end
    
    @testset "Performance Tests" begin
        @testset "Sequential vs Parallel" begin
            # Generate large test sequence
            n_tokens = 100_000
            tokens = UInt32[]
            for i in 1:n_tokens
                if i % 2 == 0
                    push!(tokens, UInt32[1000, 2000]...)
                else
                    push!(tokens, UInt32[3000, 4000]...)
                end
            end
            
            # Compare sequential vs parallel processing
            sequential_time = @elapsed TokenCompression.count_pairs(tokens[1:1000])
            parallel_time = @elapsed TokenCompression.count_pairs(tokens)
            
            # Parallel should be relatively efficient for large sequences
            @test parallel_time < sequential_time * (n_tokens ÷ 1000)
            
            # Results should be consistent
            model_seq = train_bpe(tokens[1:1000])
            model_par = train_bpe(tokens)
            
            test_seq = vcat(UInt32[1000, 2000], UInt32[3000, 4000])
            compressed_seq = optimize_tokens(test_seq, model_seq)
            compressed_par = optimize_tokens(test_seq, model_par)
            
            @test length(compressed_seq) < length(test_seq)
            @test length(compressed_par) < length(test_seq)
        end
    end
end

# Exibe resumo dos resultados
println("\nTest Summary:")
println("=============")
println("✓ Functional tests completed")
println("  - Basic compression")
println("  - Edge cases")
println("  - Pattern detection")
println("  - Model serialization")
println("  - Thread safety")
println("  - Batch processing")
println("✓ Performance tests completed")
println("  - Sequential vs Parallel")
println("  - GPU vs CPU (when available)")
println("✓ All tests completed successfully") 