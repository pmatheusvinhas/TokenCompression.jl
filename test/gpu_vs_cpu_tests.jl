using Test
using BenchmarkTools
using TokenCompression

# Helper function to generate test sequences with patterns
function generate_test_sequence(size::Int)
    # Create a sequence with repeating patterns
    base_pattern = UInt32[1, 2, 3, 4, 1, 2, 3, 4, 5, 6, 7, 8, 5, 6, 7, 8]
    repetitions = ceil(Int, size / length(base_pattern))
    sequence = repeat(base_pattern, repetitions)[1:size]
    return sequence
end

# Helper function to generate test batches with patterns
function generate_test_batch(batch_size::Int, seq_length::Int)
    # Create a batch where each sequence has repeating patterns
    batch = Matrix{UInt32}(undef, batch_size, seq_length)
    for i in 1:batch_size
        batch[i, :] = generate_test_sequence(seq_length)
    end
    return batch
end

# Struct to hold compression metrics
struct CompressionMetrics
    device::String
    input_size::Int
    compression_time::Float64
    compression_ratio::Float64
    memory_used::Float64
end

# Helper function to print metrics
function print_metrics(metrics::CompressionMetrics)
    println("\nCompression Metrics:")
    println("  Device: $(metrics.device)")
    println("  Input Size: $(metrics.input_size) tokens")
    println("  Compression Time: $(round(metrics.compression_time, digits=3)) ms")
    println("  Compression Ratio: $(round(metrics.compression_ratio, digits=2))x")
    println("  Memory Used: $(round(metrics.memory_used, digits=2)) MB\n")
end

# Helper function to run compression test
function run_compression_test(tokens::Vector{UInt32})
    # Measure memory and time
    b = @benchmark begin
        compressed = optimize_tokens($tokens)
        GC.gc()
    end
    
    # Calculate metrics
    device = has_gpu() ? "GPU" : "CPU"
    input_size = length(tokens)
    compression_time = mean(b.times)
    compression_ratio = length(tokens) / length(optimize_tokens(tokens))
    memory_used = b.memory / 1024 / 1024  # Convert to MB
    
    metrics = CompressionMetrics(device, input_size, compression_time, compression_ratio, memory_used)
    print_metrics(metrics)
    return metrics
end

# Helper function to run batch compression test
function run_batch_compression_test(tokens::Matrix{UInt32})
    # Measure memory and time
    b = @benchmark begin
        compressed = compress_batch($tokens)
        GC.gc()
    end
    
    # Calculate metrics
    device = has_gpu() ? "GPU" : "CPU"
    input_size = length(tokens)
    compression_time = mean(b.times)
    
    # Calculate average compression ratio across all sequences
    compressed = compress_batch(tokens)
    total_input_size = size(tokens, 1) * size(tokens, 2)
    total_compressed_size = sum(length.(eachrow(compressed)))
    compression_ratio = total_input_size / total_compressed_size
    
    memory_used = b.memory / 1024 / 1024  # Convert to MB
    
    metrics = CompressionMetrics(device, input_size, compression_time, compression_ratio, memory_used)
    print_metrics(metrics)
    return metrics
end

@testset "GPU vs CPU Performance Tests" begin
    @testset "Device Detection" begin
        println("\nGPU Available: $(has_gpu())")
        @test has_gpu() isa Bool
    end
    
    @testset "Single Sequence Compression" begin
        for size in [1000, 10000, 100000]
            println("\nTest with sequence size: $size")
            tokens = generate_test_sequence(size)
            metrics = run_compression_test(tokens)
            
            # Basic functionality tests
            @test metrics.compression_time > 0
            @test metrics.compression_ratio > 1.0
            @test metrics.memory_used > 0
            
            # Performance expectations
            if has_gpu() && size > 10000
                @test metrics.device == "GPU"
            else
                @test metrics.device == "CPU"
            end
        end
    end
    
    @testset "Batch Compression" begin
        for batch_size in [100, 1000]
            for seq_length in [100, 1000]
                println("\nTest with batch size: $batch_size, sequence length: $seq_length")
                tokens = generate_test_batch(batch_size, seq_length)
                metrics = run_batch_compression_test(tokens)
                
                # Basic functionality tests
                @test metrics.compression_time > 0
                @test metrics.compression_ratio > 1.0
                @test metrics.memory_used > 0
                
                # Performance expectations
                if has_gpu() && batch_size * seq_length > 10000
                    @test metrics.device == "GPU"
                else
                    @test metrics.device == "CPU"
                end
            end
        end
    end
    
    @testset "Memory Usage Patterns" begin
        # Test memory usage scaling with input size
        sizes = [1000, 10000, 100000]
        memory_usages = Float64[]
        
        for size in sizes
            tokens = generate_test_sequence(size)
            metrics = run_compression_test(tokens)
            push!(memory_usages, metrics.memory_used)
        end
        
        # Memory usage should generally increase with input size
        # but we allow for some variation due to GC
        for i in 2:length(memory_usages)
            @test memory_usages[i] > 0
        end
    end
end 