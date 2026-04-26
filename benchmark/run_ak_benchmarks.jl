# AcceleratedKernels Benchmark Runner
#
# Benchmarks AK operations (sort, reduce, accumulate, map) across backends.
# Simple @elapsed timing (not BenchmarkTools) for cross-backend comparison.
#
# Usage:
#   using Lava, AMDGPU  # load backends first
#   include("RayDemo/benchmark/run_ak_benchmarks.jl")
#   run_ak_benchmarks()

using JSON3, Dates, OrderedCollections
import AcceleratedKernels as AK
using KernelAbstractions
using StableRNGs

const AK_RESULTS_DIR = joinpath(@__DIR__, "results")

# =============================================================================
# Benchmark definitions
# =============================================================================

# Each benchmark: (name, setup_fn, bench_fn)
# setup_fn(AT, n, rng) -> data...
# bench_fn(data...) -> nothing (mutates in place or returns)

function ak_benchmarks(AT, n, rng, sync_fn)
    benchmarks = OrderedDict{String, Any}()

    # --- Sort ---
    for T in [UInt32, Float32]
        label = "sort/$T"
        v_orig = AT(rand(rng, T, n))

        v = copy(v_orig)
        AK.sort!(v)  # warmup
        sync_fn()

        times = Float64[]
        for _ in 1:3
            v = copy(v_orig)
            sync_fn()
            t = @elapsed begin; AK.sort!(v); sync_fn(); end
            push!(times, t)
        end
        benchmarks[label * "/acck"] = _stats(times)

        if AT === Array
            v = copy(Array(v_orig))
            sort!(v)  # warmup
            times = Float64[]
            for _ in 1:3
                v = copy(Array(v_orig))
                t = @elapsed sort!(v)
                push!(times, t)
            end
            benchmarks[label * "/base"] = _stats(times)
        end
    end

    # --- Reduce ---
    for T in [UInt32, Float32]
        label = "reduce/$T"
        v = AT(rand(rng, T, n))

        AK.reduce(+, v; init=zero(T))  # warmup
        sync_fn()

        times = Float64[]
        for _ in 1:5
            t = @elapsed begin; AK.reduce(+, v; init=zero(T)); sync_fn(); end
            push!(times, t)
        end
        benchmarks[label * "/acck"] = _stats(times)

        if AT === Array
            Base.reduce(+, Array(v); init=zero(T))
            times = Float64[]
            for _ in 1:5
                t = @elapsed Base.reduce(+, Array(v); init=zero(T))
                push!(times, t)
            end
            benchmarks[label * "/base"] = _stats(times)
        end
    end

    # --- Accumulate (prefix sum) ---
    for T in [UInt32, Float32]
        label = "accumulate/$T"
        v = AT(rand(rng, T, n))

        AK.accumulate(+, v; init=zero(T))  # warmup
        sync_fn()

        times = Float64[]
        for _ in 1:5
            t = @elapsed begin; AK.accumulate(+, v; init=zero(T)); sync_fn(); end
            push!(times, t)
        end
        benchmarks[label * "/acck"] = _stats(times)

        if AT === Array
            Base.accumulate(+, Array(v))
            times = Float64[]
            for _ in 1:5
                t = @elapsed Base.accumulate(+, Array(v))
                push!(times, t)
            end
            benchmarks[label * "/base"] = _stats(times)
        end
    end

    # --- Map ---
    for T in [Float32]
        label = "map/$T"
        v = AT(rand(rng, T, n))
        out = similar(v)

        AK.map!(x -> 2x, out, v)  # warmup
        sync_fn()

        times = Float64[]
        for _ in 1:5
            t = @elapsed begin; AK.map!(x -> 2x, out, v); sync_fn(); end
            push!(times, t)
        end
        benchmarks[label * "/acck_2x"] = _stats(times)

        AK.map!(sin, out, v)  # warmup
        sync_fn()
        times = Float64[]
        for _ in 1:5
            t = @elapsed begin; AK.map!(sin, out, v); sync_fn(); end
            push!(times, t)
        end
        benchmarks[label * "/acck_sin"] = _stats(times)
    end

    # --- MapReduce ---
    for T in [Float32]
        label = "mapreduce/$T"
        v = AT(rand(rng, T, n))

        AK.mapreduce(sin, +, v; init=zero(T))  # warmup
        sync_fn()

        times = Float64[]
        for _ in 1:5
            t = @elapsed begin; AK.mapreduce(sin, +, v; init=zero(T)); sync_fn(); end
            push!(times, t)
        end
        benchmarks[label * "/acck_sin"] = _stats(times)
    end

    # --- SortPerm ---
    for T in [UInt32, Float32]
        label = "sortperm/$T"
        v_orig = AT(rand(rng, T, n))
        ix = AT(collect(UInt32(1):UInt32(n)))

        try
            v = copy(v_orig)
            ix .= AT(collect(UInt32(1):UInt32(n)))
            AK.sortperm!(ix, v)  # warmup
            sync_fn()

            times = Float64[]
            for _ in 1:3
                v = copy(v_orig)
                ix .= AT(collect(UInt32(1):UInt32(n)))
                sync_fn()
                t = @elapsed begin; AK.sortperm!(ix, v); sync_fn(); end
                push!(times, t)
            end
            benchmarks[label * "/acck"] = _stats(times)
        catch e
            benchmarks[label * "/acck"] = OrderedDict("error" => sprint(showerror, e)[1:min(end,100)])
        end
    end

    return benchmarks
end

function _stats(times)
    sorted = sort(times)
    OrderedDict(
        "median_ms" => round(sorted[div(length(sorted)+1, 2)] * 1000, digits=3),
        "min_ms" => round(minimum(times) * 1000, digits=3),
        "timings_ms" => round.(times .* 1000, digits=3),
    )
end

# =============================================================================
# Runner
# =============================================================================

"""
    run_ak_benchmarks(; n, backends, platform, gpu_name, cpu_name)

Run AcceleratedKernels benchmarks on all detected backends.
Results saved as `{platform}_{device}_ak_{backend}.json`.
GPU backends use gpu_name, CPU backends use cpu_name in the filename.

Requires run_benchmarks.jl to be included first (provides detect_platform, etc.).
"""
function run_ak_benchmarks(;
    n::Int = 1_000_000,
    backends = nothing,
    platform::String = detect_platform(),
    gpu_name::String = detect_gpu_name(),
    cpu_name::String = detect_cpu_name(),
    version::String = "v1",
    force::Bool = false,
)
    mkpath(AK_RESULTS_DIR)

    if backends === nothing
        backends = _detect_ak_backends()
    end

    all_results = OrderedDict{String, Any}()
    n_short = n >= 1_000_000 ? "$(div(n, 1_000_000))m" : "$(div(n, 1_000))k"

    for (name, AT, sync_fn) in backends
        println("\n" * "#"^60)
        println("# AK Benchmarks: $name ($n elements)")
        println("#"^60)

        # CPU backends use cpu_name, GPU backends use gpu_name
        device_name = name in ("cpu", "abacus") ? cpu_name : gpu_name
        prefix = "$(platform)_$(device_name)_ak_$(n_short)_$(name)"

        if should_skip(AK_RESULTS_DIR, prefix, version; force)
            continue
        end

        rng = StableRNG(123)
        try
            benchmarks = ak_benchmarks(AT, n, rng, sync_fn)

            results = OrderedDict{String, Any}()
            results["metadata"] = OrderedDict(
                "platform" => platform,
                "gpu_name" => gpu_name,
                "cpu_name" => cpu_name,
                "device_name" => device_name,
                "backend" => name,
                "n_elements" => n,
                "version" => version,
                "timestamp" => Dates.format(now(), "yyyy-mm-dd HH:MM:SS"),
                "julia_version" => string(VERSION),
                "cpu_threads" => Sys.CPU_THREADS,
            )
            results["benchmarks"] = benchmarks

            all_results[name] = results

            json_path = joinpath(AK_RESULTS_DIR, result_filename(prefix, version))
            open(json_path, "w") do io
                JSON3.pretty(io, results)
            end
            println("Saved: $json_path")

            # Print results
            println("\n  Results:")
            for (k, v) in benchmarks
                if haskey(v, "median_ms")
                    println("    $(rpad(k, 30)) $(v["median_ms"]) ms")
                end
            end
        catch e
            @warn "Backend $name failed" exception=(e, catch_backtrace())
        end
    end

    # Print comparison
    _print_ak_comparison(all_results)
    return all_results
end

function _detect_ak_backends()
    backends = Tuple[]

    if isdefined(Main, :Lava)
        Lava = getfield(Main, :Lava)
        try
            Lava.LavaArray(Float32[0])
            push!(backends, ("lava", Lava.LavaArray, () -> Lava.vk_flush!(Lava.vk_context())))
            @info "AK backend: Lava"
        catch; end
    end

    if isdefined(Main, :AMDGPU)
        AMDGPU = getfield(Main, :AMDGPU)
        try
            if length(AMDGPU.devices()) > 0
                push!(backends, ("amdgpu", AMDGPU.ROCArray, () -> AMDGPU.synchronize()))
                @info "AK backend: AMDGPU"
            end
        catch; end
    end

    if isdefined(Main, :Abacus)
        Abacus = getfield(Main, :Abacus)
        try
            Abacus.AbacusVector(Float32[0])
            push!(backends, ("abacus", Abacus.AbacusVector, () -> KernelAbstractions.synchronize(Abacus.AbacusBackend())))
            @info "AK backend: Abacus"
        catch; end
    end

    if isdefined(Main, :CUDA)
        CUDA = getfield(Main, :CUDA)
        try
            if CUDA.functional()
                push!(backends, ("cuda", CUDA.CuArray, () -> CUDA.synchronize()))
                @info "AK backend: CUDA"
            end
        catch; end
    end

    push!(backends, ("cpu", Array, () -> nothing))
    @info "AK backend: CPU"

    return backends
end

function _print_ak_comparison(all_results)
    println("\n" * "="^70)
    println("AK BENCHMARK COMPARISON (median ms, 1M elements)")
    println("="^70)

    backends = collect(keys(all_results))
    print(rpad("Operation", 30))
    for b in backends
        print(rpad(b, 12))
    end
    println()
    println("-"^(30 + 12 * length(backends)))

    # Collect all benchmark names
    all_names = OrderedCollections.OrderedSet{String}()
    for (_, r) in all_results
        for k in keys(r["benchmarks"])
            push!(all_names, k)
        end
    end

    for name in all_names
        print(rpad(name, 30))
        for b in backends
            bench = get(all_results[b]["benchmarks"], name, nothing)
            if bench !== nothing && haskey(bench, "median_ms")
                print(rpad("$(bench["median_ms"])", 12))
            else
                print(rpad("-", 12))
            end
        end
        println()
    end
end
