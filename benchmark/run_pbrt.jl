# pbrt-v4 Benchmark Runner
#
# Runs pbrt-v4 on scenes that have .pbrt counterparts for comparison.
#
# Usage:
#   include("RayDemo/benchmark/run_pbrt.jl")
#   run_pbrt_benchmarks(gpu_name="amd_7900xtx")

using JSON3, Dates, OrderedCollections

include(joinpath(@__DIR__, "run_benchmarks.jl"))

const PBRT_BINARY = "/sim/Programmieren/RayTracing/pbrt-v4/build/pbrt"

"""
    run_pbrt_benchmarks(; gpu_name, n_warmup, n_trials, wavefront, output_dir)

Run pbrt-v4 on benchmark scenes that have .pbrt files.
Shells out to the pbrt binary and times execution.

# Arguments
- `gpu_name::String`: GPU identifier for result file naming
- `n_warmup::Int`: warmup runs (default: 1)
- `n_trials::Int`: timed runs (default: 3)
- `wavefront::Bool`: use `--wavefront` flag for GPU rendering (default: false)
- `output_dir::String`: where to save JSON results
"""
function run_pbrt_benchmarks(;
    gpu_name::String = "unknown",
    n_warmup::Int = 1,
    n_trials::Int = 3,
    wavefront::Bool = false,
    output_dir::String = BENCHMARK_RESULTS_DIR,
)
    mkpath(output_dir)

    if !isfile(PBRT_BINARY)
        error("pbrt-v4 binary not found at $PBRT_BINARY")
    end

    variant = wavefront ? "wavefront" : "cpu"
    backend_name = "pbrt_v4_$(variant)"

    results = OrderedDict{String, Any}()
    results["metadata"] = OrderedDict(
        "gpu_name" => gpu_name,
        "backend_name" => backend_name,
        "wavefront" => wavefront,
        "n_warmup" => n_warmup,
        "n_trials" => n_trials,
        "timestamp" => Dates.format(now(), "yyyy-mm-dd HH:MM:SS"),
        "pbrt_binary" => PBRT_BINARY,
    )
    results["scenes"] = OrderedDict{String, Any}()

    for (scene_name, cfg) in PBRT_SCENES
        if !isfile(cfg.pbrt_file)
            @warn "pbrt file not found: $(cfg.pbrt_file), skipping $scene_name"
            continue
        end

        println("\n" * "="^60)
        println("pbrt-v4 scene: $scene_name ($(variant))")
        println("  file: $(cfg.pbrt_file)")
        println("  spp: $(cfg.spp)")
        println("="^60)

        outfile = joinpath(output_dir, "$(scene_name)_pbrt_$(variant).exr")

        cmd_parts = [PBRT_BINARY]
        wavefront && push!(cmd_parts, "--wavefront")
        push!(cmd_parts, "--spp", string(cfg.spp))
        push!(cmd_parts, "--quiet")
        push!(cmd_parts, "--outfile", outfile)
        push!(cmd_parts, cfg.pbrt_file)

        # Warmup
        println("\n  Warmup ($n_warmup runs)...")
        for i in 1:n_warmup
            try
                run(Cmd(cmd_parts))
            catch e
                @warn "  Warmup $i failed" exception=(e, catch_backtrace())
            end
        end

        # Timed trials
        println("\n  Benchmarking ($n_trials trials)...")
        timings = Float64[]
        for trial in 1:n_trials
            t = @elapsed begin
                run(Cmd(cmd_parts))
            end
            push!(timings, t)
            println("    Trial $trial: $(round(t, digits=3))s")
        end

        sorted_timings = sort(timings)
        scene_result = OrderedDict(
            "spp" => cfg.spp,
            "pbrt_file" => cfg.pbrt_file,
            "wavefront" => wavefront,
            "timings" => timings,
            "median" => round(median(sorted_timings), digits=4),
            "min" => round(minimum(timings), digits=4),
            "max" => round(maximum(timings), digits=4),
        )
        results["scenes"][scene_name] = scene_result

        println("  Median: $(scene_result["median"])s  Min: $(scene_result["min"])s")
    end

    # Save JSON
    json_path = joinpath(output_dir, "$(gpu_name)_$(backend_name).json")
    open(json_path, "w") do io
        JSON3.pretty(io, results)
    end
    println("\nResults saved to $json_path")
    return results
end
