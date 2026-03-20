# RayDemo Benchmark Suite
#
# Auto-detects available backends and benchmarks all scenes.
# Measures only render time (colorbuffer), NOT scene creation.
#
# Usage:
#   include("RayDemo/benchmark/run_benchmarks.jl")
#   run_all_benchmarks()                    # auto-detect everything
#   run_all_benchmarks(scenes=["materials"]) # specific scenes
#
# Or manually:
#   run_julia_benchmarks(backend=Lava.LavaBackend(), backend_name="lava_sw")
#   run_pbrt_benchmarks(pbrt_mode="cpu")    # or "gpu" for OptiX

using JSON3, Dates, OrderedCollections
using RayMakie, Hikari, Makie

const RAYDEMO_DIR = joinpath(@__DIR__, "..")
const BENCHMARK_RESULTS_DIR = joinpath(@__DIR__, "results")

# =============================================================================
# Scene Registry
# =============================================================================

const BENCHMARK_SCENES = OrderedDict(
    "crown" => (
        script = joinpath(RAYDEMO_DIR, "Crown", "crown.jl"),
        resolution = (500, 700),
        samples = 16,
        max_depth = 12,
        colorbuffer_kwargs = (; exposure=1.0f0, tonemap=:aces, gamma=2.2f0),
        sensor_kwargs = (; iso=10, exposure_time=1.0, white_balance=4000),
        # pbrt scene: run from this directory so relative paths in .pbrt work
        pbrt_dir = joinpath(RAYDEMO_DIR, "Crown"),
        pbrt_file = "crown.pbrt",
    ),
    "bunny_cloud" => (
        script = joinpath(RAYDEMO_DIR, "Volumes", "bunny_cloud.jl"),
        resolution = (960, 540),
        samples = 8,
        max_depth = 50,
        colorbuffer_kwargs = (; exposure=0.5, tonemap=nothing, gamma=2.2f0),
        sensor_kwargs = (; iso=50f0, white_balance=5000),
        pbrt_dir = joinpath(RAYDEMO_DIR, "Volumes"),
        pbrt_file = "bunny-cloud.pbrt",
    ),
    "killeroo_gold" => (
        script = joinpath(RAYDEMO_DIR, "KillerooGold", "killeroo_gold.jl"),
        resolution = (684, 513),
        samples = 32,
        max_depth = 8,
        colorbuffer_kwargs = (; exposure=1.0f0, tonemap=:aces, gamma=2.2f0),
        sensor_kwargs = (; iso=100, white_balance=5500),
        pbrt_dir = joinpath(RAYDEMO_DIR, "KillerooGold"),
        pbrt_file = "killeroo-gold.pbrt",
    ),
    "materials" => (
        script = joinpath(RAYDEMO_DIR, "Materials", "materials.jl"),
        resolution = (1200, 900),
        samples = 10,
        max_depth = 50,
        colorbuffer_kwargs = (; exposure=0.6f0, tonemap=:aces, gamma=2.2f0),
        sensor_kwargs = (; iso=50, exposure_time=1.0, white_balance=0),
        pbrt_dir = joinpath(RAYDEMO_DIR, "Materials"),
        pbrt_file = "materials.pbrt",
    ),
    "black_hole" => (
        script = joinpath(RAYDEMO_DIR, "BlackHole", "black_hole.jl"),
        resolution = (800, 450),
        samples = 32,
        max_depth = 100,
        colorbuffer_kwargs = (;),
        sensor_kwargs = (; iso=30, white_balance=5500),
        pbrt_dir = nothing,
        pbrt_file = nothing,
    ),
)

# =============================================================================
# System Detection
# =============================================================================

"""Return (hostname, gpu_name) for result file naming."""
function detect_system()
    hostname = strip(read(`hostname`, String))
    gpu_name = "unknown_gpu"

    # Try Lava context first (most reliable when loaded)
    try
        gpu_name = Main.Lava.vk_context().device_name
    catch; end

    # Try NVIDIA
    if gpu_name == "unknown_gpu"
        try
            gpu_name = strip(read(`nvidia-smi --query-gpu=name --format=csv,noheader`, String))
        catch; end
    end

    # Try vulkaninfo
    if gpu_name == "unknown_gpu"
        try
            info = read(pipeline(`vulkaninfo --summary`, stderr=devnull), String)
            m = match(r"deviceName\s*=\s*(.*)", info)
            m !== nothing && (gpu_name = strip(m.captures[1]))
        catch; end
    end

    # Sanitize for filename
    gpu_name = replace(lowercase(gpu_name), r"[^a-z0-9]+" => "_")
    gpu_name = strip(gpu_name, '_')
    return String(hostname), String(gpu_name)
end

"""
Detect available backends. Returns list of (name, config) pairs.

Checks what GPU packages are loaded in Main. Load them before calling:
  `using Lava`    for Vulkan
  `using AMDGPU`  for AMD
  `using CUDA`    for NVIDIA
"""
function detect_backends()
    backends = Pair{String, Any}[]

    # Lava (Vulkan) - check if loaded
    if isdefined(Main, :Lava)
        try
            Lava = getfield(Main, :Lava)
            backend = Lava.LavaBackend()
            push!(backends, "lava_sw" => (; backend, hw_accel=false))
            try
                ctx = Lava.vk_context()
                if ctx.rt_pipeline_properties !== nothing
                    push!(backends, "lava_hw" => (; backend, hw_accel=true))
                end
            catch; end
            @info "Detected: Lava (Vulkan)"
        catch e
            @info "Lava loaded but failed: $(sprint(showerror, e)[1:min(end,80)])"
        end
    end

    # AMDGPU
    if isdefined(Main, :AMDGPU)
        try
            AMDGPU = getfield(Main, :AMDGPU)
            if length(AMDGPU.devices()) > 0
                push!(backends, "amdgpu" => (; backend=AMDGPU.ROCBackend(), hw_accel=false))
                @info "Detected: AMDGPU"
            end
        catch e
            @info "AMDGPU loaded but failed: $(sprint(showerror, e)[1:min(end,80)])"
        end
    end

    # Abacus (CPU via Vulkan SPIR-V)
    if isdefined(Main, :Abacus)
        try
            Abacus = getfield(Main, :Abacus)
            push!(backends, "abacus" => (; backend=Abacus.AbacusBackend(), hw_accel=false))
            @info "Detected: Abacus (CPU)"
        catch e
            @info "Abacus loaded but failed: $(sprint(showerror, e)[1:min(end,80)])"
        end
    end

    # CUDA
    if isdefined(Main, :CUDA)
        try
            CUDA = getfield(Main, :CUDA)
            if CUDA.functional()
                push!(backends, "cuda" => (; backend=CUDA.CUDABackend(), hw_accel=false))
                @info "Detected: CUDA"
            end
        catch e
            @info "CUDA loaded but failed: $(sprint(showerror, e)[1:min(end,80)])"
        end
    end

    # pbrt-v4
    pbrt_path = get(ENV, "PBRT_PATH", "")
    if isempty(pbrt_path)
        for p in ["/sim/Programmieren/RayTracing/pbrt-v4/build/pbrt",
                  joinpath(homedir(), "pbrt-v4", "build", "pbrt")]
            isfile(p) || continue
            pbrt_path = p
            break
        end
    end
    if !isempty(pbrt_path) && isfile(pbrt_path)
        has_nvidia = try run(pipeline(`nvidia-smi`, devnull)); true catch; false end
        push!(backends, "pbrt_cpu" => (; pbrt_binary=pbrt_path, gpu=false))
        if has_nvidia
            push!(backends, "pbrt_gpu" => (; pbrt_binary=pbrt_path, gpu=true))
            @info "Detected: pbrt-v4 (CPU + GPU/OptiX)"
        else
            @info "Detected: pbrt-v4 (CPU only)"
        end
    end

    return backends
end

# =============================================================================
# Julia/RayMakie Benchmark Runner
# =============================================================================

"""
    run_julia_benchmarks(; backend, backend_name, hw_accel, scenes, n_warmup, n_trials)

Benchmark RayMakie render time (colorbuffer only, excluding scene creation).
"""
function run_julia_benchmarks(;
    backend,
    backend_name::String,
    hw_accel::Bool = false,
    hostname::String = detect_system()[1],
    gpu_name::String = detect_system()[2],
    scenes::Vector{String} = collect(keys(BENCHMARK_SCENES)),
    n_warmup::Int = 1,
    n_trials::Int = 3,
    output_dir::String = BENCHMARK_RESULTS_DIR,
)
    mkpath(output_dir)

    results = OrderedDict{String, Any}()
    results["metadata"] = OrderedDict(
        "hostname" => hostname,
        "gpu_name" => gpu_name,
        "backend_name" => backend_name,
        "renderer" => "raymakie",
        "hw_accel" => hw_accel,
        "n_warmup" => n_warmup,
        "n_trials" => n_trials,
        "timestamp" => Dates.format(now(), "yyyy-mm-dd HH:MM:SS"),
        "julia_version" => string(VERSION),
        "cpu_threads" => Sys.CPU_THREADS,
    )
    results["scenes"] = OrderedDict{String, Any}()

    for scene_name in scenes
        haskey(BENCHMARK_SCENES, scene_name) || (@warn "Unknown scene: $scene_name"; continue)
        cfg = BENCHMARK_SCENES[scene_name]

        println("\n" * "="^60)
        println("[$backend_name] $scene_name")
        println("  $(cfg.resolution) $(cfg.samples)spp max_depth=$(cfg.max_depth)")
        println("="^60)

        # Load scene script
        m = Module()
        Core.eval(m, :(include(path) = Base.include($m, path)))
        try
            Base.include(m, cfg.script)
        catch e
            @warn "Failed to load $scene_name" exception=(e, catch_backtrace())
            results["scenes"][scene_name] = OrderedDict("error" => sprint(showerror, e))
            continue
        end

        isdefined(m, :create_scene) || (@warn "$scene_name: no create_scene()"; continue)

        # Create scene once
        print("  Creating scene... ")
        local scene
        try
            scene = Base.invokelatest(m.create_scene; resolution=cfg.resolution)
        catch e
            @warn "create_scene failed" exception=(e, catch_backtrace())
            results["scenes"][scene_name] = OrderedDict("error" => sprint(showerror, e))
            continue
        end
        println("done.")

        sensor = Base.invokelatest(Hikari.FilmSensor; cfg.sensor_kwargs...)
        integrator = Base.invokelatest(Hikari.VolPath;
            samples=cfg.samples, max_depth=cfg.max_depth, hw_accel=hw_accel)
        Base.invokelatest(RayMakie.activate!; device=backend, sensor=sensor, cfg.colorbuffer_kwargs...)

        # Warmup
        println("  Warmup ($n_warmup)...")
        local img
        for i in 1:n_warmup
            try
                img = Base.invokelatest(Makie.colorbuffer, scene;
                    backend=RayMakie, integrator=integrator)
            catch e
                @warn "  Warmup $i failed" exception=(e, catch_backtrace())
            end
            GC.gc(false)
        end

        # Save render to scene output dir
        try
            scene_dir = dirname(cfg.script)
            out_path = joinpath(scene_dir, "output", "$(scene_name)_$(backend_name).png")
            mkpath(dirname(out_path))
            Base.invokelatest(FileIO.save, out_path, img)
        catch; end

        # Timed trials
        println("  Benchmarking ($n_trials trials)...")
        timings = Float64[]
        bench_error = nothing
        for trial in 1:n_trials
            GC.gc(true)
            try
                t = @elapsed begin
                    Base.invokelatest(Makie.colorbuffer, scene;
                        backend=RayMakie, integrator=integrator)
                end
                push!(timings, t)
                println("    Trial $trial: $(round(t, digits=3))s")
            catch e
                bench_error = sprint(showerror, e)
                @warn "  Trial $trial failed" exception=(e, catch_backtrace())
                break
            end
        end

        if !isempty(timings)
            scene_result = OrderedDict(
                "resolution" => [cfg.resolution...],
                "samples" => cfg.samples,
                "max_depth" => cfg.max_depth,
                "hw_accel" => hw_accel,
                "timings" => timings,
                "median" => round(_median(sort(timings)), digits=4),
                "min" => round(minimum(timings), digits=4),
                "max" => round(maximum(timings), digits=4),
            )
            results["scenes"][scene_name] = scene_result
            println("  => median=$(scene_result["median"])s  min=$(scene_result["min"])s")
        else
            results["scenes"][scene_name] = OrderedDict("error" => something(bench_error, "no timings"))
            println("  => FAILED")
        end
    end

    json_path = joinpath(output_dir, "$(hostname)_$(gpu_name)_$(backend_name).json")
    open(json_path, "w") do io
        JSON3.pretty(io, results)
    end
    println("\nSaved: $json_path")
    return results
end

# =============================================================================
# pbrt-v4 Benchmark Runner
# =============================================================================

"""
    run_pbrt_benchmarks(; pbrt_binary, pbrt_mode, scenes, n_warmup, n_trials)

Benchmark pbrt-v4 on scenes that have .pbrt files.
`pbrt_mode` is "cpu" (default) or "gpu" (OptiX, NVIDIA only).
"""
function run_pbrt_benchmarks(;
    pbrt_binary::String = "/sim/Programmieren/RayTracing/pbrt-v4/build/pbrt",
    pbrt_mode::String = "cpu",  # "cpu" or "gpu"
    hostname::String = detect_system()[1],
    gpu_name::String = detect_system()[2],
    scenes::Vector{String} = collect(keys(BENCHMARK_SCENES)),
    n_warmup::Int = 1,
    n_trials::Int = 3,
    output_dir::String = BENCHMARK_RESULTS_DIR,
)
    mkpath(output_dir)
    isfile(pbrt_binary) || error("pbrt-v4 not found at $pbrt_binary. Set PBRT_PATH env var or pass pbrt_binary kwarg.")

    backend_name = "pbrt_$(pbrt_mode)"

    results = OrderedDict{String, Any}()
    results["metadata"] = OrderedDict(
        "hostname" => hostname,
        "gpu_name" => gpu_name,
        "backend_name" => backend_name,
        "renderer" => "pbrt-v4",
        "pbrt_mode" => pbrt_mode,
        "pbrt_binary" => pbrt_binary,
        "n_warmup" => n_warmup,
        "n_trials" => n_trials,
        "timestamp" => Dates.format(now(), "yyyy-mm-dd HH:MM:SS"),
        "cpu_threads" => Sys.CPU_THREADS,
    )
    results["scenes"] = OrderedDict{String, Any}()

    for scene_name in scenes
        haskey(BENCHMARK_SCENES, scene_name) || continue
        cfg = BENCHMARK_SCENES[scene_name]

        # Skip scenes without pbrt files
        cfg.pbrt_file === nothing && continue
        pbrt_dir = cfg.pbrt_dir
        pbrt_file = cfg.pbrt_file
        isfile(joinpath(pbrt_dir, pbrt_file)) || (@warn "$scene_name: $(joinpath(pbrt_dir, pbrt_file)) not found"; continue)

        spp = cfg.samples

        println("\n" * "="^60)
        println("[pbrt_$(pbrt_mode)] $scene_name ($(spp)spp)")
        println("="^60)

        scene_dir = dirname(cfg.script)
        outfile = joinpath(scene_dir, "output", "$(scene_name)_pbrt_$(pbrt_mode).exr")
        mkpath(dirname(outfile))

        cmd_parts = [pbrt_binary]
        pbrt_mode == "gpu" && push!(cmd_parts, "--gpu")
        append!(cmd_parts, ["--spp", string(spp), "--quiet", "--outfile", outfile, pbrt_file])

        # Warmup
        println("  Warmup ($n_warmup)...")
        for i in 1:n_warmup
            try
                run(Cmd(Cmd(cmd_parts); dir=pbrt_dir))
            catch e
                @warn "  Warmup $i failed" exception=(e, catch_backtrace())
            end
        end

        # Timed trials
        println("  Benchmarking ($n_trials trials)...")
        timings = Float64[]
        bench_error = nothing
        for trial in 1:n_trials
            try
                t = @elapsed run(Cmd(Cmd(cmd_parts); dir=pbrt_dir))
                push!(timings, t)
                println("    Trial $trial: $(round(t, digits=3))s")
            catch e
                bench_error = sprint(showerror, e)
                @warn "  Trial $trial failed" exception=(e, catch_backtrace())
                break
            end
        end

        if !isempty(timings)
            scene_result = OrderedDict(
                "resolution" => "native",  # pbrt uses its own resolution from .pbrt file
                "samples" => spp,
                "pbrt_mode" => pbrt_mode,
                "timings" => timings,
                "median" => round(_median(sort(timings)), digits=4),
                "min" => round(minimum(timings), digits=4),
                "max" => round(maximum(timings), digits=4),
            )
            results["scenes"][scene_name] = scene_result
            println("  => median=$(scene_result["median"])s  min=$(scene_result["min"])s")
        else
            results["scenes"][scene_name] = OrderedDict("error" => something(bench_error, "no timings"))
            println("  => FAILED")
        end
    end

    json_path = joinpath(output_dir, "$(hostname)_$(gpu_name)_$(backend_name).json")
    open(json_path, "w") do io
        JSON3.pretty(io, results)
    end
    println("\nSaved: $json_path")
    return results
end

# =============================================================================
# Auto-detect and run everything
# =============================================================================

"""
    run_all_benchmarks(; scenes, n_warmup, n_trials)

Auto-detect all available backends and run benchmarks on all scenes.
Results saved as `{hostname}_{gpu}_{backend}.json` in benchmark/results/.
"""
function run_all_benchmarks(;
    scenes::Vector{String} = collect(keys(BENCHMARK_SCENES)),
    n_warmup::Int = 1,
    n_trials::Int = 3,
)
    backends = detect_backends()
    if isempty(backends)
        @warn "No backends detected! Load GPU packages first: using Lava, AMDGPU, CUDA"
        return
    end

    # Detect system after backends (Lava init provides GPU name)
    hostname, gpu_name = detect_system()
    println("System: $hostname / $gpu_name / $(Sys.CPU_THREADS) threads")
    println("Detected backends: $(join([b.first for b in backends], ", "))")
    println()

    all_results = OrderedDict{String, Any}()

    for (name, config) in backends
        println("\n" * "#"^60)
        println("# Backend: $name")
        println("#"^60)

        try
            if startswith(name, "pbrt_")
                mode = name == "pbrt_gpu" ? "gpu" : "cpu"
                r = run_pbrt_benchmarks(;
                    pbrt_binary=config.pbrt_binary,
                    pbrt_mode=mode,
                    hostname, gpu_name, scenes, n_warmup, n_trials)
                all_results[name] = r
            else
                r = run_julia_benchmarks(;
                    backend=config.backend,
                    backend_name=name,
                    hw_accel=config.hw_accel,
                    hostname, gpu_name, scenes, n_warmup, n_trials)
                all_results[name] = r
            end
        catch e
            @warn "Backend $name failed" exception=(e, catch_backtrace())
        end
    end

    # Print summary
    println("\n" * "="^70)
    println("BENCHMARK SUMMARY ($hostname / $gpu_name)")
    println("="^70)
    println()
    for scene_name in scenes
        print("  $scene_name: ")
        parts = String[]
        for (bname, r) in all_results
            if haskey(r["scenes"], scene_name)
                s = r["scenes"][scene_name]
                if haskey(s, "median")
                    push!(parts, "$bname=$(s["median"])s")
                end
            end
        end
        println(isempty(parts) ? "no results" : join(parts, "  "))
    end

    return all_results
end

# =============================================================================
# Utilities
# =============================================================================

function _median(sorted::Vector{Float64})
    n = length(sorted)
    n == 0 && return NaN
    n == 1 && return sorted[1]
    isodd(n) ? sorted[(n + 1) ÷ 2] : (sorted[n ÷ 2] + sorted[n ÷ 2 + 1]) / 2
end

# Need FileIO for saving renders
using FileIO
