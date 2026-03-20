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

"""
    detect_platform() -> String

Return OS name for result file naming: "linux", "macos", or "windows".
"""
function detect_platform()
    Sys.islinux() && return "linux"
    Sys.isapple() && return "macos"
    Sys.iswindows() && return "windows"
    return "unknown"
end

# Common GPU name patterns → short names
const GPU_SHORT_NAMES = [
    r"7900\s*xtx"i => "7900xtx",
    r"7900\s*xt\b"i => "7900xt",
    r"7900\s*gre"i => "7900gre",
    r"7800\s*xt"i => "7800xt",
    r"7600\s*xt"i => "7600xt",
    r"rtx\s*4090"i => "rtx4090",
    r"rtx\s*4080"i => "rtx4080",
    r"rtx\s*4070"i => "rtx4070",
    r"rtx\s*3090"i => "rtx3090",
    r"rtx\s*3080"i => "rtx3080",
    r"rtx\s*3070"i => "rtx3070",
    r"rtx\s*5090"i => "rtx5090",
    r"rtx\s*5080"i => "rtx5080",
    r"rtx\s*5070"i => "rtx5070",
    r"m4\s*ultra"i => "m4ultra",
    r"m4\s*max"i => "m4max",
    r"m4\s*pro"i => "m4pro",
    r"m3\s*ultra"i => "m3ultra",
    r"m3\s*max"i => "m3max",
    r"m3\s*pro"i => "m3pro",
    r"m2\s*ultra"i => "m2ultra",
    r"m2\s*max"i => "m2max",
    r"m2\s*pro"i => "m2pro",
    r"arc\s*a770"i => "a770",
    r"arc\s*b580"i => "b580",
]

# Common GPU name patterns → short names
const CPU_SHORT_NAMES = [
    r"7950x3d"i => "7950x3d",
    r"7950x"i => "7950x",
    r"7900x3d"i => "7900x3d",
    r"7900x"i => "7900x",
    r"7800x3d"i => "7800x3d",
    r"7700x"i => "7700x",
    r"9950x3d"i => "9950x3d",
    r"9950x"i => "9950x",
    r"9900x"i => "9900x",
    r"9800x3d"i => "9800x3d",
    r"13900k"i => "13900k",
    r"14900k"i => "14900k",
    r"i9.?13900"i => "13900",
    r"i9.?14900"i => "14900",
    r"i7.?13700"i => "13700",
    r"i7.?14700"i => "14700",
    r"m4\s*ultra"i => "m4ultra",
    r"m4\s*max"i => "m4max",
    r"m4\s*pro"i => "m4pro",
    r"m3\s*ultra"i => "m3ultra",
    r"m3\s*max"i => "m3max",
    r"m3\s*pro"i => "m3pro",
]

"""
    shorten_device_name(full_name, patterns) -> String

Map a full device name to a short identifier for filenames.
"""
function shorten_device_name(full_name::AbstractString, patterns)
    for (pat, short) in patterns
        occursin(pat, full_name) && return short
    end
    # Fallback: sanitize the full name
    short = replace(lowercase(full_name), r"[^a-z0-9]+" => "_")
    short = strip(short, '_')
    # Trim common prefixes
    for prefix in ["amd_radeon_rx_", "amd_radeon_", "nvidia_geforce_", "nvidia_",
                    "intel_arc_", "apple_", "amd_ryzen_9_", "amd_ryzen_7_",
                    "amd_ryzen_5_", "intel_core_"]
        startswith(short, prefix) && (short = short[length(prefix)+1:end])
    end
    # Trim trailing qualifiers like "_12_core_processor"
    short = replace(short, r"_\d+_core_processor$" => "")
    return String(short)
end

"""
    detect_gpu_name() -> String

Detect the GPU and return a short name for filenames.
"""
function detect_gpu_name()
    # Try Lava context first (most reliable when loaded)
    try
        return shorten_device_name(Main.Lava.vk_context().device_name, GPU_SHORT_NAMES)
    catch; end

    # Try NVIDIA
    try
        full = strip(read(`nvidia-smi --query-gpu=name --format=csv,noheader`, String))
        return shorten_device_name(full, GPU_SHORT_NAMES)
    catch; end

    # Try vulkaninfo
    try
        info = read(pipeline(`vulkaninfo --summary`, stderr=devnull), String)
        m = match(r"deviceName\s*=\s*(.*)", info)
        m !== nothing && return shorten_device_name(strip(m.captures[1]), GPU_SHORT_NAMES)
    catch; end

    return "unknown_gpu"
end

"""
    detect_cpu_name() -> String

Detect the CPU and return a short name for filenames.
"""
function detect_cpu_name()
    try
        cpuinfo = read("/proc/cpuinfo", String)
        m = match(r"model name\s*:\s*(.*)", cpuinfo)
        m !== nothing && return shorten_device_name(strip(m.captures[1]), CPU_SHORT_NAMES)
    catch; end
    # macOS
    try
        full = strip(read(`sysctl -n machdep.cpu.brand_string`, String))
        return shorten_device_name(full, CPU_SHORT_NAMES)
    catch; end
    return "unknown_cpu"
end

"""
    detect_system() -> (platform, gpu_name, cpu_name)

Return (platform, gpu_short_name, cpu_short_name) for result file naming.
E.g. ("linux", "7900xtx", "7900x")
"""
function detect_system()
    return detect_platform(), detect_gpu_name(), detect_cpu_name()
end

"""
    device_name_for_backend(backend_name, gpu_name, cpu_name) -> String

Return the appropriate device name for a backend.
GPU backends use gpu_name, CPU backends use cpu_name.
"""
function device_name_for_backend(backend_name::String, gpu_name::String, cpu_name::String)
    # CPU-based backends
    backend_name in ("pbrt_cpu", "cpu", "abacus") && return cpu_name
    return gpu_name
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
    platform::String = detect_platform(),
    gpu_name::String = detect_gpu_name(),
    cpu_name::String = detect_cpu_name(),
    scenes::Vector{String} = collect(keys(BENCHMARK_SCENES)),
    n_warmup::Int = 1,
    n_trials::Int = 3,
    output_dir::String = BENCHMARK_RESULTS_DIR,
)
    mkpath(output_dir)
    device_name = device_name_for_backend(backend_name, gpu_name, cpu_name)

    results = OrderedDict{String, Any}()
    results["metadata"] = OrderedDict(
        "platform" => platform,
        "gpu_name" => gpu_name,
        "cpu_name" => cpu_name,
        "device_name" => device_name,
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

        # Save render to results/renders/
        try
            renders_dir = joinpath(output_dir, "renders")
            mkpath(renders_dir)
            out_path = joinpath(renders_dir, "$(scene_name)_$(backend_name).png")
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

    json_path = joinpath(output_dir, "$(platform)_$(device_name)_$(backend_name).json")
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
    platform::String = detect_platform(),
    gpu_name::String = detect_gpu_name(),
    cpu_name::String = detect_cpu_name(),
    scenes::Vector{String} = collect(keys(BENCHMARK_SCENES)),
    n_warmup::Int = 1,
    n_trials::Int = 3,
    output_dir::String = BENCHMARK_RESULTS_DIR,
)
    mkpath(output_dir)
    isfile(pbrt_binary) || error("pbrt-v4 not found at $pbrt_binary. Set PBRT_PATH env var or pass pbrt_binary kwarg.")

    backend_name = "pbrt_$(pbrt_mode)"
    device_name = device_name_for_backend(backend_name, gpu_name, cpu_name)

    results = OrderedDict{String, Any}()
    results["metadata"] = OrderedDict(
        "platform" => platform,
        "gpu_name" => gpu_name,
        "cpu_name" => cpu_name,
        "device_name" => device_name,
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

    json_path = joinpath(output_dir, "$(platform)_$(device_name)_$(backend_name).json")
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
    run_all_benchmarks(; scenes, n_warmup, n_trials, ak_n)

Auto-detect all available backends and run ALL benchmarks (render + AK).
Results saved as `{platform}_{gpu}_{backend}.json` in benchmark/results/.

To run only render benchmarks: `run_all_render_benchmarks()`
To run only AK benchmarks: include run_ak_benchmarks.jl and call `run_ak_benchmarks()`
"""
function run_all_benchmarks(;
    scenes::Vector{String} = collect(keys(BENCHMARK_SCENES)),
    n_warmup::Int = 1,
    n_trials::Int = 3,
    ak_n::Int = 10_000_000,
)
    # Run render benchmarks
    render_results = run_all_render_benchmarks(; scenes, n_warmup, n_trials)

    # Run AK benchmarks
    platform, gpu_name, cpu_name = detect_system()
    println("\n\n" * "#"^60)
    println("# AcceleratedKernels Benchmarks ($ak_n elements)")
    println("#"^60)
    try
        include(joinpath(@__DIR__, "run_ak_benchmarks.jl"))
        Base.invokelatest(run_ak_benchmarks; n=ak_n, platform, gpu_name, cpu_name)
    catch e
        @warn "AK benchmarks failed" exception=(e, catch_backtrace())
    end

    return render_results
end

"""
    run_all_render_benchmarks(; scenes, n_warmup, n_trials)

Auto-detect all available backends and run render benchmarks on all scenes.
Results saved as `{platform}_{gpu}_{backend}.json` in benchmark/results/.
"""
function run_all_render_benchmarks(;
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
    platform, gpu_name, cpu_name = detect_system()
    println("System: $platform / GPU=$gpu_name / CPU=$cpu_name / $(Sys.CPU_THREADS) threads")
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
                    platform, gpu_name, cpu_name, scenes, n_warmup, n_trials)
                all_results[name] = r
            else
                r = run_julia_benchmarks(;
                    backend=config.backend,
                    backend_name=name,
                    hw_accel=config.hw_accel,
                    platform, gpu_name, cpu_name, scenes, n_warmup, n_trials)
                all_results[name] = r
            end
        catch e
            @warn "Backend $name failed" exception=(e, catch_backtrace())
        end
    end

    # Print summary
    println("\n" * "="^70)
    println("BENCHMARK SUMMARY ($platform / GPU=$gpu_name / CPU=$cpu_name)")
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
