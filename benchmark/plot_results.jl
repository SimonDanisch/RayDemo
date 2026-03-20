# Benchmark Results Plotter
#
# Reads all JSON result files from benchmark/results/ and produces comparison plots.
# Organized by device: each device gets a single 4-panel figure (render times,
# speedup, AK 10M, AK 100M). Cross-device overview plots compare backends.
#
# Usage:
#   include("RayDemo/benchmark/plot_results.jl")
#   plot_device("7900xtx")           # 4-panel figure for one device
#   plot_all_devices()               # one 4-panel figure per device
#   plot_overview()                   # cross-device comparison
#   summarize_results()              # text summary

using JSON3, OrderedCollections, Colors

include(joinpath(@__DIR__, "run_benchmarks.jl"))

# Ensure CairoMakie is active for headless plot generation
# (run_benchmarks.jl loads RayMakie which steals the backend)
try
    CairoMakie = Base.require(Main, :CairoMakie)
    CairoMakie.activate!()
catch
    @warn "CairoMakie not available, plots may not render correctly. Run: Pkg.add(\"CairoMakie\")"
end

# =============================================================================
# Loading & Grouping
# =============================================================================

function load_all_results(; results_dir::String = BENCHMARK_RESULTS_DIR)
    all = OrderedDict{String, Any}()
    isdir(results_dir) || return all
    for f in sort(readdir(results_dir))
        endswith(f, ".json") || continue
        path = joinpath(results_dir, f)
        name = splitext(f)[1]
        try
            all[name] = JSON3.read(read(path, String))
        catch e
            @warn "Failed to parse $f" exception=e
        end
    end
    return all
end

function _get(d, keys...)
    for k in keys
        d === nothing && return nothing
        if d isa AbstractDict
            d = get(d, k, get(d, Symbol(k), nothing))
        else
            d = nothing
        end
    end
    return d
end

"""
    group_by_device(; results_dir) -> OrderedDict{device => (render, ak_by_size)}

Group all results by device name.
Returns `device => (render_entries, ak_by_size)` where:
- render_entries: Vector of (backend_name, result) for render benchmarks
- ak_by_size: Dict{n_elements => Vector of (backend_name, result)} for AK benchmarks
"""
function group_by_device(; results_dir::String = BENCHMARK_RESULTS_DIR)
    all = load_all_results(; results_dir)
    devices = OrderedDict{String, @NamedTuple{
        platform::String,
        render::Vector{Tuple{String,Any}},
        ak::OrderedDict{Int, Vector{Tuple{String,Any}}}
    }}()

    # First pass: collect all entries by device
    cpu_ak = Tuple{String, Int, String, Any}[]  # (platform, n, bname, res)

    for (_, res) in all
        meta = _get(res, "metadata")
        meta === nothing && continue
        device = String(something(_get(meta, "device_name"), "unknown"))
        platform = String(something(_get(meta, "platform"), "unknown"))

        if !haskey(devices, device)
            devices[device] = (platform=platform, render=Tuple{String,Any}[], ak=OrderedDict{Int,Vector{Tuple{String,Any}}}())
        end
        dev = devices[device]

        if haskey(res, :scenes) || haskey(res, "scenes")
            bname = String(something(_get(meta, "backend_name"), "?"))
            push!(dev.render, (bname, res))
        elseif haskey(res, :benchmarks) || haskey(res, "benchmarks")
            bname = String(something(_get(meta, "backend"), "?"))
            n = Int(something(_get(meta, "n_elements"), 0))
            n == 0 && continue
            if !haskey(dev.ak, n)
                dev.ak[n] = Tuple{String,Any}[]
            end
            push!(dev.ak[n], (bname, res))
            # Track CPU AK entries for cross-device inclusion
            if bname in ("cpu", "abacus")
                push!(cpu_ak, (platform, n, bname, res))
            end
        end
    end

    # Second pass: add CPU AK entries to GPU device groups on the same platform
    for (device, dev) in devices
        for (plat, n, bname, res) in cpu_ak
            plat == dev.platform || continue
            # Skip if this IS the CPU device (already has it)
            existing = get(dev.ak, n, Tuple{String,Any}[])
            any(e[1] == bname for e in existing) && continue
            if !haskey(dev.ak, n)
                dev.ak[n] = Tuple{String,Any}[]
            end
            push!(dev.ak[n], (bname, res))
        end
    end

    return devices
end

# =============================================================================
# Backend display helpers
# =============================================================================

const BACKEND_DISPLAY_NAMES = OrderedDict(
    "lava_hw" => "Lava HW RT",
    "lava_sw" => "Lava SW",
    "amdgpu" => "AMDGPU",
    "cuda" => "CUDA",
    "pbrt_cpu" => "pbrt-v4 CPU",
    "pbrt_gpu" => "pbrt-v4 GPU",
    "abacus" => "Abacus",
    "lava" => "Lava",
    "cpu" => "CPU",
)

backend_label(name::String) = get(BACKEND_DISPLAY_NAMES, name, name)

const BACKEND_COLORS = OrderedDict(
    "lava_hw" => colorant"#0072B2",
    "lava_sw" => colorant"#E69F00",
    "amdgpu" => colorant"#009E73",
    "cuda" => colorant"#CC79A7",
    "pbrt_cpu" => colorant"#999999",
    "pbrt_gpu" => colorant"#F0E442",
    "abacus" => colorant"#D55E00",
    "lava" => colorant"#0072B2",
    "cpu" => colorant"#999999",
)

backend_color(name::String) = get(BACKEND_COLORS, name, :gray)

# =============================================================================
# Axis-level plotting helpers
# =============================================================================

"""Sort backends by geometric mean time (fastest first)."""
function sort_backends_by_speed(timing_matrix)
    n_backends = size(timing_matrix, 2)
    mean_times = Float64[]
    for bi in 1:n_backends
        vals = filter(!isnan, timing_matrix[:, bi])
        push!(mean_times, isempty(vals) ? Inf : exp(sum(log.(vals)) / length(vals)))
    end
    return sortperm(mean_times)
end

"""Compute geometric mean speedup vs slowest backend."""
function geomean_speedup(timing_matrix::Matrix{Float64}, bi::Int)
    speedups = Float64[]
    for si in axes(timing_matrix, 1)
        row = timing_matrix[si, :]
        valid = filter(!isnan, row)
        isempty(valid) && continue
        val = timing_matrix[si, bi]
        isnan(val) && continue
        push!(speedups, maximum(valid) / val)
    end
    isempty(speedups) && return 1.0
    return exp(sum(log.(speedups)) / length(speedups))
end

"""Build legend label: "Lava HW RT (17.6x)" """
function speedup_label(bname::String, timing_matrix::Matrix{Float64}, bi::Int)
    label = backend_label(bname)
    s = geomean_speedup(timing_matrix, bi)
    label *= s >= 1.05 ? " ($(round(s, digits=1))x)" : " (1x)"
    return label
end

"""
Draw grouped bars with per-group sorting (shortest left, tallest right).
Color identifies backend. Legend entry added once per backend.
"""
function draw_sorted_grouped_bars!(ax, timing_matrix, backend_names, label_fn;
        bar_width, fillto=0)
    n_ops = size(timing_matrix, 1)
    labeled = Set{Int}()

    for si in 1:n_ops
        row = timing_matrix[si, :]
        valid = findall(!isnan, row)
        isempty(valid) && continue
        order = valid[sortperm(row[valid])]

        n_valid = length(order)
        for (slot, bi) in enumerate(order)
            offset = si + (slot - (n_valid + 1) / 2) * bar_width
            kw = (width=bar_width * 0.9, color=backend_color(backend_names[bi]), fillto=fillto)
            if bi ∉ labeled
                barplot!(ax, [offset], [row[bi]]; kw..., label=label_fn(bi))
                push!(labeled, bi)
            else
                barplot!(ax, [offset], [row[bi]]; kw...)
            end
        end
    end
end

# =============================================================================
# Axis builders (operate on a single Axis)
# =============================================================================

"""Build render timing matrix from entries. Returns (scene_list, bnames, timing_matrix)."""
function build_render_data(entries)
    scene_names = OrderedCollections.OrderedSet{String}()
    for (_, res) in entries
        scenes = _get(res, "scenes")
        scenes !== nothing && for k in keys(scenes); push!(scene_names, String(k)); end
    end
    scene_list = collect(scene_names)
    bnames = [e[1] for e in entries]

    timing = fill(NaN, length(scene_list), length(entries))
    for (bi, (_, res)) in enumerate(entries)
        for (si, sn) in enumerate(scene_list)
            med = _get(res, "scenes", sn, "median")
            med !== nothing && (timing[si, bi] = Float64(med))
        end
    end

    # Sort backends by speed
    perm = sort_backends_by_speed(timing)
    return scene_list, bnames[perm], timing[:, perm]
end

"""Build AK timing matrix from entries. Returns (op_labels, bnames, timing_matrix)."""
function build_ak_data(entries)
    bench_names = OrderedCollections.OrderedSet{String}()
    for (_, res) in entries
        benchmarks = _get(res, "benchmarks")
        benchmarks !== nothing && for k in keys(benchmarks)
            ks = String(k)
            endswith(ks, "/base") && continue
            push!(bench_names, ks)
        end
    end
    bench_list = collect(bench_names)
    bnames = [e[1] for e in entries]

    timing = fill(NaN, length(bench_list), length(entries))
    for (bi, (_, res)) in enumerate(entries)
        for (si, bn) in enumerate(bench_list)
            med = _get(res, "benchmarks", bn, "median_ms")
            med !== nothing && (timing[si, bi] = Float64(med))
        end
    end

    perm = sort_backends_by_speed(timing)
    short_labels = [replace(replace(replace(b,
                            "Float32" => "f32", "UInt32" => "u32"),
                            r"/acck.*" => ""),
                            "accumulate" => "accum", "mapreduce" => "mapred", "sortperm" => "sortpm")
                    for b in bench_list]
    return short_labels, bnames[perm], timing[:, perm]
end

"""Plot render times on an Axis."""
function plot_render_times!(ax, entries; show_legend=true)
    scene_list, bnames, timing = build_render_data(entries)
    isempty(scene_list) && return nothing

    ax.title = "Render Time (s, lower is better)"
    ax.xlabel = "Scene"
    ax.ylabel = "Time (s)"
    ax.xticks = (1:length(scene_list), scene_list)
    ax.xticklabelrotation = pi/6

    bar_width = 0.8 / length(bnames)
    draw_sorted_grouped_bars!(ax, timing, bnames,
        bi -> speedup_label(bnames[bi], timing, bi);
        bar_width)
    return timing, bnames
end

"""Plot render speedup on an Axis."""
function plot_render_speedup!(ax, entries; show_legend=true)
    scene_list, bnames, timing = build_render_data(entries)
    isempty(scene_list) && return nothing
    n_scenes, n_backends = size(timing)

    ref_times = [let v = filter(!isnan, timing[si, :])
                     isempty(v) ? NaN : maximum(v)
                 end for si in 1:n_scenes]

    speedup = fill(NaN, n_scenes, n_backends)
    for si in 1:n_scenes, bi in 1:n_backends
        s = ref_times[si] / timing[si, bi]
        (!isnan(s) && !isinf(s)) && (speedup[si, bi] = s)
    end

    ax.title = "Speedup vs Slowest (higher is better)"
    ax.xlabel = "Scene"
    ax.ylabel = "Speedup (x)"
    ax.xticks = (1:n_scenes, scene_list)
    ax.xticklabelrotation = pi/6

    bar_width = 0.8 / n_backends
    draw_sorted_grouped_bars!(ax, speedup, bnames,
        bi -> speedup_label(bnames[bi], timing, bi);
        bar_width)
    hlines!(ax, [1.0]; color=:gray, linestyle=:dash, linewidth=1)
    return speedup, bnames
end

"""Plot AK benchmark times on an Axis."""
function plot_ak_times!(ax, entries, n_str::String)
    labels, bnames, timing = build_ak_data(entries)
    isempty(labels) && return nothing
    n_ops, n_backends = size(timing)

    all_vals = filter(!isnan, vec(timing))
    y_min = isempty(all_vals) ? 0.01 : minimum(all_vals) * 0.5

    ax.title = "AK $n_str (ms, lower is better)"
    ax.xlabel = "Operation"
    ax.ylabel = "Time (ms)"
    ax.xticks = (1:n_ops, labels)
    ax.xticklabelrotation = pi/3

    bar_width = 0.8 / n_backends
    draw_sorted_grouped_bars!(ax, timing, bnames,
        bi -> speedup_label(bnames[bi], timing, bi);
        bar_width, fillto=y_min)

    # Set log scale AFTER plotting so Makie has valid data limits
    ax.yscale = log10
    return timing, bnames
end

# =============================================================================
# Per-device 4-panel figure
# =============================================================================

"""
    plot_device(device_name; results_dir, output_dir) -> Figure

Generate a single 4-panel figure for one device:
  [render_times | speedup  ]
  [ak_10m       | ak_100m  ]

Saves to `{output_dir}/{device_name}.png`.
"""
function plot_device(device_name::String;
    results_dir::String = BENCHMARK_RESULTS_DIR,
    output_dir::String = joinpath(BENCHMARK_RESULTS_DIR, "plots"),
)
    mkpath(output_dir)
    devices = group_by_device(; results_dir)

    if !haskey(devices, device_name)
        available = join(keys(devices), ", ")
        error("Device '$device_name' not found. Available: $available")
    end

    dev = devices[device_name]
    platform = dev.platform
    ak_sizes = sort(collect(keys(dev.ak)))

    # Determine grid: render row + one row per AK size
    has_render = !isempty(dev.render)
    has_ak = !isempty(ak_sizes)
    n_rows = (has_render ? 1 : 0) + (has_ak ? 1 : 0)
    n_rows == 0 && error("No data for device '$device_name'")

    # Columns: 2 for render (times + speedup), max(n_ak_sizes, 2) for AK
    n_ak = length(ak_sizes)
    n_cols = max(has_render ? 2 : 0, n_ak)

    fig = Figure(size=(600 * n_cols, 500 * n_rows + 40))
    Label(fig[0, 1:n_cols], "$platform / $device_name", fontsize=20, font=:bold)

    row = 1

    # --- Render row ---
    if has_render
        ax1 = Axis(fig[row, 1])
        result1 = plot_render_times!(ax1, dev.render)

        ax2 = Axis(fig[row, 2])
        result2 = plot_render_speedup!(ax2, dev.render)

        # Shared legend for render row — use the times axis (has speedup labels)
        if result1 !== nothing
            Legend(fig[row, n_cols + 1], ax1, framevisible=false, padding=(0,0,0,0))
        end

        for c in 3:n_cols
            # empty cell
        end
        row += 1
    end

    # --- AK row ---
    if has_ak
        local ak_ax = nothing
        for (ci, n_elem) in enumerate(ak_sizes)
            n_str = n_elem >= 1_000_000 ? "$(div(n_elem, 1_000_000))M" : "$(div(n_elem, 1_000))K"
            ax = Axis(fig[row, ci])
            result = plot_ak_times!(ax, dev.ak[n_elem], n_str)
            if result !== nothing && ak_ax === nothing
                ak_ax = ax
            end
        end
        # Shared legend for AK row
        if ak_ax !== nothing
            Legend(fig[row, n_cols + 1], ak_ax, framevisible=false, padding=(0,0,0,0))
        end
        for c in (n_ak+1):n_cols
            # empty cell
        end
    end

    # Give legend column minimal width
    colsize!(fig.layout, n_cols + 1, Auto(0.3))

    save_path = joinpath(output_dir, "$(device_name).png")
    save(save_path, fig; px_per_unit=2)
    println("Saved: $save_path")
    return fig
end

"""
    plot_all_devices(; results_dir, output_dir) -> Dict{String, Figure}

Generate a 4-panel figure for every device found in results.
"""
function plot_all_devices(;
    results_dir::String = BENCHMARK_RESULTS_DIR,
    output_dir::String = joinpath(BENCHMARK_RESULTS_DIR, "plots"),
)
    devices = group_by_device(; results_dir)
    figs = OrderedDict{String, Any}()

    # Collect which devices have GPU results (render or GPU AK)
    gpu_devices = String[]
    cpu_devices = String[]
    for (name, dev) in devices
        has_gpu_data = !isempty(dev.render) || any(
            any(b[1] ∉ ("cpu", "abacus") for b in entries) for entries in values(dev.ak))
        if has_gpu_data
            push!(gpu_devices, name)
        else
            push!(cpu_devices, name)
        end
    end

    # Plot GPU devices (have render + AK), then CPU-only devices (AK only)
    for name in vcat(gpu_devices, cpu_devices)
        try
            figs[name] = plot_device(name; results_dir, output_dir)
        catch e
            @warn "Failed to plot device $name" exception=(e, catch_backtrace())
        end
    end
    return figs
end

# =============================================================================
# Cross-device overview: mean speedup per backend across all devices
# =============================================================================

"""
    plot_overview(; results_dir, output_dir) -> Figure

Cross-device overview: geometric mean speedup per GPU backend on render benchmarks.
Only includes devices with GPU render data. Excludes AK (incomparable scales)
and CPU-only backends.
"""
function plot_overview(;
    results_dir::String = BENCHMARK_RESULTS_DIR,
    output_dir::String = joinpath(BENCHMARK_RESULTS_DIR, "plots"),
)
    mkpath(output_dir)
    devices = group_by_device(; results_dir)

    # Only devices with ≥2 render backends
    gpu_backends = Set(["lava_hw", "lava_sw", "lava", "cuda", "amdgpu", "pbrt_gpu", "pbrt_cpu", "abacus"])
    backend_order = ["lava_hw", "lava_sw", "cuda", "amdgpu", "pbrt_gpu", "pbrt_cpu", "abacus"]

    plot_devices = String[]
    device_speedups = OrderedDict{String, OrderedDict{String, Float64}}()

    for (dname, dev) in devices
        isempty(dev.render) && continue
        length(dev.render) < 2 && continue

        _, bnames, timing = build_render_data(dev.render)
        speedups = OrderedDict{String, Float64}()
        for (bi, bn) in enumerate(bnames)
            speedups[bn] = geomean_speedup(timing, bi)
        end
        if length(speedups) >= 2
            push!(plot_devices, dname)
            device_speedups[dname] = speedups
        end
    end

    isempty(plot_devices) && (@warn "No multi-backend render devices"; return nothing)

    # Collect all backends present, sorted
    all_backends = OrderedCollections.OrderedSet{String}()
    for s in values(device_speedups), bn in keys(s)
        push!(all_backends, bn)
    end
    present_backends = sort(collect(all_backends),
        by=b -> something(findfirst(==(b), backend_order), 100))

    fig = Figure(size=(max(700, 180 * length(plot_devices)), 400))
    ax = Axis(fig[1, 1],
        title = "Render Speedup vs Slowest (geometric mean across scenes)",
        ylabel = "Speedup (×)",
        xticks = (1:length(plot_devices), plot_devices),
        xticklabelrotation = pi/6,
    )

    bar_width = 0.8 / length(present_backends)
    for (bi, bn) in enumerate(present_backends)
        xs = Float64[]
        ys = Float64[]
        for (di, dname) in enumerate(plot_devices)
            s = get(device_speedups[dname], bn, NaN)
            if !isnan(s)
                offset = di + (bi - (length(present_backends) + 1) / 2) * bar_width
                push!(xs, offset)
                push!(ys, s)
            end
        end
        if !isempty(xs)
            barplot!(ax, xs, ys; width=bar_width * 0.9,
                     color=backend_color(bn), label=backend_label(bn))
        end
    end

    hlines!(ax, [1.0]; color=:gray, linestyle=:dash, linewidth=1)
    Legend(fig[1, 2], ax, framevisible=false)

    save_path = joinpath(output_dir, "overview.png")
    save(save_path, fig; px_per_unit=2)
    println("Saved: $save_path")
    return fig
end

# =============================================================================
# Legacy standalone plot functions (still useful)
# =============================================================================

"""Generate standalone render benchmark plots (all devices combined)."""
function plot_all_results(;
    results_dir::String = BENCHMARK_RESULTS_DIR,
    output_dir::String = joinpath(BENCHMARK_RESULTS_DIR, "plots"),
)
    mkpath(output_dir)
    all = load_all_results(; results_dir)

    # Collect all render entries
    entries = Tuple{String,Any}[]
    for (_, res) in all
        (haskey(res, :scenes) || haskey(res, "scenes")) || continue
        meta = _get(res, "metadata")
        bname = String(something(_get(meta, "backend_name"), "?"))
        push!(entries, (bname, res))
    end
    isempty(entries) && (@warn "No render results"; return nothing)

    fig = Figure(size=(900, 550))
    ax = Axis(fig[1, 1])
    plot_render_times!(ax, entries)
    Legend(fig[1, 2], ax)
    save_path = joinpath(output_dir, "benchmark_times.png")
    save(save_path, fig; px_per_unit=2)
    println("Saved: $save_path")

    fig2 = Figure(size=(900, 550))
    ax2 = Axis(fig2[1, 1])
    plot_render_speedup!(ax2, entries)
    Legend(fig2[1, 2], ax2)
    save_path2 = joinpath(output_dir, "benchmark_speedup.png")
    save(save_path2, fig2; px_per_unit=2)
    println("Saved: $save_path2")

    return fig, fig2
end

"""Generate standalone AK benchmark plots (all devices combined)."""
function plot_ak_results(;
    results_dir::String = BENCHMARK_RESULTS_DIR,
    output_dir::String = joinpath(BENCHMARK_RESULTS_DIR, "plots"),
)
    mkpath(output_dir)
    all = load_all_results(; results_dir)

    by_size = OrderedDict{Int, Vector{Tuple{String,Any}}}()
    for (_, res) in all
        (haskey(res, :benchmarks) || haskey(res, "benchmarks")) || continue
        meta = _get(res, "metadata")
        bname = String(something(_get(meta, "backend"), "?"))
        n = _get(meta, "n_elements")
        n === nothing && continue
        n = Int(n)
        push!(get!(by_size, n, Tuple{String,Any}[]), (bname, res))
    end

    figs = []
    for (n_elem, entries) in sort(by_size)
        n_str = n_elem >= 1_000_000 ? "$(div(n_elem, 1_000_000))M" : "$(div(n_elem, 1_000))K"
        fig = Figure(size=(1000, 550))
        ax = Axis(fig[1, 1])
        plot_ak_times!(ax, entries, n_str)
        Legend(fig[1, 2], ax)
        save_path = joinpath(output_dir, "ak_benchmarks_$(lowercase(n_str)).png")
        save(save_path, fig; px_per_unit=2)
        println("Saved: $save_path")
        push!(figs, fig)
    end
    return figs
end

# =============================================================================
# Text Summary
# =============================================================================

function summarize_results(; results_dir::String = BENCHMARK_RESULTS_DIR)
    devices = group_by_device(; results_dir)

    for (device, dev) in devices
        println("=" ^ 75)
        println("$(dev.platform) / $device")
        println("=" ^ 75)

        if !isempty(dev.render)
            println("\n  RENDER (median seconds):")
            scene_list, bnames, timing = build_render_data(dev.render)
            print("  " * rpad("Scene", 18))
            for bn in bnames; print(rpad(backend_label(bn), 14)); end
            println()
            println("  " * "-" ^ (18 + 14 * length(bnames)))
            for (si, sn) in enumerate(scene_list)
                print("  " * rpad(sn, 18))
                for bi in eachindex(bnames)
                    v = timing[si, bi]
                    print(rpad(isnan(v) ? "-" : "$(round(v, digits=3))s", 14))
                end
                println()
            end
        end

        for n_elem in sort(collect(keys(dev.ak)))
            n_str = n_elem >= 1_000_000 ? "$(div(n_elem, 1_000_000))M" : "$(div(n_elem, 1_000))K"
            println("\n  AK $n_str elements (median ms):")
            labels, bnames, timing = build_ak_data(dev.ak[n_elem])
            print("  " * rpad("Operation", 24))
            for bn in bnames; print(rpad(backend_label(bn), 14)); end
            println()
            println("  " * "-" ^ (24 + 14 * length(bnames)))
            for (si, lab) in enumerate(labels)
                print("  " * rpad(lab, 24))
                for bi in eachindex(bnames)
                    v = timing[si, bi]
                    print(rpad(isnan(v) ? "-" : "$(round(v, digits=3))", 14))
                end
                println()
            end
        end
        println()
    end
end
