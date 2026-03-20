# Benchmark Results Plotter
#
# Reads all JSON result files from benchmark/results/ and produces comparison plots.
# Handles both render benchmarks and AK (AcceleratedKernels) benchmarks.
# Supports multi-platform results (files from different machines/GPUs).
#
# Usage:
#   include("RayDemo/benchmark/plot_results.jl")
#   plot_all_results()       # render benchmark plots
#   plot_ak_results()        # AK benchmark plots
#   summarize_results()      # text summary of all results

using JSON3, OrderedCollections

include(joinpath(@__DIR__, "run_benchmarks.jl"))

# =============================================================================
# Loading
# =============================================================================

"""
    load_all_results(; results_dir) -> OrderedDict

Load all JSON benchmark results. Returns dict of filename => parsed JSON.
Separates render results (have "scenes" key) from AK results (have "benchmarks" key).
"""
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

function split_results(all_results)
    render = OrderedDict{String, Any}()
    ak = OrderedDict{String, Any}()
    for (name, res) in all_results
        if haskey(res, :scenes) || haskey(res, "scenes")
            render[name] = res
        elseif haskey(res, :benchmarks) || haskey(res, "benchmarks")
            ak[name] = res
        end
    end
    return render, ak
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

# =============================================================================
# Text Summary
# =============================================================================

"""
    summarize_results(; results_dir)

Print a text summary of all benchmark results, grouped by platform/GPU.
"""
function summarize_results(; results_dir::String = BENCHMARK_RESULTS_DIR)
    all = load_all_results(; results_dir)
    render, ak = split_results(all)

    if !isempty(render)
        println("=" ^ 75)
        println("RENDER BENCHMARKS (median seconds)")
        println("=" ^ 75)

        # Group by platform (all backends from same machine together)
        systems = OrderedDict{String, Vector{Pair{String,Any}}}()
        for (name, res) in render
            meta = _get(res, "metadata")
            plat = something(_get(meta, "platform"), "?")
            push!(get!(systems, plat, Pair{String,Any}[]), name => res)
        end

        for (sys_key, entries) in systems
            println("\n--- $sys_key ---")
            # Collect all scene names
            scene_names = OrderedCollections.OrderedSet{String}()
            for (_, res) in entries
                scenes = _get(res, "scenes")
                scenes !== nothing && for k in keys(scenes); push!(scene_names, String(k)); end
            end

            # Header
            backend_names = [String(something(_get(r, "metadata", "backend_name"), n)) for (n, r) in entries]
            print(rpad("Scene", 20))
            for bn in backend_names; print(rpad(bn, 14)); end
            println()
            println("-" ^ (20 + 14 * length(backend_names)))

            for scene in scene_names
                print(rpad(scene, 20))
                for (_, res) in entries
                    med = _get(res, "scenes", scene, "median")
                    if med !== nothing
                        print(rpad("$(med)s", 14))
                    else
                        print(rpad("-", 14))
                    end
                end
                println()
            end
        end
    end

    if !isempty(ak)
        println("\n\n" * "=" ^ 75)
        println("ACCELERATED KERNELS BENCHMARKS (median ms)")
        println("=" ^ 75)

        # Group by platform (all backends from same machine together)
        systems = OrderedDict{String, Vector{Pair{String,Any}}}()
        for (name, res) in ak
            meta = _get(res, "metadata")
            plat = something(_get(meta, "platform"), "?")
            push!(get!(systems, plat, Pair{String,Any}[]), name => res)
        end

        for (sys_key, entries) in systems
            n_elem = _get(entries[1].second, "metadata", "n_elements")
            n_str = n_elem !== nothing ? " ($(n_elem) elements)" : ""
            println("\n--- $sys_key$n_str ---")

            bench_names = OrderedCollections.OrderedSet{String}()
            for (_, res) in entries
                benchmarks = _get(res, "benchmarks")
                benchmarks !== nothing && for k in keys(benchmarks); push!(bench_names, String(k)); end
            end

            backend_names = [String(something(_get(r, "metadata", "backend"), n)) for (n, r) in entries]
            print(rpad("Operation", 30))
            for bn in backend_names; print(rpad(bn, 14)); end
            println()
            println("-" ^ (30 + 14 * length(backend_names)))

            for bench in bench_names
                print(rpad(bench, 30))
                for (_, res) in entries
                    med = _get(res, "benchmarks", bench, "median_ms")
                    if med !== nothing
                        print(rpad("$(med)", 14))
                    else
                        print(rpad("-", 14))
                    end
                end
                println()
            end
        end
    end

    isempty(render) && isempty(ak) && println("No benchmark results found in $results_dir")
end

# =============================================================================
# Render Plots
# =============================================================================

"""
    plot_all_results(; results_dir, output_dir)

Generate comparison bar charts from render benchmark results.
"""
function plot_all_results(;
    results_dir::String = BENCHMARK_RESULTS_DIR,
    output_dir::String = joinpath(BENCHMARK_RESULTS_DIR, "plots"),
)
    mkpath(output_dir)

    all = load_all_results(; results_dir)
    render, _ = split_results(all)
    if isempty(render)
        @warn "No render result files found in $results_dir"
        return nothing
    end

    # Collect scene names and backend labels
    scene_names = OrderedCollections.OrderedSet{String}()
    backend_labels = String[]
    backend_keys = String[]

    for (key, res) in render
        push!(backend_keys, key)
        meta = _get(res, "metadata")
        plat = something(_get(meta, "platform"), "")
        gpu = something(_get(meta, "gpu_name"), "")
        bname = something(_get(meta, "backend_name"), key)
        hw = _get(meta, "hw_accel")
        label = isempty(plat) ? bname : "$(plat)/$(gpu)/$(bname)"
        hw === true && (label *= " (HW)")
        push!(backend_labels, label)
        scenes = _get(res, "scenes")
        scenes !== nothing && for k in keys(scenes); push!(scene_names, String(k)); end
    end

    scene_list = collect(scene_names)
    n_scenes = length(scene_list)
    n_backends = length(backend_keys)

    if n_scenes == 0 || n_backends == 0
        @warn "No data to plot"
        return nothing
    end

    # Build timing matrix
    timing_matrix = fill(NaN, n_scenes, n_backends)
    for (bi, key) in enumerate(backend_keys)
        for (si, sn) in enumerate(scene_list)
            med = _get(render[key], "scenes", sn, "median")
            med !== nothing && (timing_matrix[si, bi] = Float64(med))
        end
    end

    # --- Plot 1: Grouped bar chart ---
    fig = Figure(size=(max(800, n_scenes * 200), 500))
    ax = Axis(fig[1, 1],
        title = "RayDemo Benchmark — Render Time (seconds)",
        xlabel = "Scene", ylabel = "Time (s)",
        xticks = (1:n_scenes, scene_list),
        xticklabelrotation = pi/6,
    )

    bar_width = 0.8 / n_backends
    colors = Makie.wong_colors()

    for bi in 1:n_backends
        offsets = (1:n_scenes) .+ (bi - (n_backends + 1) / 2) * bar_width
        vals = timing_matrix[:, bi]
        mask = .!isnan.(vals)
        any(mask) && barplot!(ax, offsets[mask], vals[mask];
            width=bar_width * 0.9,
            color=colors[mod1(bi, length(colors))],
            label=backend_labels[bi],
        )
    end
    Legend(fig[1, 2], ax)

    save_path = joinpath(output_dir, "benchmark_times.png")
    save(save_path, fig; px_per_unit=2)
    println("Saved: $save_path")

    # --- Plot 2: Speedup chart ---
    ref_times = [maximum(filter(!isnan, timing_matrix[si, :]); init=NaN) for si in 1:n_scenes]

    fig2 = Figure(size=(max(800, n_scenes * 200), 500))
    ax2 = Axis(fig2[1, 1],
        title = "RayDemo Benchmark — Speedup (higher is better)",
        xlabel = "Scene", ylabel = "Speedup (x)",
        xticks = (1:n_scenes, scene_list),
        xticklabelrotation = pi/6,
    )

    for bi in 1:n_backends
        offsets = (1:n_scenes) .+ (bi - (n_backends + 1) / 2) * bar_width
        speedups = ref_times ./ timing_matrix[:, bi]
        mask = .!isnan.(speedups)
        any(mask) && barplot!(ax2, offsets[mask], speedups[mask];
            width=bar_width * 0.9,
            color=colors[mod1(bi, length(colors))],
            label=backend_labels[bi],
        )
    end
    hlines!(ax2, [1.0]; color=:gray, linestyle=:dash, linewidth=1)
    Legend(fig2[1, 2], ax2)

    save_path2 = joinpath(output_dir, "benchmark_speedup.png")
    save(save_path2, fig2; px_per_unit=2)
    println("Saved: $save_path2")

    return fig, fig2
end

# =============================================================================
# AK Plots
# =============================================================================

"""
    plot_ak_results(; results_dir, output_dir)

Generate comparison bar charts from AcceleratedKernels benchmark results.
"""
function plot_ak_results(;
    results_dir::String = BENCHMARK_RESULTS_DIR,
    output_dir::String = joinpath(BENCHMARK_RESULTS_DIR, "plots"),
)
    mkpath(output_dir)

    all = load_all_results(; results_dir)
    _, ak = split_results(all)
    if isempty(ak)
        @warn "No AK result files found in $results_dir"
        return nothing
    end

    # Collect benchmark names and backend labels
    bench_names = OrderedCollections.OrderedSet{String}()
    backend_labels = String[]
    backend_keys = String[]

    for (key, res) in ak
        push!(backend_keys, key)
        meta = _get(res, "metadata")
        plat = something(_get(meta, "platform"), "")
        gpu = something(_get(meta, "gpu_name"), "")
        bname = something(_get(meta, "backend"), key)
        label = isempty(plat) ? bname : "$(plat)/$(gpu)/$(bname)"
        push!(backend_labels, label)
        benchmarks = _get(res, "benchmarks")
        benchmarks !== nothing && for k in keys(benchmarks); push!(bench_names, String(k)); end
    end

    bench_list = collect(bench_names)
    n_benches = length(bench_list)
    n_backends = length(backend_keys)

    if n_benches == 0 || n_backends == 0
        @warn "No AK data to plot"
        return nothing
    end

    # Build timing matrix
    timing_matrix = fill(NaN, n_benches, n_backends)
    for (bi, key) in enumerate(backend_keys)
        for (si, bn) in enumerate(bench_list)
            med = _get(ak[key], "benchmarks", bn, "median_ms")
            med !== nothing && (timing_matrix[si, bi] = Float64(med))
        end
    end

    fig = Figure(size=(max(1000, n_benches * 120), 500))
    ax = Axis(fig[1, 1],
        title = "AcceleratedKernels Benchmark (ms, lower is better)",
        xlabel = "Operation", ylabel = "Time (ms)",
        xticks = (1:n_benches, bench_list),
        xticklabelrotation = pi/4,
        yscale = log10,
    )

    bar_width = 0.8 / n_backends
    colors = Makie.wong_colors()

    for bi in 1:n_backends
        offsets = (1:n_benches) .+ (bi - (n_backends + 1) / 2) * bar_width
        vals = timing_matrix[:, bi]
        mask = .!isnan.(vals)
        any(mask) && barplot!(ax, offsets[mask], vals[mask];
            width=bar_width * 0.9,
            color=colors[mod1(bi, length(colors))],
            label=backend_labels[bi],
        )
    end
    Legend(fig[1, 2], ax)

    save_path = joinpath(output_dir, "ak_benchmarks.png")
    save(save_path, fig; px_per_unit=2)
    println("Saved: $save_path")

    return fig
end
