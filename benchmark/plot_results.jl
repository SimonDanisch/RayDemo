# Benchmark Results Plotter
#
# Reads all JSON result files from benchmark/results/ and produces comparison plots.
#
# Usage:
#   include("RayDemo/benchmark/plot_results.jl")
#   plot_all_results()

using JSON3, OrderedCollections
using Makie

include(joinpath(@__DIR__, "run_benchmarks.jl"))

"""
    load_all_results(; results_dir)

Load all JSON benchmark result files from the results directory.
Returns a Dict of filename (without .json) => parsed results.
"""
function load_all_results(; results_dir::String = BENCHMARK_RESULTS_DIR)
    all_results = OrderedDict{String, Any}()
    for f in sort(readdir(results_dir))
        endswith(f, ".json") || continue
        path = joinpath(results_dir, f)
        name = splitext(f)[1]
        all_results[name] = JSON3.read(read(path, String))
    end
    return all_results
end

"""
    plot_all_results(; results_dir, output_dir)

Generate comparison bar charts from all benchmark results.

Produces:
1. Grouped bar chart: render time per scene, grouped by backend
2. Speedup chart: relative to slowest backend (or pbrt-v4 if available)
"""
function plot_all_results(;
    results_dir::String = BENCHMARK_RESULTS_DIR,
    output_dir::String = joinpath(BENCHMARK_RESULTS_DIR, "plots"),
)
    mkpath(output_dir)

    all_results = load_all_results(; results_dir)
    if isempty(all_results)
        @warn "No result files found in $results_dir"
        return nothing
    end

    # Collect all scene names and backend names
    scene_names = OrderedCollections.OrderedSet{String}()
    backend_labels = String[]
    backend_keys = String[]

    for (key, res) in all_results
        push!(backend_keys, key)
        meta = res["metadata"]
        label = "$(meta["backend_name"])"
        if haskey(meta, "hw_accel") && meta["hw_accel"]
            label *= " (HW RT)"
        end
        push!(backend_labels, label)
        for scene_name in keys(res["scenes"])
            push!(scene_names, scene_name)
        end
    end

    scene_list = collect(scene_names)
    n_scenes = length(scene_list)
    n_backends = length(backend_keys)

    if n_scenes == 0 || n_backends == 0
        @warn "No data to plot"
        return nothing
    end

    # Build timing matrix: scenes × backends
    timing_matrix = fill(NaN, n_scenes, n_backends)
    for (bi, key) in enumerate(backend_keys)
        res = all_results[key]
        for (si, scene_name) in enumerate(scene_list)
            if haskey(res["scenes"], scene_name)
                scene_data = res["scenes"][scene_name]
                if haskey(scene_data, "median")
                    timing_matrix[si, bi] = scene_data["median"]
                end
            end
        end
    end

    # --- Plot 1: Grouped bar chart ---
    fig = Figure(size=(max(800, n_scenes * 200), 500))
    ax = Axis(fig[1, 1],
        title = "RayDemo Benchmark — Render Time (seconds)",
        xlabel = "Scene",
        ylabel = "Time (s)",
        xticks = (1:n_scenes, scene_list),
        xticklabelrotation = pi/6,
    )

    bar_width = 0.8 / n_backends
    colors = Makie.wong_colors()

    for bi in 1:n_backends
        offsets = (1:n_scenes) .+ (bi - (n_backends + 1) / 2) * bar_width
        vals = timing_matrix[:, bi]
        mask = .!isnan.(vals)
        if any(mask)
            barplot!(ax, offsets[mask], vals[mask];
                width=bar_width * 0.9,
                color=colors[mod1(bi, length(colors))],
                label=backend_labels[bi],
            )
        end
    end

    Legend(fig[1, 2], ax)

    save_path = joinpath(output_dir, "benchmark_times.png")
    save(save_path, fig; px_per_unit=2)
    println("Saved: $save_path")

    # --- Plot 2: Speedup chart relative to reference ---
    # Use the slowest backend per scene as reference (1.0x)
    ref_times = [maximum(filter(!isnan, timing_matrix[si, :]); init=NaN) for si in 1:n_scenes]

    fig2 = Figure(size=(max(800, n_scenes * 200), 500))
    ax2 = Axis(fig2[1, 1],
        title = "RayDemo Benchmark — Speedup (higher is better)",
        xlabel = "Scene",
        ylabel = "Speedup (x)",
        xticks = (1:n_scenes, scene_list),
        xticklabelrotation = pi/6,
    )

    for bi in 1:n_backends
        offsets = (1:n_scenes) .+ (bi - (n_backends + 1) / 2) * bar_width
        speedups = ref_times ./ timing_matrix[:, bi]
        mask = .!isnan.(speedups)
        if any(mask)
            barplot!(ax2, offsets[mask], speedups[mask];
                width=bar_width * 0.9,
                color=colors[mod1(bi, length(colors))],
                label=backend_labels[bi],
            )
        end
    end

    hlines!(ax2, [1.0]; color=:gray, linestyle=:dash, linewidth=1)
    Legend(fig2[1, 2], ax2)

    save_path2 = joinpath(output_dir, "benchmark_speedup.png")
    save(save_path2, fig2; px_per_unit=2)
    println("Saved: $save_path2")

    return fig, fig2
end
