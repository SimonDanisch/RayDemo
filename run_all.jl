# run_all.jl — smoke-test all demos at low quality
#
# Usage:
#   julia --project run_all.jl                  # auto-runs at samples=5
#   include("run_all.jl"); run_all(samples=5)   # call manually

include(joinpath(@__DIR__, "common", "common.jl"))

function run_all(;
    device=DEVICE,
    resolution=(300, 300),
    samples=5,
    max_depth=6,
    nframes=5,
    render_videos=true,
    output_folder=joinpath(@__DIR__, "all_renders"),
)
    mkpath(output_folder)
    # SandCat / Ark / koeln_flooding are prototypes; Geant4 needs a heavy
    # external toolkit. common / assets / all_renders aren't scene dirs.
    skip_dirs = Set(["common", "assets", "all_renders", ".git", ".claude",
                     "Geant4", "SandCat", "Ark", "koeln_flooding"])

    results = Tuple{String,Symbol}[]   # (label, :ok | :load_fail | :render_fail)

    for dir in sort(readdir(@__DIR__))
        dirpath = joinpath(@__DIR__, dir)
        isdir(dirpath) || continue
        dir in skip_dirs && continue
        startswith(dir, ".") && continue

        for f in reverse(sort(readdir(dirpath)))
            endswith(f, ".jl") || continue
            f == "protein_trajectory.jl" && continue # Too long to render at low quality
            filepath = joinpath(dirpath, f)
            name = splitext(f)[1]
            label = "$dir/$name"

            # Only run actual scene scripts — files that define a render entry
            # point. Helper / MWE / WIP files (no render_scene/render_video) are
            # skipped WITHOUT being included, so we never execute their top-level
            # code. Some of those (e.g. Waterlily/mwe_volume_backend.jl) do GPU
            # work at top level that hard-faults the device and would abort the
            # whole sweep — a try/catch can't recover a GPU memory fault.
            src = read(filepath, String)
            (occursin("function render_scene", src) ||
             occursin("function render_video", src)) || continue

            @info "Loading $dir/$f..."
            m = Module()
            # Seed `include` so scripts can use include() inside the anonymous module
            Core.eval(m, :(include(path) = Base.include($m, path)))
            try
                Base.include(m, filepath)
            catch e
                @warn "Failed to load $dir/$f" exception=(e, catch_backtrace())
                push!(results, (label, :load_fail))
                continue
            end

            if isdefined(m, :render_scene)
                outpath = joinpath(output_folder, "$(dir)_$(name).png")
                @info "  render_scene → $outpath"
                try
                    Base.invokelatest(m.render_scene; device, resolution, samples, max_depth, output_path=outpath)
                    push!(results, ("$label [img]", :ok))
                catch e
                    @warn "  render_scene failed" exception=(e, catch_backtrace())
                    push!(results, ("$label [img]", :render_fail))
                end
            end

            if render_videos && isdefined(m, :render_video)
                outpath = joinpath(output_folder, "$(dir)_$(name).mp4")
                @info "  render_video → $outpath"
                try
                    Base.invokelatest(m.render_video; device, resolution, samples, max_depth, nframes, output_path=outpath)
                    push!(results, ("$label [vid]", :ok))
                catch e
                    @warn "  render_video failed" exception=(e, catch_backtrace())
                    push!(results, ("$label [vid]", :render_fail))
                end
            end
        end
    end

    println("\n", "="^64)
    println("RUN_ALL SUMMARY  (samples=$samples, resolution=$resolution, nframes=$nframes)")
    println("="^64)
    for (label, status) in results
        println("  [", status === :ok ? "OK  " : "FAIL", "] ", label, "  ($status)")
    end
    nok = count(r -> r[2] === :ok, results)
    println("-"^64)
    println("  $nok/$(length(results)) ok")
    println("="^64)
    return results
end


if abspath(PROGRAM_FILE) == @__FILE__
    run_all(resolution=(300, 300), samples=5, max_depth=6, nframes=5)
end
