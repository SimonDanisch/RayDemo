# run_all.jl — Test all demos at low quality
#
# Usage:
#   include("run_all.jl")
#   run_all(resolution=(500,500), samples=20, nframes=10)

include(joinpath(@__DIR__, "common", "common.jl"))

function run_all(;
    device=DEVICE,
    resolution=(500, 500),
    samples=20,
    max_depth=6,
    nframes=10,
    output_folder=joinpath(@__DIR__, "all_renders"),
)
    mkpath(output_folder)
    skip_dirs = Set(["common", "assets", ".git", ".claude" ,"Geant4"])

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

            @info "Loading $dir/$f..."
            m = Module()
            # Seed `include` so scripts can use include() inside the anonymous module
            Core.eval(m, :(include(path) = Base.include($m, path)))
            try
                Base.include(m, filepath)
            catch e
                @warn "Failed to load $dir/$f" exception=(e, catch_backtrace())
                continue
            end

            if isdefined(m, :render_scene)
                outpath = joinpath(output_folder, "$(dir)_$(name).png")
                @info "  render_scene → $outpath"
                try
                    Base.invokelatest(m.render_scene; device, resolution, samples, max_depth, output_path=outpath)
                catch e
                    @warn "  render_scene failed" exception=(e, catch_backtrace())
                end
            end

            if isdefined(m, :render_video)
                outpath = joinpath(output_folder, "$(dir)_$(name).mp4")
                @info "  render_video → $outpath"
                try
                    Base.invokelatest(m.render_video; device, resolution, samples, max_depth, nframes, output_path=outpath)
                catch e
                    @warn "  render_video failed" exception=(e, catch_backtrace())
                end
            end
        end
    end
end


run_all(
    resolution=(300, 300),
    samples=15,
    max_depth=6,
    nframes=5,
)
