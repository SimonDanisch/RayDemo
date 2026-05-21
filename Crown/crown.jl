# Crown Scene — pbrt-v4 port via RayMakie.pbrt_to_makie
#
# Parses crown.pbrt using Hikari's full pbrt-v4 parser (materials, area lights
# with blackbody spectra, homogeneous media, 786 PLY meshes).
include("../common/common.jl")

const CROWN_DIR = @__DIR__
@assert isfile(joinpath(CROWN_DIR, "crown.pbrt")) "crown.pbrt not found in $CROWN_DIR"

function create_scene(; resolution=nothing)
    result = RayMakie.pbrt_to_makie(joinpath(CROWN_DIR, "crown.pbrt"))
    scene = result.scene
    resolution !== nothing && resize!(scene, resolution)
    return scene
end

function render_scene(;
    device=DEVICE,
    resolution=(800, 800),
    samples=100,
    max_depth=16,
    output_path=joinpath(@__DIR__, "output", "crown.png"),
    hw_accel=false,
)
    result = RayMakie.pbrt_to_makie(joinpath(CROWN_DIR, "crown.pbrt"))
    # Override resolution if requested
    scene = resolution === nothing ? result.scene :
        begin
            s = result.scene
            resize!(s, resolution)
            s
        end

    RayMakie.activate!(; device=device)
    integrator = Hikari.VolPath(; samples=samples, max_depth=max_depth, hw_accel=hw_accel)
    @time img = colorbuffer(scene; backend=RayMakie, integrator=integrator, update=false)
    mkpath(dirname(output_path))
    save(output_path, img)
    @info "Saved → $output_path"
    return img
end

if abspath(PROGRAM_FILE) == @__FILE__
    using Lava
    scene = create_scene()
    screen = display(scene; backend=RayMakie, update=false, device=Lava.LavaBackend(),
        integrator=Hikari.VolPath(max_depth=8, samples=1, hw_accel=true))
end
