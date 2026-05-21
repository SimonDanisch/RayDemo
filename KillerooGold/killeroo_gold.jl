# Killeroo Gold Scene — pbrt-v4 port via RayMakie.pbrt_to_makie
#
# Parses killeroo-gold.pbrt using Hikari's full pbrt-v4 parser (Gold conductor,
# loopsubdiv mesh at 3 levels, area light, textured floor/walls).

include("../common/common.jl")

const KILLEROO_DIR = @__DIR__
@assert isfile(joinpath(KILLEROO_DIR, "killeroo-gold.pbrt")) "killeroo-gold.pbrt not found in $KILLEROO_DIR"

function create_scene(; resolution=nothing)
    result = RayMakie.pbrt_to_makie(joinpath(KILLEROO_DIR, "killeroo-gold.pbrt"))
    scene = result.scene
    resolution !== nothing && resize!(scene, resolution)
    return scene
end

function render_scene(;
    device=DEVICE,
    resolution=(684, 513),
    samples=64,
    max_depth=8,
    output_path=joinpath(@__DIR__, "output", "killeroo_gold.png"),
    hw_accel=false,
)
    result = RayMakie.pbrt_to_makie(joinpath(KILLEROO_DIR, "killeroo-gold.pbrt"))
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
    screen = RayMakie.vulkan_viewer(scene;
        integrator=Hikari.VolPath(max_depth=8, samples=64, hw_accel=true))
end
