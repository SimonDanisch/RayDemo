include("../common/common.jl")
using ProtPlot, ProteinChains

structure = pdb"1HQK"

ground_material = Hikari.MatteMaterial(Kd=Hikari.RGBSpectrum(0.15f0, 0.15f0, 0.18f0))
protein_material = Hikari.GlassMaterial()
set_theme!(Mesh=(material=protein_material,))

function create_scene(; resolution=(1500, 1500), intensity=1f0)
    scene = Scene(size=resolution, lights=[
            PointLight(RGBf(4, 3, 1) * 1000, Vec3f(50), 1f0),
            Makie.SunSkyLight(Vec3f(1, 2, 9); intensity=intensity, turbidity=3f0, ground_enabled=false),
        ]
    )
    cam3d!(scene)
    ribbon!(scene, structure; colormap=:jet, coil_diameter=0.5, helix_width=2.2, strand_width=2.2)

    center!(scene)
    cam = scene.camera_controls
    lookat = cam.lookat[]

    ground_mesh = Rect3f(Point3f(lookat[1] - 1000, lookat[2] - 40, lookat[3] - 1000), Vec3f(2000, 0.1, 2000))
    mesh!(scene, ground_mesh; material=ground_material)

    cam.eyeposition[] = [165.43, -10, 233.86]
    cam.lookat[] = [148.04, 0, 126.64]
    cam.upvector[] = [0, 1, 0]
    cam.fov[] = 45.0
    Makie.update_cam!(scene, cam)
    return scene
end

function render_scene(;
    device=DEVICE,
    resolution=(1000, 1000),
    samples=10,
    intensity=1f0,
    max_depth=12,
    output_path=joinpath(@__DIR__, "protein_glass.png"),
)
    scene = create_scene(; intensity=intensity, resolution=resolution)
    integrator = Hikari.VolPath(samples=samples, max_depth=max_depth)
    sensor = Hikari.FilmSensor(iso=100, white_balance=6500)
    @time result = Makie.colorbuffer(scene;
        tonemap=:aces,
        device=device, integrator=integrator, sensor=sensor, update=false
    )
    isnothing(output_path) || save(output_path, result)
    @info "Saved → $output_path"
    return result
end

function render_interactive(;
    device=DEVICE,
    resolution=(800, 800),
)
    scene = create_scene(; resolution=resolution)
    RayMakie.interactive_window(scene; device=device)
    display(scene; backend=GLMakie, update=false)
end
render_scene(samples=1000)
