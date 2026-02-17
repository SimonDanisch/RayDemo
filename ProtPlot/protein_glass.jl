include(joinpath(@__DIR__, "..", "common", "common.jl"))
using ProtPlot, ProteinChains

structure = pdb"1HQK"

ground_material = Hikari.MatteMaterial(Kd=Hikari.RGBSpectrum(0.15f0, 0.15f0, 0.18f0))
protein_material = Hikari.GlassMaterial()
set_theme!(Mesh=(material=protein_material,))

function create_scene(; resolution=(1500, 1500))
    fig = Figure(size=resolution)
    ax = LScene(fig[1, 1]; show_axis=false, scenekw=(;
        lights=[
            PointLight(RGBf(15, 5, 15), Vec3f(1000)),
            Makie.SunSkyLight(Vec3f(1, 2, 8); intensity=1.0f0, turbidity=3.0f0, ground_enabled=false),
        ]
    ))

    ribbon!(ax, structure; colormap=:jet, coil_diameter=0.5, helix_width=2.2, strand_width=2.2)

    center!(ax.scene)
    cam = ax.scene.camera_controls
    lookat = cam.lookat[]

    ground_mesh = Rect3f(Point3f(lookat[1] - 1000, lookat[2] - 40, lookat[3] - 1000), Vec3f(2000, 0.1, 2000))
    mesh!(ax, ground_mesh; material=ground_material)

    cam.eyeposition[] = [165.43, -10, 233.86]
    cam.lookat[] = [148.04, 0, 126.64]
    cam.upvector[] = [0, 1, 0]
    cam.fov[] = 45.0
    Makie.update_cam!(ax.scene, cam)
    return fig, ax
end

function render_scene(;
    device=DEVICE,
    resolution=(1000, 1000),
    samples=10,
    max_depth=12,
    output_path=joinpath(@__DIR__, "protein_glass.png"),
)
    fig, ax = create_scene(; resolution=resolution)
    integrator = Hikari.VolPath(samples=samples, max_depth=max_depth)
    sensor = Hikari.FilmSensor(iso=100, white_balance=6500)
    @time result = Makie.colorbuffer(fig;
        device=device, integrator=integrator, sensor=sensor, update=false
    )
    save(output_path, result)
    @info "Saved → $output_path"
    return result
end

function render_interactive(;
    device=DEVICE,
    resolution=(800, 800),
)
    fig, ax = create_scene(; resolution=resolution)
    TraceMakie.interactive_window(fig; device=device)
    display(fig; backend=GLMakie, update=false)
end

# render_scene()
