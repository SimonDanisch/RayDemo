include("../common/common.jl")
using ProtPlot, ProteinChains

structure = pdb"7PKZ"

ground_material = Hikari.MatteMaterial(Kd=Hikari.RGBSpectrum(0.15f0, 0.15f0, 0.18f0))
protein_material = Hikari.CoatedDiffuseMaterial(
    reflectance=(0.95f0, 0.95f0, 0.95f0), roughness=0.05f0, eta=1.5f0, thickness=0.01f0
)
set_theme!(Mesh=(material=protein_material,))

function create_scene(; resolution=(800, 800))
    fig = Figure(size=resolution)
    ax = LScene(fig[1, 1]; show_axis=false, scenekw=(;
        backgroundcolor=RGBf(0.02, 0.02, 0.03),
        lights=[
            Makie.SunSkyLight(Vec3f(0.4, -0.3, 0.7); intensity=1.0f0, turbidity=3.0f0, ground_enabled=false),
        ]
    ))

    rb = ribbon!(ax, structure;
        colormap=:jet, coil_diameter=0.5, helix_width=2.2, strand_width=2.2,
        coil_spline_quality=4, helix_spline_quality=4, strand_spline_quality=4, show_gaps=false
    )

    dl = Makie.data_limits(rb)
    origin = minimum(dl)
    widths = Makie.widths(dl)
    center = origin .+ widths ./ 2
    radius = maximum(widths) / 2

    ps = 3000
    mesh!(ax, Rect3f(Point3f(center[1] - ps, center[2] - ps, origin[3] - 5), Vec3f(2ps, 2ps, 0.1));
        material=ground_material)

    cam = ax.scene.camera_controls
    cam.lookat[] = center
    cam.eyeposition[] = center .+ Vec3f(2.8, 0.5, 1.0) .* radius
    cam.upvector[] = Vec3f(0, 0, 1)
    cam.fov[] = 40.0
    Makie.update_cam!(ax.scene, cam)
    return fig, ax
end

function render_scene(;
    device=DEVICE,
    resolution=(1000, 1000),
    samples=10,
    max_depth=12,
    output_path=joinpath(@__DIR__, "protein_7pkz.png"),
)
    fig, ax = create_scene(; resolution=resolution)
    integrator = Hikari.VolPath(samples=samples, max_depth=max_depth)
    sensor = Hikari.FilmSensor(iso=50, white_balance=6500)
    @time result = Makie.colorbuffer(ax.scene;
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
    display(fig; backend=GLMakie)
end

# render_scene()
