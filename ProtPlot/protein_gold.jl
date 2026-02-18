include("../common/common.jl")
using ProtPlot, ProteinChains

function create_scene(; resolution=(800, 800))
    structure = pdb"7PKZ"
    r = 100
    ground_material = Hikari.MatteMaterial(Kd=Hikari.RGBSpectrum(0.15f0, 0.15f0, 0.18f0))
    protein_material = Hikari.Gold(roughness=0.1f0)
    set_theme!(Mesh=(material=protein_material,))
    ax = Scene(size=resolution; lights=[
        Makie.SunSkyLight(Vec3f(0.4, -0.3, 0.7); intensity=1.0f0, turbidity=3.0f0, ground_enabled=false),
        Makie.PointLight(RGBf(50 * r, 20 * r, 20 * r), Vec3f(30.39, 631.13, 260.65)),
    ])
    cam3d!(ax)

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

    cam = ax.camera_controls
    cam.eyeposition[] = [243.88, 351.91, 623.44]
    cam.lookat[] = [243.88, 355.07, 379.38]
    cam.upvector[] = [-1.0, 0.0, 0.0]
    cam.fov[] = 40.0
    cam.lens_radius[] = 0.3
    cam.focal_distance[] = 100.0
    Makie.update_cam!(ax, cam)
    return ax
end

function render_scene(;
    device=DEVICE,
    resolution=(800, 800),
    samples=1000,
    max_depth=12,
    output_path=joinpath(@__DIR__, "protein_gold.png"),
)
    GC.gc(true)
    fig = create_scene(; resolution=resolution)
    integrator = Hikari.VolPath(samples=samples, max_depth=max_depth)
    sensor = Hikari.FilmSensor(iso=80, white_balance=6500)
    @time result = Makie.colorbuffer(fig;
        device=device, integrator=integrator, sensor=sensor, update=false
    )
    save(output_path, result)
    @info "Saved → $output_path"
    return result
end

# render_scene()
