include(joinpath(@__DIR__, "..", "common", "common.jl"))
using PlantGeom, XPalm, XPalm.VPalm
import PlantGeom: symbol, descendants

# Materials
leaf_material = Hikari.CoatedDiffuseTransmissionMaterial(
    reflectance=(0.07f0, 0.15f0, 0.04f0),
    transmittance=(0.03f0, 0.08f0, 0.02f0),
    roughness=0.15f0,   # waxy cuticle (slightly rough for broad highlights)
    eta=1.5f0,           # typical plant cuticle IOR
    thickness=0.01f0
)
stem_material = Hikari.CoatedDiffuseMaterial(
    reflectance=(0.55f0, 0.35f0, 0.2f0), roughness=0.4f0, eta=1.5f0, thickness=0.001f0
)
ground_material = Hikari.MatteMaterial(Kd=Hikari.RGBSpectrum(0.15f0, 0.12f0, 0.08f0))

# Build plant model
file = joinpath(dirname(dirname(pathof(XPalm))), "test", "references", "vpalm-parameter_file.yml")
parameters = read_parameters(file)
mtg = build_mockup(parameters; merge_scale=:leaflet)

PlantGeom.traverse!(mtg) do node
    if symbol(node) == "Petiole"
        segs = descendants(node, symbol=["PetioleSegment", "RachisSegment"])
        colormap = cgrad([colorant"sienna", colorant"burlywood"], length(segs), scale=:log2)
        for (i, seg) in enumerate(segs)
            seg[:color_type] = colormap[i]
        end
    elseif symbol(node) == "Leaflet"
        node[:color_type] = nothing
    elseif symbol(node) == "Leaf"
        node[:color_type] = nothing
    end
end

function create_scene(; resolution=(1600, 1200))
    fig = Scene(; size=resolution,
        lights = [Makie.SunSkyLight(Vec3f(1, 2, 8); intensity=1.0f0, turbidity=3.0f0, ground_enabled=false)]
    )
    cam3d!(fig)
    leaflet_plot = plantviz!(fig, mtg, symbol="Leaflet")
    stem_plot = plantviz!(fig, mtg, color=:color_type, symbol=["RachisSegment", "PetioleSegment", "Internode"])
    base_plot = plantviz!(fig, mtg, color=:color_type, symbol="Leaf")
    leaflet_plot.plots[1].material = leaf_material
    stem_plot.plots[1].material = stem_material
    base_plot.plots[1].material = stem_material

    Makie.mesh!(fig, Rect3f(Vec3f(-20, -20, -0.05), Vec3f(40, 40, 0.05)); material=ground_material)

    center!(fig)
    cam = fig.camera_controls
    cam.eyeposition[] = Vec3f(12, 0, 4)
    cam.lookat[] = Vec3f(0, 0, 3)
    cam.upvector[] = Vec3f(0, 0, 1)
    cam.fov[] = 45f0
    update_cam!(fig, cam)
    return fig
end

function render_scene(;
    device=DEVICE,
    resolution=(1600, 1200),
    samples=1000,
    max_depth=30,
    output_path=joinpath(@__DIR__, "plants.png"),
)
    fig = create_scene(; resolution=resolution)
    integrator = Hikari.VolPath(samples=samples, max_depth=max_depth)
    sensor = Hikari.FilmSensor(iso=100, white_balance=6500)
    @time result = Makie.colorbuffer(fig;
        device=device, integrator=integrator, sensor=sensor, update=false
    )
    save(output_path, result)
    @info "Saved → $output_path"
    return result
end

# render_scene()
