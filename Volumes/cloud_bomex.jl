# BOMEX Cloud Visualization — Disney Cloud Style
# Volumetric ray tracing of LES cloud data from Oceananigans BOMEX simulation
# Scene setup follows pbrt-v4-scenes/disney-cloud/disney-cloud.pbrt
#
# Requires: bomex_1024.nanovdb (generate with convert_bomex.jl)

include("../common/common.jl")
using GeometryBasics

function create_scene(nvdb_path; resolution=(1280, 720))
    s = Scene(size=resolution; lights=[Makie.SunSkyLight(Vec3f(1, 2, 8); intensity=2.0f0, turbidity=3.0f0, ground_enabled=false)])
    cam3d!(s)

    # Load cloud medium from pre-converted NanoVDB file
    # Data was rescaled during conversion: extent/1000, density*100
    cloud_medium = Hikari.NanoVDBMedium(nvdb_path;
        σ_a=Hikari.RGBSpectrum(0.0f0),
        σ_s=Hikari.RGBSpectrum(1500f0),
        majorant_res=Vec3i(64, 64, 64),
        g=0.8f0,
    )

    # Get world bounds from the medium for the enclosing geometry
    cloud_origin = Point3f(cloud_medium.bounds.p_min)
    grid_extent = Vec3f(cloud_medium.bounds.p_max - cloud_medium.bounds.p_min)

    # Transparent boundary enclosing the medium
    transparent = Hikari.GlassMaterial(
        Kr=Hikari.RGBSpectrum(0f0), Kt=Hikari.RGBSpectrum(1f0), index=1.0f0
    )
    volume_material = Hikari.MediumInterface(transparent; inside=cloud_medium, outside=nothing)
    mesh!(s, Rect3f(cloud_origin, grid_extent); material=volume_material)

    # Camera — perspective view matching PBRT Disney cloud (fov 31)
    cam_pos = grid_extent .* Vec3f(0.5, 0.5, -0.4)
    look_at = Vec3f(0, 0, grid_extent[3])
    update_cam!(s, cam_pos, look_at, Vec3f(0, 0, 1))
    s.camera_controls.fov[] = 31.0
    return s
end

function render_scene(;
    device=DEVICE,
    resolution=(2880, 1000),
    samples=100,
    max_depth=100,
    output_path=joinpath(@__DIR__, "cloud_bomex.png"),
)
    s = create_scene(joinpath(@__DIR__, "bomex_1024.nanovdb"); resolution=resolution)
    integrator = Hikari.VolPath(samples=samples, max_depth=max_depth)
    sensor = Hikari.FilmSensor(iso=100, white_balance=6500)
    result = Makie.colorbuffer(s;
        device=device, integrator=integrator, sensor=sensor, update=false
    )
    save(output_path, result)
    @info "Saved → $output_path"
    return result
end

# render_scene()
