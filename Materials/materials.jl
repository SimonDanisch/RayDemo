# Materials Scene - 20 Material Showcase
# Grid of spheres demonstrating Hikari's material system:
# Glass, volumetrics, metals, coated materials, emissives, diffuse
#
# Showcases: Dielectric, ThinDielectric, HomogeneousMedium (milk/smoke/coffee),
# GridMedium (cloud), Conductor (gold/silver/copper), Mirror, CoatedConductor,
# CoatedDiffuse, Plastic, Emissive, DiffuseTransmission, Textures

include("../common/common.jl")
using GeometryBasics, Colors

function make_perlin_texture(resolution::Int; scale=4.0, bias=0.5, contrast=1.0)
    tex = Matrix{Float32}(undef, resolution, resolution)
    for j in 1:resolution, i in 1:resolution
        u, v = (i - 0.5) / resolution, (j - 0.5) / resolution
        n = Hikari.fbm3d(u * scale, v * scale, 0.0; octaves=4, persistence=0.5)
        tex[i, j] = Float32(clamp(bias + contrast * n, 0, 1))
    end
    return tex
end

function make_perlin_rgb_texture(resolution::Int; scale=4.0, base_color=(1.0, 1.0, 1.0), variation=0.3)
    tex = Matrix{Hikari.RGBSpectrum}(undef, resolution, resolution)
    for j in 1:resolution, i in 1:resolution
        u, v = (i - 0.5) / resolution, (j - 0.5) / resolution
        n = Hikari.fbm3d(u * scale, v * scale, 0.0; octaves=4)
        n2 = Hikari.fbm3d(u * scale + 5.3, v * scale - 2.1, 0.0; octaves=3)
        r = Float32(clamp(base_color[1] + variation * n, 0, 1))
        g = Float32(clamp(base_color[2] + variation * n2, 0, 1))
        b = Float32(clamp(base_color[3] + variation * (n + n2) * 0.5, 0, 1))
        tex[i, j] = Hikari.RGBSpectrum(r, g, b, 1.0f0)
    end
    return tex
end

function create_scene(; resolution=(1200, 900))
    lights = [
        PointLight(RGBf(60, 60, 60), Vec3f(8, 8, 10)),
        PointLight(RGBf(20, 20, 20), Vec3f(-2, -6, 3)),
    ]

    ax = Scene(; size=resolution, lights=lights, ambient=RGBf(0.02, 0.02, 0.025))
    cam3d!(ax)

    # --- Glass and transparent materials (front row) ---
    glass = Hikari.Dielectric(Kt=(1, 1, 1), index=1.5)
    thin_glass = Hikari.ThinDielectric(eta=1.5)
    glass_tint_tex = make_perlin_rgb_texture(64; scale=3.0, base_color=(0.95, 0.98, 1.0), variation=0.08)
    textured_glass = Hikari.Dielectric(Kt=Hikari.Texture(glass_tint_tex), index=1.5)

    # --- Volumetric materials ---
    milk_medium = Hikari.Milk(scale=0.1)
    milk_glass = Hikari.MediumInterface(
        Hikari.Dielectric(Kt=(1, 1, 1), index=1.5);
        inside=milk_medium, outside=nothing
    )

    smoke_medium = Hikari.Smoke(density=5.0, albedo=0.95, g=0.3)
    smoke_vol = Hikari.MediumInterface(
        Hikari.Dielectric(Kt=(1, 1, 1), index=1.0);
        inside=smoke_medium, outside=nothing
    )

    coffee_medium = Hikari.Coffee(scale=0.5)
    coffee_glass = Hikari.MediumInterface(
        Hikari.Dielectric(Kt=(0.95, 0.9, 0.85), index=1.5);
        inside=coffee_medium, outside=nothing
    )

    # Cloud (GridMedium)
    sphere_radius = 0.25f0
    spacing = 0.7f0
    nrows, ncols = 5, 4

    cloud_row, cloud_col = 2, 2
    cloud_x = (cloud_col - (ncols + 1) / 2) * spacing
    cloud_y = (cloud_row - (nrows + 1) / 2) * spacing - 4.5
    sphere_center = Vec3f(cloud_x, cloud_y, sphere_radius)
    cloud_origin = sphere_center - Vec3f(sphere_radius)
    cube_size = sphere_radius * 2

    cloud_density = Hikari.generate_cloud_density(128;
        scale=2.5, threshold=0.15, worley_weight=0.2,
        edge_sharpness=4.0, density_scale=4.5
    )
    cloud_grid = Hikari.GridMedium(
        cloud_density;
        σ_a = Hikari.RGBSpectrum(0.5f0),
        σ_s = Hikari.RGBSpectrum(15.0f0),
        g = 0.0f0,
        bounds=Hikari.Bounds3(cloud_origin, cloud_origin + Vec3f(cube_size)),
        majorant_res=Vec3i(32)
    )
    cloud_vol = Hikari.MediumInterface(
        Hikari.Dielectric(Kt=(1, 1, 1), index=1.0);
        inside=cloud_grid, outside=nothing
    )

    # --- Metals with textures ---
    gold_roughness_tex = make_perlin_texture(64; scale=6.0, bias=0.03, contrast=0.08)
    textured_gold = Hikari.Conductor(
        eta = (0.143f0, 0.374f0, 1.442f0),
        k = (3.983f0, 2.385f0, 1.603f0),
        roughness = Hikari.Texture(gold_roughness_tex)
    )

    silver = Hikari.Silver(roughness=0.02)
    copper = Hikari.Copper(roughness=0.08)
    mirror = Hikari.Mirror(Kr=(0.95, 0.95, 0.95))

    # --- Coated materials ---
    coated_gold = Hikari.Gold()
    car_paint = Hikari.CoatedConductor(
        interface_roughness=0.08,
        reflectance=(0.85, 0.1, 0.1),
        conductor_roughness=0.01
    )
    coated_blue = Hikari.CoatedDiffuse(reflectance=(0.1, 0.2, 0.7), roughness=0.05)
    plastic_white = Hikari.Plastic(color=(0.9, 0.9, 0.9), roughness=0.15)

    # --- Emissive materials ---
    emissive_white = Hikari.MediumInterface(Hikari.Emissive(Le=(4, 4, 4)))
    emissive_warm = Hikari.MediumInterface(Hikari.Emissive(Le=(2.0, 1.2, 0.5)))
    emissive_cyan = Hikari.MediumInterface(Hikari.Emissive(Le=(0.3, 1.5, 1.5)))
    emissive_pattern_tex = make_perlin_rgb_texture(64; scale=5.0, base_color=(1.5, 0.3, 1.2), variation=0.8)
    textured_emissive = Hikari.MediumInterface(Hikari.Emissive(Le=Hikari.Texture(emissive_pattern_tex)))

    # --- Simple materials ---
    diffuse_gray = Hikari.Diffuse(Kd=(0.6, 0.6, 0.6))
    paper = Hikari.DiffuseTransmission(reflectance=(0.85, 0.85, 0.85), transmittance=(0.4, 0.4, 0.4))

    # ========================================================================
    # Arrange materials in 5x4 grid
    # ========================================================================
    materials = [
        glass          textured_glass  milk_glass     smoke_vol;
        emissive_white cloud_vol       coffee_glass   thin_glass;
        textured_gold  silver          copper         mirror;
        coated_gold    car_paint       coated_blue    plastic_white;
        emissive_warm  paper           diffuse_gray   textured_emissive
    ]

    # Floor
    floor_material = Hikari.Diffuse(Kd=(0.7, 0.7, 0.7))
    floor_mesh = Rect3f(Vec3f(-10, -10, -0.001), Vec3f(20, 20, 0.001))
    mesh!(ax, floor_mesh; material=floor_material)

    # Place spheres in grid
    for i in CartesianIndices(materials)
        row, col = Tuple(i)
        mat = materials[i]
        x = (col - (ncols + 1) / 2) * spacing
        y = (row - (nrows + 1) / 2) * spacing - 4.5
        pos = Point3f(x, y, sphere_radius)
        mesh!(ax, Sphere(pos, sphere_radius), material=mat)
    end

    # Camera
    cam = cameracontrols(ax)
    cam.eyeposition[] = Vec3f(0, -7.5, 2.5)
    cam.lookat[] = Vec3f(0, -4.7, 0)
    cam.upvector[] = Vec3f(0, 0, 1)
    cam.fov[] = 42
    update_cam!(ax, cam)
    return ax
end

# =============================================================================
# Standardized RayDemo render_scene API
# =============================================================================

function render_scene(;
    device=DEVICE,
    resolution=(1200, 900),
    samples=10,
    max_depth=50,
    output_path=joinpath(@__DIR__, "output", "materials.png"),
    hw_accel=false,
)
    scene = create_scene(; resolution=resolution)
    RayMakie.activate!(; device=device, exposure=0.6f0, tonemap=:aces, gamma=2.2f0)
    integrator = Hikari.VolPath(; samples=samples, max_depth=max_depth, hw_accel=hw_accel)
    @time img = colorbuffer(scene; backend=RayMakie, integrator=integrator, update=false)
    mkpath(dirname(output_path))
    save(output_path, img)
    @info "Saved → $output_path"
    return img
end

if abspath(PROGRAM_FILE) == @__FILE__
    using Lava
    DEVICE = Lava.LavaBackend()
    scene = create_scene()
    RayMakie.vulkan_viewer(scene)
    cam = cameracontrols(scene)
    cam.eyeposition[] = Vec3f(0, -7.5, 2.5)
    cam.lookat[] = Vec3f(0, -4.7, 0)
    cam.upvector[] = Vec3f(0, 0, 1)
    cam.fov[] = 42
    update_cam!(scene, cam)
    colorbuffer(scene; update=false)
    data_limits(scene)
end
