# WaterLily Sine Wave — Raytraced Ocean Demo
# Marching cubes isosurface + Laplacian smoothing for artifact-free water surface.
# GlassMaterial (IOR 1.33) gives Fresnel reflections + refraction → realistic water.
# HomogeneousMedium inside for volumetric blue color at depth.

using GLMakie, RayMakie, Hikari
using ReadVTK, Meshing, GeometryBasics, StaticArrays, Statistics
using ImageFiltering, ImageTransformations
using GeometryBasics: Point3f, Vec3f, GLTriangleFace, Rect3f, normal_mesh
using FileIO

function load_wave_vti(path)
    vtk = VTKFile(path)
    pd = get_point_data(vtk)
    f = Float32.(get_data(pd["f"]))
    nx = ny = nz = 258
    return reshape(f, nx, ny, nz)
end

"""
    extract_smooth_surface(f3d; iso=0.5f0, upsample=3, σ=1.1f0, depth=50)

Extract a closed thick isosurface mesh from volume data.
Zeroes the volume boundaries so marching tetrahedra naturally closes the mesh.
`depth` controls how many z-slices below the surface to include for thickness.
"""
function extract_smooth_surface(f3d; iso=0.5f0, upsample=3, σ=1.1f0, depth=50)
    nx, ny, nz = size(f3d)

    # Find where the surface lives (z range with 0 < mean < 1)
    z_means = [mean(@view f3d[:, :, z]) for z in 1:nz]
    z_surface = findfirst(m -> m < 1.0f0 - iso, z_means)
    z_top = findlast(m -> m > iso, z_means)
    z_bot = max(1, z_surface - depth)
    margin_top = 3

    f_cropped = f3d[:, :, z_bot:min(nz - margin_top, z_top + margin_top)]

    # Zero out boundaries so the isosurface closes naturally
    f_vol = copy(f_cropped)
    f_vol[:, :, 1] .= 0f0       # bottom
    f_vol[:, :, end] .= 0f0     # top
    f_vol[1, :, :] .= 0f0       # x edges
    f_vol[end, :, :] .= 0f0
    f_vol[:, 1, :] .= 0f0       # y edges
    f_vol[:, end, :] .= 0f0

    # 1. Gaussian blur
    @info "Gaussian filtering volume (σ=$σ)..."
    f_smooth = imfilter(f_vol, Kernel.gaussian((σ, σ, σ)))

    # 2. Upsample to finer grid
    nx_up, ny_up, nz_up = upsample .* size(f_smooth)
    @info "Upsampling volume $(size(f_smooth)) → ($nx_up, $ny_up, $nz_up)..."
    f_fine = Float32.(imresize(f_smooth, (nx_up, ny_up, nz_up)))

    X = LinRange(0f0, Float32(nx-1), nx_up)
    Y = LinRange(0f0, Float32(ny-1), ny_up)
    Z = LinRange(Float32(z_bot-1), Float32(z_bot - 1 + size(f_vol, 3) - 1), nz_up)

    @info "Running marching tetrahedra on upsampled volume..."
    verts_raw, faces_raw = Meshing.isosurface(f_fine, MarchingTetrahedra(iso=iso), X, Y, Z)
    @info "Mesh: $(length(verts_raw)) vertices, $(length(faces_raw)) triangles"

    verts = Point3f.(verts_raw)
    faces = GLTriangleFace.(faces_raw)
    mean_z = mean(v -> v[3], verts)
    scale_f = 10f0 / Float32(nx - 1)
    cx, cy = Float32(nx - 1) / 2, Float32(ny - 1) / 2
    offset = Vec3f(-cx, -cy, -mean_z)
    normalized = map(v -> (v .- offset) * scale_f, verts)

    water_mesh = normal_mesh(normalized, faces)
    return water_mesh, scale_f, mean_z
end

# =============================================================================
# Materials
# =============================================================================

# Water surface: physically correct dielectric (Fresnel handles reflection split)
# No volumetric medium — color comes from Fresnel sky reflections + tinted transmission
water_material = Hikari.GlassMaterial(
    Kr = Hikari.RGBSpectrum(1f0, 1f0, 1f0),
    Kt = Hikari.RGBSpectrum(0.7f0, 0.85f0, 0.95f0),    # blue tint on transmission
    roughness = 0.001f0,                                   # slight surface roughness
    index = 1.33f0,                                       # water IOR
)

# =============================================================================
# Scene Construction
# =============================================================================

function build_wave_scene(;
    vti_path = joinpath(@__DIR__, "MultidiagSineWave3DN256CMOM-CoMaFL-DirSplit_000636.vti"),
    resolution = (1920, 1080),
    upsample = 3,
    σ = 1.5f0,
)
    @info "Loading VTI data..."
    f3d = load_wave_vti(vti_path)

    water_mesh, scale, water_z = extract_smooth_surface(f3d;
        upsample, σ)

    # Scene
    fig = Figure(; size=resolution)
    ax = LScene(fig[1, 1]; show_axis=false, scenekw=(;
        lights=[
            Makie.SunSkyLight(
                Vec3f(0.4, -0.3, 0.5);
                intensity = 2.0f0,
                turbidity = 2.5f0,
                ground_enabled = false,
            ),
        ]
    ))

    # Flat ocean surface extending to the horizon
    S = 200f0
    # ocean_bg = mesh!(ax,
    #     Rect3f(Vec3f(-S, -S, -0.02f0), Vec3f(2S, 2S, 0.04f0));
    #     color = RGBf(0.1, 0.3, 0.5),
    # )
    # ocean_bg.material[] = water_material

    # Water surface (smoothed marching cubes isosurface)
    water_plot = mesh!(ax, water_mesh;
        color = RGBf(0.1, 0.3, 0.5),
        material=water_material
    )
    # Ocean floor — sandy bottom, not too deep so caustics are visible
    # floor_z = -water_z - 2f0
    # sandy_floor = Hikari.MatteMaterial(Kd=(0.15f0, 0.13f0, 0.1f0), σ=20f0)
    # mesh!(ax, Rect3f(Vec3f(-S, -S, floor_z-0.01f0), Vec3f(2S, 2S, 0.02f0));
    #     color=RGBf(0.15, 0.13, 0.1), material=sandy_floor)

    return fig, ax
end

# =============================================================================
# Rendering
# =============================================================================

function render_image(;
    resolution = (1920, 1080),
    samples = 128,
    max_depth = 16,
    kwargs...
)
    fig, ax = build_wave_scene(; resolution, kwargs...)

    integrator = Hikari.VolPath(;
        samples, max_depth,
        max_component_value = 10f0,
        regularize = true, hw_accel = true,
    )
    sensor = Hikari.FilmSensor(; iso=150, white_balance=7500)

    # First colorbuffer triggers Makie.center! — init then override camera
    @info "Initializing..."
    init_int = Hikari.VolPath(; samples=1, max_depth=4, hw_accel=true)


    # cam = ax.scene.camera_controls
    # cam.eyeposition[] = Vec3f(5.5, -6.5, 2.5)
    # cam.lookat[]      = Vec3f(-0.5, 0.5, -0.3)
    # cam.upvector[]    = Vec3f(0, 0, 1)
    # cam.fov[]         = 42f0
    # update_cam!(ax.scene, cam)

    @info "Rendering $(samples) spp with HW RT..."
    img = colorbuffer(ax.scene;
        integrator, sensor,
        tonemap = :aces, gamma = 2.2f0, exposure = 0.6f0,
    )
    return fig, img
end

function render_interactive(; resolution=(1600, 900), kwargs...)
    fig, ax = build_wave_scene(; resolution, kwargs...)
    integrator = Hikari.VolPath(;
        samples=1, max_depth=12,
        max_component_value=10f0, regularize=true, hw_accel=true,
    )
    sensor = Hikari.FilmSensor(; iso=150, white_balance=7500)
    vulkan_viewer(fig; integrator, sensor,
        tonemap=:aces, gamma=2.2f0, exposure=0.6f0)
    return fig, ax
end

begin
    GC.gc(true)
    fig, img = render_image(; samples=10, max_depth=20, σ=1.0f0, resolution=(1000, 800))
    img
end

# save(joinpath(@__DIR__, "wave_ocean_hq.png"), img)
