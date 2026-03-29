# WaterLily Sphere Wake — Raytraced Smoke Visualization
# Realistic cloud-like look with RGBGridMedium colormap

include("../common/common.jl")
include(joinpath(@__DIR__, "..", "common", "waterlily_simulation.jl"))
using GeometryBasics

#==============================================================================#
#                            SCENE CONSTRUCTION                                #
#==============================================================================#

# Smoke colormap: transparent at low density, near-white at high density.
# Alpha ramp controls where smoke appears; RGB is near-white for realistic look.
smoke_cmap = [
    RGBAf(1, 1, 1, 0.0),     # transparent
    RGBAf(1, 1, 0.93, 0.1),      # faint wisps
    RGBAf(0.93, 0.93, 0.93, 0.4),     # light haze
    RGBAf(0.97, 0.96, 0.98, 0.7),     # medium smoke
    RGBAf(1.0, 1, 0.9, 0.9),        # dense core
]

"""
    preprocess_vorticity(vorticity; threshold=0.05f0, ref_max=nothing) -> Array{Float32,3}

Normalize and threshold vorticity for volume rendering.
"""
function preprocess_vorticity(vorticity; threshold=0.05f0, ref_max=nothing)
    rmax = something(ref_max, maximum(vorticity))
    vort_normalized = vorticity ./ max(rmax, 1f-10)
    return map(vort_normalized) do v
        v > threshold ? (v - threshold) / (1f0 - threshold) : 0f0
    end
end

"""
    pad_volume(data; pad=8) -> Array{Float32,3}

Pad volume data with zeros on all sides to hide bounding box edges.
"""
function pad_volume(data; pad=8)
    nx, ny, nz = size(data)
    padded = zeros(Float32, nx + 2pad, ny + 2pad, nz + 2pad)
    padded[pad+1:pad+nx, pad+1:pad+ny, pad+1:pad+nz] .= data
    return padded
end

const VOL_PAD = 8  # voxels of zero-padding around volume

function build_scene(vorticity, params;
    resolution=(900, 500),
    extinction_scale=12f0,
    asymmetry_g=0.85f0,
    single_scatter_albedo=0.999f0,
    vorticity_threshold=0.05f0,
    vorticity_ref_max=nothing,
    colormap=smoke_cmap)

    (; m, R) = derived_params(params)
    n = 5m ÷ 2  # Domain length

    # Preprocess and pad vorticity (padding hides bounding box edges)
    vort_clean = preprocess_vorticity(vorticity;
        threshold=vorticity_threshold, ref_max=vorticity_ref_max)
    nx, ny, nz = size(vort_clean)
    vort_padded = pad_volume(vort_clean; pad=VOL_PAD)

    # Front-lit sun with clean blue sky
    lights = [
        SunSkyLight(Vec3f(-0.3, 0.6, 0.6);
            intensity=2.5f0,
            turbidity=2.0f0,
            ground_enabled=false),
        PointLight(RGBf(10000, 8000, 1000), Vec3f(60, -30, 40))
    ]

    fig = Figure(; size=resolution)
    ax = LScene(fig[1, 1]; show_axis=false, scenekw=(; lights))

    # Volume intervals — expanded to account for padding
    dx, dy, dz = Float64(n) / nx, Float64(m) / ny, Float64(m) / nz
    pad = VOL_PAD
    x_interval = -pad*dx .. Float64(n) + pad*dx
    y_interval = -pad*dy .. Float64(m) + pad*dy
    z_interval = -pad*dz .. Float64(m) + pad*dz

    # Volume with colormap → RGBGridMedium (supports in-place density updates)
    vol_plot = volume!(ax, x_interval, y_interval, z_interval, vort_padded;
        material=(;
            extinction_scale=extinction_scale,
            asymmetry_g=asymmetry_g,
            single_scatter_albedo=single_scatter_albedo,
        ),
        colormap=colormap,
        algorithm=:absorption,
        transparency=true,
    )

    # Dark matte sphere
    sphere_center = Point3f(m ÷ 2, m ÷ 2, m ÷ 2)
    sphere_mesh = GeometryBasics.normal_mesh(Sphere(sphere_center, Float32(R)))
    sphere_material = Hikari.CoatedDiffuseMaterial(
        reflectance=(0.07f0, 0.06f0, 0.01f0),
        roughness=0.1f0,
        eta=1.5f0,
    )
    mesh!(ax, sphere_mesh; color=RGBf(0.07, 0.06, 0.01), material=sphere_material)

    # Dark reflective ground plane at z=0
    S = 5000f0
    cx, cy = Float32(n / 2), Float32(m / 2)
    gz = -0.5f0
    gv = Point3f[
        Point3f(cx - S, cy - S, gz), Point3f(cx + S, cy - S, gz),
        Point3f(cx + S, cy + S, gz), Point3f(cx - S, cy + S, gz),
    ]
    gf = GLTriangleFace[(1, 2, 3), (1, 3, 4)]
    ground_mesh = GeometryBasics.normal_mesh(gv, gf)
    ground_material = Hikari.CoatedDiffuseMaterial(
        reflectance=(0.1f0, 0.1f0, 0.12f0),
        roughness=0.1f0,
        eta=1.5f0,
    )
    mesh!(ax, ground_mesh; color=RGBf(1, 0.8, 0.3), material=ground_material)

    # Camera: side view from downstream, looking into the wake
    center = Vec3f(n * 0.5, m / 2, m * 0.35)
    cam_pos = Vec3f(n * 0.5, m * 3.5, m * 0.65)
    cam = ax.scene.camera_controls
    cam.eyeposition[] = cam_pos
    cam.lookat[] = center
    cam.upvector[] = Vec3f(0, 0, 1)
    cam.fov[] = 35f0
    update_cam!(ax.scene, cam)

    return fig, ax, vol_plot
end

#==============================================================================#
#                            ANIMATION WITH Makie.record                       #
#==============================================================================#

"""
    record_animation(sim, params=PARAMS; duration=100, step=0.5, kwargs...)

Record an animation of the simulation advancing through time.
Uses `Makie.record_longrunning` with in-place RGBGridMedium density updates.
"""
function record_animation(sim, params=PARAMS;
    device=DEVICE,
    duration=100,
    step=0.5,
    filename=joinpath(@__DIR__, "smoke_wake.mp4"),
    framerate=30,
    compression=18,
    samples=50,
    max_depth=10,
    exposure=0.6f0,
    gamma=2.2f0,
    tonemap=:aces,
    iso=100,
    kwargs...)

    # Allocate SGS arrays
    S = prepare_sgs(sim, params)

    # Extract first frame and establish reference max for consistent brightness
    vort_first = extract_vorticity(sim)
    ref_max = maximum(vort_first)

    # Build scene with first frame data
    fig, ax, vol_plot = build_scene(vort_first, params; kwargs...)

    # Configure rendering
    integrator = Hikari.VolPath(;
        samples=samples,
        max_depth=max_depth,
        max_component_value=10f0,
        regularize=true,
    )
    sensor = Hikari.FilmSensor(; iso=iso, white_balance=6500)

    # Time range for animation
    t₀ = WaterLily.sim_time(sim)
    times = range(t₀, t₀ + duration; step=step)
    n_total = length(times)
    # Precompute threshold from build_scene defaults (or kwargs)
    vorticity_threshold = get(kwargs, :vorticity_threshold, 0.05f0)
    @info "Recording $n_total frames to $filename"
    colorbuffer(ax.scene; device=device, integrator=integrator, sensor=sensor,
        exposure=exposure, tonemap=tonemap, gamma=gamma, update=false)  # warm up GPU and cache shaders
    Makie.record_longrunning(ax.scene, filename, enumerate(times); overwrite=true, update=false,
        framerate=framerate, compression=compression,
        device=device, integrator=integrator, sensor=sensor,
        exposure=exposure, tonemap=tonemap, gamma=gamma) do (i, t)

        # Advance simulation
        sim_step!(sim, t; remeasure=false)
        # Extract, preprocess, and pad vorticity (must match padded dimensions)
        vort = extract_vorticity(sim)
        vort_clean = preprocess_vorticity(vort;
            threshold=vorticity_threshold, ref_max=ref_max)

        # Update volume density in-place (triggers RGBGridMedium update, no rebuild)
        vol_plot[4] = pad_volume(vort_clean; pad=VOL_PAD)
    end

    @info "Animation saved to $filename"
    return filename
end

#==============================================================================#
#                                   USAGE                                      #
#==============================================================================#

function render_video(;
    device=DEVICE,
    resolution=(900, 500),
    samples=50,
    max_depth=10,
    nframes=200,
    output_path=joinpath(@__DIR__, "smoke_wake.mp4"),
)
    sim = load_simulation(100)
    record_animation(sim;
        device=device,
        resolution=resolution,
        samples=samples,
        max_depth=max_depth,
        filename=output_path,
        duration=nframes * 0.5,
        step=0.5,
    )
    @info "Saved → $output_path"
end

# Single frame example:
# vorticity = extract_vorticity(sim)
# fig, ax, _ = build_scene(vorticity, PARAMS; kwargs...)
# integrator = Hikari.VolPath(; samples=10, max_depth=100, max_component_value=10f0, regularize=true)
# sensor = Hikari.FilmSensor(; iso=100, white_balance=6500)
# img = Makie.colorbuffer(ax.scene; device=DEVICE, integrator=integrator, sensor=sensor, exposure=0.6f0, tonemap=:aces, gamma=2.2f0)

# render_video()
