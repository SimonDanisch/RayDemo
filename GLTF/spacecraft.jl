# SpaceCraft HL-20 Demo - Ray Traced Visualization
include(joinpath(@__DIR__, "..", "common", "common.jl"))
using MeshIO

# Create rocket exhaust volume with emission that fades along the plume
function create_exhaust_medium(;
    res::Int=32,
    bounds::Hikari.Bounds3,  # Required: world-space bounds for the volume
    core_color=Hikari.RGBSpectrum(0.8f0, 0.9f0, 1.0f0),   # Blue-white hot core
    edge_color=Hikari.RGBSpectrum(1.0f0, 0.4f0, 0.1f0),   # Orange edge
    emission_scale::Float32=500f0,
    density_scale::Float32=10f0
)
    Le_grid = Array{Hikari.RGBSpectrum, 3}(undef, res, res, res)
    σ_a_grid = Array{Hikari.RGBSpectrum, 3}(undef, res, res, res)
    σ_s_grid = Array{Hikari.RGBSpectrum, 3}(undef, res, res, res)

    for iz in 1:res, iy in 1:res, ix in 1:res
        x = (ix - 0.5f0) / res
        y = (iy - 0.5f0) / res
        z = (iz - 0.5f0) / res

        dx, dy = x - 0.5f0, y - 0.5f0
        r = sqrt(dx*dx + dy*dy)
        cone_radius = 0.1f0 + 0.35f0 * z
        dist_from_edge = r - cone_radius

        if dist_from_edge < 0
            z_falloff = exp(-6f0 * z)
            r_norm = r / max(cone_radius, 0.01f0)
            r_falloff = exp(-2f0 * r_norm * r_norm)
            edge_fade = clamp(-dist_from_edge / 0.05f0, 0f0, 1f0)
            intensity = z_falloff * r_falloff * edge_fade
            color = core_color * (1f0 - r_norm) + edge_color * r_norm

            Le_grid[ix, iy, iz] = color * intensity
            σ_a_grid[ix, iy, iz] = Hikari.RGBSpectrum(2.0f0) * intensity
            σ_s_grid[ix, iy, iz] = Hikari.RGBSpectrum(0.5f0) * intensity
        else
            Le_grid[ix, iy, iz] = Hikari.RGBSpectrum(0f0)
            σ_a_grid[ix, iy, iz] = Hikari.RGBSpectrum(0f0)
            σ_s_grid[ix, iy, iz] = Hikari.RGBSpectrum(0f0)
        end
    end

    return Hikari.RGBGridMedium(
        σ_a_grid=σ_a_grid, σ_s_grid=σ_s_grid, Le_grid=Le_grid,
        sigma_scale=density_scale, Le_scale=emission_scale,
        g=0.0f0, bounds=bounds
    )
end


function create_scene(; resolution=(1500, 1500))
    fig = Figure(size=resolution)
    radiance = 300
    envlight = FileIO.load(joinpath(@__DIR__, "..", "assets", "sky.exr"))
    ax = LScene(fig[1, 1]; show_axis=false, scenekw=(;
        backgroundcolor=RGBf(0.02, 0.02, 0.03),
        lights=[
            PointLight(RGBf(radiance, radiance, radiance), Vec3f(-6.2, 6.88, 8.97)),
            EnvironmentLight(1.0, envlight,
                rotation_angle=90f0, rotation_axis=Vec3f(1, 0, 0)),
        ]
    ))

    # Load GLTF model via MeshIO — returns a MetaMesh with embedded materials
    # GLTF has 0.01 scale baked in, so scale up to make visible
    # From JuliaHub
    spacecraft_mesh = FileIO.load(joinpath(@__DIR__, "assets", "hl20.gltf"); up=Vec3f(0, 1, 0))
    spacecraft_material = Hikari.CoatedDiffuseMaterial(
        reflectance=(0.95f0, 0.95f0, 0.95f0),
        roughness=0.001f0, eta=1.5f0, thickness=0.03f0
    )
    p = mesh!(ax, spacecraft_mesh; material=spacecraft_material)
    Makie.scale!(p, 100, 100, 100)

    # Set up camera
    center!(ax.scene)
    cam = ax.scene.camera_controls

    # Rocket exhaust plume
    exhaust_size = Vec3f(2.0, 2.0, 5.0)
    exhaust_pos = Point3f(-1.0, 0.0, 3.4)
    exhaust_bounds = Hikari.Bounds3(exhaust_pos, exhaust_pos + exhaust_size)
    exhaust_medium = create_exhaust_medium(res=32, emission_scale=5000f0, bounds=exhaust_bounds)
    exhaust_material = Hikari.MediumInterface(
        Hikari.GlassMaterial(Kr=Hikari.RGBSpectrum(0f0), Kt=Hikari.RGBSpectrum(1f0), index=1.0f0);
        inside=exhaust_medium, outside=nothing
    )
    mesh!(ax, Rect3f(exhaust_pos, exhaust_size); material=exhaust_material)

    # Ground plane
    lookat = cam.lookat[]
    min_y = lookat[2] - 2
    ground_material = Hikari.MatteMaterial(Kd=Hikari.RGBSpectrum(0.15f0, 0.15f0, 0.18f0))
    mesh!(ax, Rect3f(Point3f(lookat[1] - 50, min_y, lookat[3] - 50), Vec3f(100, 0.1, 100));
        material=ground_material, color=Makie.wong_colors()[2])

    # Camera
    cam = cameracontrols(ax.scene)
    cam.eyeposition[] = [6.2, 7.88, -8.97]
    cam.lookat[] = [0.72, 0.29, -1.34]
    cam.upvector[] = [-0.28, 0.77, 0.57]
    cam.fov[] = 45.0
    Makie.update_cam!(ax.scene, cam)

    return fig, ax
end

function render_scene(;
    device=DEVICE,
    resolution=(1500, 1500),
    samples=1000,
    max_depth=50,
    output_path=joinpath(@__DIR__, "spacecraft.png"),
)
    fig, ax = create_scene(; resolution=resolution)
    integrator = Hikari.VolPath(samples=samples, max_depth=max_depth)
    sensor = Hikari.FilmSensor(iso=10, white_balance=6500)
    result = Makie.colorbuffer(ax.scene;
        device=device, integrator=integrator, sensor=sensor, update=false
    )
    FileIO.save(output_path, result)
    @info "Saved → $output_path"
    return result
end

# render_scene()
