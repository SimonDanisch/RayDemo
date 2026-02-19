# Schlieren Imaging Demo — Visualizing density gradients via ray deflection
#
# Schlieren imaging renders a nearly transparent medium where the visual
# information comes from ray deflection distorting a background stripe pattern.

include("../common/common.jl")
using GeometryBasics
using Raycore
using LinearAlgebra: normalize, norm, cross, dot, I
using FFMPEG_jll

# ============================================================================
# Gradient Computation
# ============================================================================

function compute_density_gradient(density::Array{Float32,3})
    nx, ny, nz = size(density)
    grad_x = zeros(Float32, nx, ny, nz)
    grad_y = zeros(Float32, nx, ny, nz)
    grad_z = zeros(Float32, nx, ny, nz)

    for iz in 1:nz, iy in 1:ny, ix in 1:nx
        grad_x[ix, iy, iz] = if ix == 1
            density[2, iy, iz] - density[1, iy, iz]
        elseif ix == nx
            density[nx, iy, iz] - density[nx-1, iy, iz]
        else
            (density[ix+1, iy, iz] - density[ix-1, iy, iz]) * 0.5f0
        end

        grad_y[ix, iy, iz] = if iy == 1
            density[ix, 2, iz] - density[ix, 1, iz]
        elseif iy == ny
            density[ix, ny, iz] - density[ix, ny-1, iz]
        else
            (density[ix, iy+1, iz] - density[ix, iy-1, iz]) * 0.5f0
        end

        grad_z[ix, iy, iz] = if iz == 1
            density[ix, iy, 2] - density[ix, iy, 1]
        elseif iz == nz
            density[ix, iy, nz] - density[ix, iy, nz-1]
        else
            (density[ix, iy, iz+1] - density[ix, iy, iz-1]) * 0.5f0
        end
    end

    return grad_x, grad_y, grad_z
end

# ============================================================================
# SchlierenMedium — A medium that bends light through density gradients
# ============================================================================

struct SchlierenMedium{T<:AbstractArray{Float32,3}} <: Hikari.Medium
    density::T
    density_res::Vec3{Int}
    max_density::Float32
    grad_x::T
    grad_y::T
    grad_z::T
    max_grad_magnitude::Float32
    bounds::Hikari.Bounds3
    render_to_medium::Mat4f
    medium_to_render::Mat4f
    σ_a::Hikari.RGBSpectrum
    σ_s::Hikari.RGBSpectrum
    g::Float32
    deflection_scale::Float32
    deflection_padding::Float32
end

function SchlierenMedium(
    density::AbstractArray{Float32,3};
    σ_a::Hikari.RGBSpectrum = Hikari.RGBSpectrum(0.0001f0),
    σ_s::Hikari.RGBSpectrum = Hikari.RGBSpectrum(0.0001f0),
    g::Float32 = 0f0,
    bounds::Hikari.Bounds3 = Hikari.Bounds3(Point3f(-5f0, -5f0, -5f0), Point3f(5f0, 5f0, 5f0)),
    transform::Mat4f = Mat4f(I),
    deflection_scale::Float32 = 50f0,
    deflection_padding::Float32 = 5f0
)
    inv_transform = inv(transform)
    max_density = Float32(maximum(density))
    nx, ny, nz = size(density)
    density_res = Vec3{Int}(nx, ny, nz)

    grad_x, grad_y, grad_z = compute_density_gradient(Array(density))

    max_grad_mag = 0f0
    for iz in 1:nz, iy in 1:ny, ix in 1:nx
        mag = sqrt(grad_x[ix,iy,iz]^2 + grad_y[ix,iy,iz]^2 + grad_z[ix,iy,iz]^2)
        max_grad_mag = max(max_grad_mag, mag)
    end

    # Convert gradient arrays to same type as density (for GPU compatibility)
    T = typeof(density)
    gx = convert(T, grad_x)::T
    gy = convert(T, grad_y)::T
    gz = convert(T, grad_z)::T
    d = convert(T, Array(density))::T

    SchlierenMedium(
        d, density_res, max_density,
        gx, gy, gz, max_grad_mag,
        bounds, inv_transform, transform,
        σ_a, σ_s, g,
        deflection_scale, deflection_padding
    )
end

# ============================================================================
# Trilinear Interpolation
# ============================================================================

@inline function sample_field(arr, density_res::Vec3{Int}, bounds::Hikari.Bounds3, p_medium::Point3f)::Float32
    p_norm = (p_medium - bounds.p_min) ./ (bounds.p_max - bounds.p_min)
    if any(p_norm .< 0f0) || any(p_norm .> 1f0)
        return 0f0
    end

    nx, ny, nz = density_res[1], density_res[2], density_res[3]
    gx = p_norm[1] * (nx - 1) + 1
    gy = p_norm[2] * (ny - 1) + 1
    gz = p_norm[3] * (nz - 1) + 1

    ix = clamp(floor(Int, gx), 1, nx - 1)
    iy = clamp(floor(Int, gy), 1, ny - 1)
    iz = clamp(floor(Int, gz), 1, nz - 1)

    fx, fy, fz = Float32(gx - ix), Float32(gy - iy), Float32(gz - iz)

    d000 = arr[ix, iy, iz]
    d100 = arr[ix+1, iy, iz]
    d010 = arr[ix, iy+1, iz]
    d110 = arr[ix+1, iy+1, iz]
    d001 = arr[ix, iy, iz+1]
    d101 = arr[ix+1, iy, iz+1]
    d011 = arr[ix, iy+1, iz+1]
    d111 = arr[ix+1, iy+1, iz+1]

    fx1 = 1f0 - fx
    d00 = d000 * fx1 + d100 * fx
    d10 = d010 * fx1 + d110 * fx
    d01 = d001 * fx1 + d101 * fx
    d11 = d011 * fx1 + d111 * fx

    fy1 = 1f0 - fy
    d0 = d00 * fy1 + d10 * fy
    d1 = d01 * fy1 + d11 * fy

    return d0 * (1f0 - fz) + d1 * fz
end

# ============================================================================
# Hikari Interface Methods
# ============================================================================

Hikari.is_emissive(::SchlierenMedium) = false

@inline Hikari.get_template_grid(::SchlierenMedium) = Hikari.EmptyMajorantGrid()

function Hikari.sample_point(
    medium::SchlierenMedium,
    media,
    table::Hikari.RGBToSpectrumTable,
    p::Point3f,
    λ::Hikari.Wavelengths
)::Hikari.MediumProperties
    p4 = medium.render_to_medium * Vec4f(p[1], p[2], p[3], 1f0)
    p_medium = Point3f(p4[1], p4[2], p4[3])

    density_arr = Raycore.deref(media, medium.density)
    d = sample_field(density_arr, medium.density_res, medium.bounds, p_medium)
    σ_a = Hikari.uplift_rgb_unbounded(table, medium.σ_a, λ) * d
    σ_s = Hikari.uplift_rgb_unbounded(table, medium.σ_s, λ) * d

    return Hikari.MediumProperties(σ_a, σ_s, Hikari.SpectralRadiance(0f0), medium.g)
end

function Hikari.create_majorant_iterator(
    medium::SchlierenMedium,
    table::Hikari.RGBToSpectrumTable,
    ray::Raycore.Ray,
    t_max::Float32,
    λ::Hikari.Wavelengths
)
    t_enter, t_exit = Hikari.ray_bounds_intersect(ray.o, ray.d, medium.bounds)
    t_enter = max(t_enter, 0f0)
    t_exit = min(t_exit, t_max)
    if t_enter >= t_exit
        return Hikari.RayMajorantIterator_homogeneous(0f0, 0f0, Hikari.SpectralRadiance(0f0))
    end
    σ_a = Hikari.uplift_rgb_unbounded(table, medium.σ_a, λ)
    σ_s = Hikari.uplift_rgb_unbounded(table, medium.σ_s, λ)
    σ_maj = (σ_a + σ_s) * medium.max_density + Hikari.SpectralRadiance(medium.deflection_padding)
    return Hikari.RayMajorantIterator_homogeneous(t_enter, t_exit, σ_maj)
end

using Base: @propagate_inbounds
@propagate_inbounds function Hikari.create_majorant_iterator(
    medium::SchlierenMedium,
    table::Hikari.RGBToSpectrumTable,
    ray::Raycore.Ray,
    t_max::Float32,
    λ::Hikari.Wavelengths,
    template_grid::M
) where {M<:Hikari.MajorantGrid}
    t_enter, t_exit = Hikari.ray_bounds_intersect(ray.o, ray.d, medium.bounds)
    t_enter = max(t_enter, 0f0)
    t_exit = min(t_exit, t_max)
    res = template_grid.res
    if t_enter >= t_exit
        return Hikari.RayMajorantIterator{M}(
            Int32(0), Hikari.SpectralRadiance(0f0),
            0f0, 0f0, false, template_grid,
            (Int32(res[1]), Int32(res[2]), Int32(res[3])),
            (0f0, 0f0, 0f0), (0f0, 0f0, 0f0),
            (Int32(0), Int32(0), Int32(0)),
            (Int32(0), Int32(0), Int32(0)),
            (Int32(0), Int32(0), Int32(0))
        )
    end
    σ_a = Hikari.uplift_rgb_unbounded(table, medium.σ_a, λ)
    σ_s = Hikari.uplift_rgb_unbounded(table, medium.σ_s, λ)
    σ_maj = (σ_a + σ_s) * medium.max_density + Hikari.SpectralRadiance(medium.deflection_padding)
    return Hikari.RayMajorantIterator{M}(
        Int32(1), σ_maj,
        t_enter, t_exit, false, template_grid,
        (Int32(res[1]), Int32(res[2]), Int32(res[3])),
        (0f0, 0f0, 0f0), (0f0, 0f0, 0f0),
        (Int32(0), Int32(0), Int32(0)),
        (Int32(0), Int32(0), Int32(0)),
        (Int32(0), Int32(0), Int32(0))
    )
end

function Hikari.apply_deflection(
    medium::SchlierenMedium,
    p::Point3f,
    ray_d::Vec3f,
    dt::Float32
)::Vec3f
    p4 = medium.render_to_medium * Vec4f(p[1], p[2], p[3], 1f0)
    p_medium = Point3f(p4[1], p4[2], p4[3])

    # Sample pre-computed gradients via trilinear interpolation
    # NOTE: Accesses gradient arrays directly (no Raycore.deref) since
    # apply_deflection has no `media` parameter. Works if T supports
    # direct indexing (CPU Array or GPU device array).
    gx = sample_field(medium.grad_x, medium.density_res, medium.bounds, p_medium)
    gy = sample_field(medium.grad_y, medium.density_res, medium.bounds, p_medium)
    gz = sample_field(medium.grad_z, medium.density_res, medium.bounds, p_medium)

    grad_mag = sqrt(gx*gx + gy*gy + gz*gz)
    if grad_mag < 1f-8
        return ray_d
    end

    # Convert gradient from voxel-index-space to world-space
    extent = medium.bounds.p_max - medium.bounds.p_min
    world_gx = gx / extent[1]
    world_gy = gy / extent[2]
    world_gz = gz / extent[3]

    # Eikonal deflection: bend ray toward density gradient
    deflection = Vec3f(world_gx, world_gy, world_gz) * medium.deflection_scale * dt
    new_d = ray_d + deflection
    n = norm(new_d)
    if n < 1f-8
        return ray_d
    end
    return Vec3f(new_d / n)
end

# ============================================================================
# Procedural Density Field — Thermal Plume
# ============================================================================

function create_thermal_plume(; resolution::Int=128, time_phase::Float32=0f0)
    pad = 8
    inner_res = resolution
    total_res = inner_res + 2 * pad
    density = zeros(Float32, total_res, total_res, total_res)

    for iz in 1:inner_res, iy in 1:inner_res, ix in 1:inner_res
        x = (ix - 0.5f0) / inner_res
        y = (iy - 0.5f0) / inner_res
        z = (iz - 0.5f0) / inner_res

        dx = x - 0.5f0
        dy = y - 0.5f0
        r = sqrt(dx*dx + dy*dy)

        # Plume widens with altitude (entrainment)
        plume_width = 0.06f0 + 0.12f0 * z
        radial_profile = exp(-r*r / (2f0 * plume_width^2))

        # Vertical envelope: starts at bottom, fades at top
        z_envelope = z * exp(-2.5f0 * z)

        # Sinusoidal pseudo-turbulence perturbations
        θ = atan(dy, dx)
        turb1 = 0.3f0 * sin(6f0 * θ + 8f0 * z + time_phase)
        turb2 = 0.15f0 * sin(12f0 * θ - 5f0 * z + 2.3f0 * time_phase)
        turb3 = 0.1f0 * cos(4f0 * θ + 15f0 * z - 1.7f0 * time_phase)
        turbulence = 1f0 + turb1 + turb2 + turb3

        d = radial_profile * z_envelope * turbulence
        d = max(d, 0f0)

        density[ix + pad, iy + pad, iz + pad] = d
    end

    dmax = maximum(density)
    if dmax > 0f0
        density ./= dmax
    end

    return density
end

# ============================================================================
# Scene Construction
# ============================================================================

function create_schlieren_scene(;
    resolution=(1280, 720),
    density_resolution::Int=128,
    deflection_scale::Float32=50f0,
    deflection_padding::Float32=5f0,
    σ_a=Hikari.RGBSpectrum(0.0001f0),
    σ_s=Hikari.RGBSpectrum(0.0001f0),
    time_phase::Float32=0f0,
    stripe_count::Int=40,
    volume_size::Float32=10f0,
)
    @info "Creating thermal plume density field..."
    density = create_thermal_plume(; resolution=density_resolution, time_phase=time_phase)

    half = volume_size / 2f0
    bounds = Hikari.Bounds3(Point3f(-half, -half, -half), Point3f(half, half, half))

    @info "Building SchlierenMedium (computing gradients)..."
    medium = SchlierenMedium(
        density;
        σ_a=σ_a,
        σ_s=σ_s,
        bounds=bounds,
        deflection_scale=deflection_scale,
        deflection_padding=deflection_padding,
    )

    volume_material = Hikari.MediumInterface(
        Hikari.ThinDielectricMaterial(eta=1.0f0);
        inside=medium,
        outside=nothing,
    )

    # Scene
    fig = Figure(; size=resolution)
    ax = LScene(fig[1, 1]; show_axis=false, scenekw=(;
        lights=[
            PointLight(RGBf(80000, 80000, 80000), Vec3f(0, 30, 10)),
            AmbientLight(RGBf(0.3, 0.3, 0.3)),
        ]
    ))

    # --- Background screen with vertical stripes ---
    # Separate meshes for white/black stripes (solid color per mesh avoids
    # vertex-color textures which JLBackend cannot convert)
    screen_dist = volume_size * 2f0
    screen_half = volume_size * 1.5f0
    stripe_width = 2f0 * screen_half / stripe_count

    white_material = Hikari.MatteMaterial(Kd=Hikari.RGBSpectrum(0.9f0, 0.9f0, 0.9f0))
    black_material = Hikari.MatteMaterial(Kd=Hikari.RGBSpectrum(0.05f0, 0.05f0, 0.05f0))

    for (stripe_color, mat, col) in [(:white, white_material, RGBf(0.95, 0.95, 0.95)),
                                      (:black, black_material, RGBf(0.05, 0.05, 0.05))]
        verts = Point3f[]
        tris = GLTriangleFace[]
        for i in 0:stripe_count-1
            want_white = iseven(i)
            (stripe_color == :white) != want_white && continue
            x_lo = -screen_half + i * stripe_width
            x_hi = x_lo + stripe_width
            base = length(verts)
            push!(verts,
                Point3f(x_lo, -screen_dist, -screen_half),
                Point3f(x_hi, -screen_dist, -screen_half),
                Point3f(x_hi, -screen_dist,  screen_half),
                Point3f(x_lo, -screen_dist,  screen_half),
            )
            push!(tris,
                GLTriangleFace(base+1, base+2, base+3),
                GLTriangleFace(base+1, base+3, base+4),
            )
        end
        m = GeometryBasics.normal_mesh(verts, tris)
        mesh!(ax, m; color=col, material=mat)
    end

    # Medium boundary sphere (same pattern as BlackHole)
    boundary_radius = half * sqrt(3f0)
    mesh!(ax, Sphere(Point3f(0, 0, 0), boundary_radius);
        color=:black, visible=false,
        material=volume_material,
        transparency=true,
    )

    # Camera: side-on view looking through the medium at the background
    cam = ax.scene.camera_controls
    cam.eyeposition[] = Vec3f(0, volume_size * 2.5f0, volume_size * 0.1f0)
    cam.lookat[] = Vec3f(0, 0, 0)
    cam.upvector[] = Vec3f(0, 0, 1)
    cam.fov[] = 35f0
    update_cam!(ax.scene, cam)

    return fig, ax
end

# ============================================================================
# Rendering
# ============================================================================

function render_scene(;
    device=DEVICE,
    resolution=(1280, 720),
    samples=100,
    max_depth=50,
    output_path=joinpath(@__DIR__, "schlieren.png"),
)
    fig, ax = create_schlieren_scene(; resolution=resolution)
    GC.gc(true)
    integrator = Hikari.VolPath(samples=samples, max_depth=max_depth)
    sensor = Hikari.FilmSensor(iso=100, white_balance=6500)
    @time result = Makie.colorbuffer(ax.scene;
        backend=RayMakie, device=device, integrator=integrator, sensor=sensor,
    )
    save(output_path, result)
    @info "Saved → $output_path"
    return result
end

function render_video(;
    device=DEVICE,
    resolution=(1280, 720),
    samples=100,
    max_depth=50,
    nframes=100,
    output_path=joinpath(@__DIR__, "schlieren.mp4"),
)
    times = range(0f0, 4f0 * Float32(π); length=nframes)
    integrator = Hikari.VolPath(samples=samples, max_depth=max_depth)
    sensor = Hikari.FilmSensor(iso=100, white_balance=6500)

    # Render frames with time-shifting turbulence phase
    frame_dir = mktempdir()
    for (i, t) in enumerate(times)
        fig, ax = create_schlieren_scene(; resolution=resolution, time_phase=Float32(t))
        GC.gc(true)
        result = Makie.colorbuffer(ax.scene;
            backend=RayMakie, device=device, integrator=integrator, sensor=sensor,
        )
        frame_path = joinpath(frame_dir, "frame_$(lpad(i, 4, '0')).png")
        save(frame_path, result)
        @info "Frame $i/$(length(times))"
    end

    # Encode with FFMPEG_jll
    frame_pattern = joinpath(frame_dir, "frame_%04d.png")
    FFMPEG_jll.ffmpeg() do exe
        run(`$exe -y -framerate 30 -i $frame_pattern -c:v libx264 -pix_fmt yuv420p $output_path`)
    end

    @info "Saved → $output_path"
    return output_path
end
