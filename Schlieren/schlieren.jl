# Schlieren Imaging Demo — Visualizing density gradients via absorption
#
# Schlieren/shadowgraph imaging makes density gradients visible by modulating
# light absorption proportional to the gradient magnitude.  Regions of strong
# refractive-index gradient (plume edges) absorb more light, creating dark
# bands against a bright striped background — the hallmark schlieren pattern.
#
# The medium also implements apply_deflection for ray-path curvature through
# the density field (eikonal deflection), which subtly shifts sampling positions
# during delta tracking.

include("../common/common.jl")
using GeometryBasics
using Raycore
using LinearAlgebra: normalize, norm, cross, dot, I
using FFMPEG_jll

# ============================================================================
# Gradient Computation
# ============================================================================

"""
    compute_density_gradient(density) -> (grad_x, grad_y, grad_z)

Central finite differences on a 3D grid, one-sided at boundaries.
"""
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
# SchlierenMedium — Gradient-absorption medium with eikonal deflection
# ============================================================================

# Module-level gradient store: apply_deflection has no `media` parameter,
# so it cannot call Raycore.deref on TextureRef fields.
const GRADIENT_STORE = Dict{UInt64, Array{Vec3f,3}}()

struct SchlierenMedium{T<:AbstractArray{Float32,3}} <: Hikari.Medium
    density::T              # density field (for deflection path sampling)
    grad_magnitude::T       # |∇ρ| field (for absorption-based visualization)
    density_res::Vec3{Int}
    max_density::Float32
    max_grad_magnitude::Float32
    gradient_id::UInt64     # key into GRADIENT_STORE for apply_deflection
    bounds::Hikari.Bounds3
    render_to_medium::Mat4f
    medium_to_render::Mat4f
    σ_a::Hikari.RGBSpectrum # absorption color
    g::Float32
    deflection_scale::Float32
    deflection_padding::Float32
    absorption_scale::Float32  # strength of gradient-based absorption
end

function SchlierenMedium(
    density::AbstractArray{Float32,3};
    σ_a::Hikari.RGBSpectrum = Hikari.RGBSpectrum(1f0),
    g::Float32 = 0f0,
    bounds::Hikari.Bounds3 = Hikari.Bounds3(Point3f(-5f0), Point3f(5f0)),
    transform::Mat4f = Mat4f(I),
    deflection_scale::Float32 = 50f0,
    deflection_padding::Float32 = 0.1f0,
    absorption_scale::Float32 = 15f0,
)
    inv_transform = inv(transform)
    max_density = Float32(maximum(density))
    nx, ny, nz = size(density)
    density_res = Vec3{Int}(nx, ny, nz)

    grad_x, grad_y, grad_z = compute_density_gradient(Array(density))

    # Build gradient vector array (for deflection) and magnitude array (for absorption)
    gradient = Array{Vec3f,3}(undef, nx, ny, nz)
    grad_mag = zeros(Float32, nx, ny, nz)
    max_grad_mag = 0f0
    for iz in 1:nz, iy in 1:ny, ix in 1:nx
        gv = Vec3f(grad_x[ix,iy,iz], grad_y[ix,iy,iz], grad_z[ix,iy,iz])
        gradient[ix,iy,iz] = gv
        mag = norm(gv)
        grad_mag[ix,iy,iz] = mag
        max_grad_mag = max(max_grad_mag, mag)
    end

    # Normalize gradient magnitude to [0,1]
    if max_grad_mag > 0f0
        grad_mag ./= max_grad_mag
    end

    # Store gradient vectors in module-level dict (scalars survive TextureRef)
    gid = objectid(gradient)
    GRADIENT_STORE[gid] = gradient

    SchlierenMedium(
        density, grad_mag, density_res,
        max_density, max_grad_mag,
        gid,
        bounds, inv_transform, transform,
        σ_a, g,
        deflection_scale, deflection_padding,
        absorption_scale,
    )
end

# ============================================================================
# Trilinear Interpolation
# ============================================================================

@inline function sample_field(arr, density_res::Vec3{Int}, bounds::Hikari.Bounds3, p_medium::Point3f)
    p_norm = (p_medium - bounds.p_min) ./ (bounds.p_max - bounds.p_min)
    if any(p_norm .< 0f0) || any(p_norm .> 1f0)
        return zero(eltype(arr))
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

    # Absorption proportional to gradient magnitude (schlieren/shadowgraph effect)
    grad_mag_arr = Raycore.deref(media, medium.grad_magnitude)
    gm = sample_field(grad_mag_arr, medium.density_res, medium.bounds, p_medium)
    σ_a = Hikari.uplift_rgb_unbounded(table, medium.σ_a, λ) * (gm * medium.absorption_scale)

    # No scattering — medium is transparent except for gradient-based absorption
    σ_s = Hikari.SpectralRadiance(0f0)

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
    # Majorant must bound the maximum possible σ_a (normalized grad_mag ≤ 1)
    σ_a_max = Hikari.uplift_rgb_unbounded(table, medium.σ_a, λ) * medium.absorption_scale
    σ_maj = σ_a_max + Hikari.SpectralRadiance(medium.deflection_padding)
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
    σ_a_max = Hikari.uplift_rgb_unbounded(table, medium.σ_a, λ) * medium.absorption_scale
    σ_maj = σ_a_max + Hikari.SpectralRadiance(medium.deflection_padding)
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

    # Look up gradient from module-level store (survives TextureRef conversion)
    grad = sample_field(GRADIENT_STORE[medium.gradient_id], medium.density_res, medium.bounds, p_medium)

    grad_mag = norm(grad)
    if grad_mag < 1f-8
        return ray_d
    end

    # Convert gradient from voxel-index-space to world-space
    extent = medium.bounds.p_max - medium.bounds.p_min
    world_grad = Vec3f(grad[1] / extent[1], grad[2] / extent[2], grad[3] / extent[3])

    # Eikonal deflection: bend ray toward density gradient
    deflection = world_grad * medium.deflection_scale * dt
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

        # Plume widens with altitude via entrainment — narrow source, broad top
        plume_width = 0.03f0 + 0.15f0 * z * z
        radial_profile = exp(-r*r / (2f0 * plume_width^2))

        # Sharper radial edge — schlieren shows edges clearly
        edge_ring = exp(-((r - plume_width * 1.2f0)^2) / (0.003f0))

        # Vertical envelope: gradual onset, sustained through middle, fade at top
        z_envelope = smoothstep(0.01f0, 0.1f0, z) * exp(-1.5f0 * max(z - 0.65f0, 0f0))

        # Multi-scale turbulence for realistic fine structure
        θ = atan(dy, dx)
        # Large-scale billowing (mushroom puffs)
        turb1 = 0.4f0 * sin(4f0 * θ + 6f0 * z + time_phase)
        # Medium-scale whorls
        turb2 = 0.3f0 * sin(8f0 * θ - 10f0 * z + 2.3f0 * time_phase)
        # Fine-scale wrinkles
        turb3 = 0.2f0 * cos(14f0 * θ + 20f0 * z - 1.7f0 * time_phase)
        turb4 = 0.15f0 * sin(22f0 * θ + 30f0 * z + 0.8f0 * time_phase)
        # Extra fine detail
        turb5 = 0.08f0 * cos(30f0 * θ - 15f0 * z + 3.1f0 * time_phase)
        # Axial modulation — creates distinct puffs along the plume
        axial_mod = 1f0 + 0.4f0 * sin(10f0 * z + time_phase * 0.7f0)
        turbulence = (1f0 + turb1 + turb2 + turb3 + turb4 + turb5) * axial_mod

        # Combine: core profile + edge ring (schlieren highlights both)
        d = (0.6f0 * radial_profile + 0.4f0 * edge_ring) * z_envelope * turbulence
        d = max(d, 0f0)

        density[ix + pad, iy + pad, iz + pad] = d
    end

    dmax = maximum(density)
    if dmax > 0f0
        density ./= dmax
    end

    return density
end

# Smooth step helper (GLSL-style)
function smoothstep(edge0::Float32, edge1::Float32, x::Float32)
    t = clamp((x - edge0) / (edge1 - edge0), 0f0, 1f0)
    return t * t * (3f0 - 2f0 * t)
end

# ============================================================================
# Stripe Environment Map
# ============================================================================

"""
    create_schlieren_envmap(; map_size=512, style=:color_gradient) -> Matrix{RGBf}

Create a square environment map in equal-area octahedral format for schlieren background.
Colors are assigned based on 3D direction to ensure correct appearance in the camera view.
- `:color_gradient` — vivid warm→cool gradient keyed to world-space X (left→right)
- `:bright` — uniform bright white (classic shadowgraph)
"""
function create_schlieren_envmap(; map_size::Int=512, style::Symbol=:color_gradient)
    img = Matrix{RGBf}(undef, map_size, map_size)
    for j in 1:map_size, i in 1:map_size
        # Map pixel to [0,1]² UV, then to 3D direction via octahedral inverse
        uv = Point2f((j - 0.5f0) / map_size, (i - 0.5f0) / map_size)
        dir = Hikari.equal_area_square_to_sphere(uv)

        if style == :color_gradient
            # Azimuthal angle relative to +Y (camera forward direction)
            angle = atan(dir[1], dir[2])  # -π to π
            # Map visible FOV to [0, 1] — wider range for smoother transition
            t = clamp((angle + 0.5f0) / 1f0, 0f0, 1f0)
            # Classic schlieren 5-color ramp: blue → cyan → green → yellow → red
            r = clamp(min(4f0 * (t - 0.5f0), 4f0 * (1.25f0 - t)), 0f0, 1f0)
            g = clamp(min(4f0 * (t - 0.15f0), 4f0 * (0.85f0 - t)), 0f0, 1f0)
            b = clamp(min(4f0 * (0.55f0 - t), 1f0), 0f0, 1f0)
            # Slight brightness boost to keep background bright
            img[i, j] = RGBf(
                clamp(r * 0.8f0 + 0.2f0, 0f0, 1f0),
                clamp(g * 0.8f0 + 0.2f0, 0f0, 1f0),
                clamp(b * 0.8f0 + 0.2f0, 0f0, 1f0),
            )
        else
            img[i, j] = RGBf(0.97f0, 0.95f0, 0.91f0)  # warm paper tint
        end
    end
    return img
end

# ============================================================================
# Scene Construction
# ============================================================================

function create_schlieren_scene(;
    resolution=(1280, 720),
    density_resolution::Int=128,
    deflection_scale::Float32=50f0,
    deflection_padding::Float32=0.1f0,
    absorption_scale::Float32=6f0,
    time_phase::Float32=0f0,
    envmap_style::Symbol=:bright,
    volume_size::Float32=10f0,
)
    @info "Creating thermal plume density field..."
    density = create_thermal_plume(; resolution=density_resolution, time_phase=time_phase)

    half = volume_size / 2f0
    bounds = Hikari.Bounds3(Point3f(-half, -half, -half), Point3f(half, half, half))

    @info "Building SchlierenMedium (computing gradients)..."
    # Neutral absorption → all wavelengths see same σ → minimal spectral noise
    # Post-process toning could add color; neutral gives cleanest shadowgraph
    medium = SchlierenMedium(
        density;
        bounds=bounds,
        deflection_scale=deflection_scale,
        deflection_padding=deflection_padding,
        absorption_scale=absorption_scale,
    )

    volume_material = Hikari.MediumInterface(
        Hikari.ThinDielectricMaterial(eta=1.0f0);
        inside=medium,
        outside=nothing,
    )

    # Schlieren background environment map
    envmap_image = create_schlieren_envmap(; map_size=512, style=envmap_style)

    fig = Figure(; size=resolution)
    ax = LScene(fig[1, 1]; show_axis=false, scenekw=(;
        lights=[
            Makie.EnvironmentLight(5f0, envmap_image),
        ]
    ))

    # Medium boundary sphere — large enough to keep edges out of frame
    boundary_radius = half * sqrt(3f0)
    mesh!(ax, Sphere(Point3f(0, 0, 0), boundary_radius);
        color=:black, visible=false,
        material=volume_material,
        transparency=true,
    )

    # Camera: side-on view, tight crop on the plume to hide sphere boundary
    cam = ax.scene.camera_controls
    cam.eyeposition[] = Vec3f(0, volume_size * 2.0f0, volume_size * 0.05f0)
    cam.lookat[] = Vec3f(0, 0, volume_size * 0.05f0)
    cam.upvector[] = Vec3f(0, 0, 1)
    cam.fov[] = 30f0
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
