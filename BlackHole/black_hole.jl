include("../common/common.jl")
using GeometryBasics
using Raycore
using LinearAlgebra: normalize, norm, cross, dot, I

# ============================================================================
# SpacetimeMedium - A medium that bends light like a black hole
# ============================================================================

"""
    SpacetimeMedium

A participating medium that simulates gravitational lensing around a black hole.
Light rays are deflected toward the black hole center based on the Schwarzschild metric.

The medium also contains a density field for dust/accretion disk visualization.
"""
struct SpacetimeMedium{T<:AbstractArray{Float32,3}} <: Hikari.Medium
    center::Point3f
    schwarzschild_radius::Float32
    lensing_strength::Float32
    bounds::Hikari.Bounds3
    render_to_medium::Mat4f
    medium_to_render::Mat4f
    σ_a::Hikari.RGBSpectrum
    σ_s::Hikari.RGBSpectrum
    density::T
    density_res::Vec3{Int}
    emission_scale::Float32
    g::Float32
    deflection_padding::Float32
    max_density::Float32
end

function SpacetimeMedium(
    density::AbstractArray{Float32,3};
    center::Point3f = Point3f(0f0, 0f0, 0f0),
    schwarzschild_radius::Float32 = 1f0,
    lensing_strength::Float32 = 1f0,
    σ_a::Hikari.RGBSpectrum = Hikari.RGBSpectrum(0.01f0),
    σ_s::Hikari.RGBSpectrum = Hikari.RGBSpectrum(0.1f0),
    emission_scale::Float32 = 1f0,
    g::Float32 = 0f0,
    bounds::Hikari.Bounds3 = Hikari.Bounds3(Point3f(-10f0, -10f0, -10f0), Point3f(10f0, 10f0, 10f0)),
    transform::Mat4f = Mat4f(I),
    deflection_padding::Float32 = 2f0
)
    inv_transform = inv(transform)
    max_density = Float32(maximum(density))
    nx, ny, nz = size(density)
    density_res = Vec3{Int}(nx, ny, nz)

    SpacetimeMedium(
        center, schwarzschild_radius, lensing_strength, bounds,
        inv_transform, transform, σ_a, σ_s, density, density_res,
        emission_scale, g, deflection_padding, max_density
    )
end

Hikari.is_emissive(m::SpacetimeMedium) = m.emission_scale > 0f0

@inline function sample_density(density_arr, density_res::Vec3{Int}, bounds::Hikari.Bounds3, p_medium::Point3f)::Float32
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

    d000 = density_arr[ix, iy, iz]
    d100 = density_arr[ix+1, iy, iz]
    d010 = density_arr[ix, iy+1, iz]
    d110 = density_arr[ix+1, iy+1, iz]
    d001 = density_arr[ix, iy, iz+1]
    d101 = density_arr[ix+1, iy, iz+1]
    d011 = density_arr[ix, iy+1, iz+1]
    d111 = density_arr[ix+1, iy+1, iz+1]

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

function Hikari.sample_point(
    medium::SpacetimeMedium,
    media,  # StaticMultiTypeSet for dereferencing TextureRefs
    table::Hikari.RGBToSpectrumTable,
    p::Point3f,
    λ::Hikari.Wavelengths
)::Hikari.MediumProperties
    p4 = medium.render_to_medium * Vec4f(p[1], p[2], p[3], 1f0)
    p_medium = Point3f(p4[1], p4[2], p4[3])

    r_vec = p_medium - medium.center
    r = norm(r_vec)

    if r < medium.schwarzschild_radius
        return Hikari.MediumProperties(
            Hikari.SpectralRadiance(1000f0),
            Hikari.SpectralRadiance(0f0),
            Hikari.SpectralRadiance(0f0),
            0f0
        )
    end

    density_arr = Raycore.deref(media, medium.density)
    d = sample_density(density_arr, medium.density_res, medium.bounds, p_medium)
    σ_a = Hikari.uplift_rgb_unbounded(table, medium.σ_a, λ) * d
    σ_s = Hikari.uplift_rgb_unbounded(table, medium.σ_s, λ) * d

    Le = if d > 0.005f0 && medium.emission_scale > 0f0
        temp_factor = clamp(2.5f0 * medium.schwarzschild_radius / r, 0f0, 1f0)
        r_col = 1.0f0
        g_col = 0.3f0 + 0.7f0 * temp_factor
        b_col = 0.1f0 + 0.6f0 * temp_factor^2
        emission_rgb = Hikari.RGBSpectrum(r_col, g_col, b_col) * medium.emission_scale * d * (0.5f0 + temp_factor)
        Hikari.uplift_rgb_unbounded(table, emission_rgb, λ)
    else
        Hikari.SpectralRadiance(0f0)
    end

    return Hikari.MediumProperties(σ_a, σ_s, Le, medium.g)
end

function Hikari.create_majorant_iterator(
    medium::SpacetimeMedium,
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
    # Homogeneous majorant: σ_t * max_density + deflection padding
    σ_a = Hikari.uplift_rgb_unbounded(table, medium.σ_a, λ)
    σ_s = Hikari.uplift_rgb_unbounded(table, medium.σ_s, λ)
    σ_maj = (σ_a + σ_s) * medium.max_density + Hikari.SpectralRadiance(medium.deflection_padding)
    return Hikari.RayMajorantIterator_homogeneous(t_enter, t_exit, σ_maj)
end

# 6-arg version with template_grid for GPU type consistency
using Base: @propagate_inbounds
@propagate_inbounds function Hikari.create_majorant_iterator(
    medium::SpacetimeMedium,
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

@inline Hikari.get_template_grid(::SpacetimeMedium) = Hikari.EmptyMajorantGrid()

function Hikari.apply_deflection(
    medium::SpacetimeMedium,
    p::Point3f,
    ray_d::Vec3f,
    dt::Float32
)::Vec3f
    r_vec = p - medium.center
    r = norm(r_vec)
    rs = medium.schwarzschild_radius

    if r < rs || r > 50f0 * rs
        return ray_d
    end

    to_bh = -normalize(r_vec)
    parallel_component = dot(to_bh, ray_d) * ray_d
    perp_to_ray = to_bh - parallel_component
    perp_norm = norm(perp_to_ray)

    if perp_norm < 1f-6
        return ray_d
    end

    deflection_dir = perp_to_ray / perp_norm
    deflection_amount = medium.lensing_strength * rs / (r * r) * dt
    new_d = ray_d + deflection_dir * deflection_amount
    return Vec3f(normalize(new_d))
end

# ============================================================================
# Density Field Generation
# ============================================================================

smoothstep(edge0, edge1, x) = let t = clamp((x - edge0) / (edge1 - edge0), 0f0, 1f0)
    t * t * (3f0 - 2f0 * t)
end

function create_black_hole_density(resolution::Int=64;
    disk_density::Float32=1.5f0,
    particle_density::Float32=1.2f0,
    inner_radius::Float32=1.5f0,
    outer_radius::Float32=14f0,
    num_particle_streams::Int=5,
    coord_scale::Float32=15f0
)
    density = zeros(Float32, resolution, resolution, resolution)
    center = resolution / 2

    for iz in 1:resolution, iy in 1:resolution, ix in 1:resolution
        x = (ix - center) / center * coord_scale
        y = (iy - center) / center * coord_scale
        z = (iz - center) / center * coord_scale

        r_cyl = sqrt(x^2 + y^2)
        θ = atan(y, x)

        # Accretion disk
        if r_cyl > inner_radius && r_cyl < outer_radius
            disk_height = 0.6f0 * (1f0 - 0.4f0 * (r_cyl - inner_radius) / (outer_radius - inner_radius))
            disk_profile = exp(-(z^2) / (2f0 * disk_height^2))
            radial_profile = exp(-(r_cyl - inner_radius) / 3f0)
            spiral_phase = θ + r_cyl * 0.5f0
            spiral_mod = 0.6f0 + 0.4f0 * sin(spiral_phase * 3f0)
            density[ix, iy, iz] += disk_density * disk_profile * radial_profile * spiral_mod
        end

        # Particle streams in the disk
        for stream in 1:num_particle_streams
            stream_θ = 2π * stream / num_particle_streams + 0.3f0
            spiral_k = 0.25f0
            r0 = outer_radius * 0.9f0

            if r_cyl > inner_radius * 1.2f0 && r_cyl < r0
                expected_θ = stream_θ - log(r_cyl / r0) / spiral_k
                angle_diff = mod(θ - expected_θ + π, 2π) - π
                stream_width = 0.2f0 + 0.15f0 * (r_cyl / r0)
                z_scale = 0.3f0 + 0.4f0 * (r_cyl / r0)
                stream_factor = exp(-(angle_diff^2) / (2f0 * stream_width^2))
                stream_factor *= exp(-(z^2) / (2f0 * z_scale^2))
                brightness = 1f0 + 3f0 * (1f0 - r_cyl / r0)^2
                density[ix, iy, iz] += particle_density * stream_factor * brightness
            end
        end

        # Photon ring
        photon_ring_r = 2.5f0
        if abs(r_cyl - photon_ring_r) < 0.5f0 && abs(z) < 0.3f0
            ring_factor = exp(-((r_cyl - photon_ring_r)^2) / 0.1f0) * exp(-(z^2) / 0.05f0)
            density[ix, iy, iz] += 2f0 * ring_factor
        end
    end

    return density
end

function create_black_hole_scene(;
    resolution=(1600, 900),
    density_resolution::Int=64,
    schwarzschild_radius=1.0f0,
    lensing_strength=3f0,
    σ_a=Hikari.RGBSpectrum(0.008f0, 0.008f0, 0.008f0),
    σ_s=Hikari.RGBSpectrum(0.002f0, 0.002f0, 0.002f0),
    emission_scale=0.4f0,
    g=0.3f0,
    disk_density=1.8f0,
    particle_density=1.5f0,
    inner_radius=1.5f0,
    outer_radius=14f0,
    num_particle_streams=5,
    coord_scale=15f0,
    deflection_padding=3f0,
    boundary_radius=30f0,
    eyeposition=Vec3f(28, 10, 7),
    lookat=Vec3f(0, 0, 0),
    fov=42f0,
)
    @info "Creating black hole density field..."
    density = create_black_hole_density(density_resolution;
        disk_density=Float32(disk_density),
        particle_density=Float32(particle_density),
        inner_radius=Float32(inner_radius),
        outer_radius=Float32(outer_radius),
        num_particle_streams=num_particle_streams,
        coord_scale=Float32(coord_scale)
    )

    spacetime = SpacetimeMedium(
        density;
        center=Point3f(0f0, 0f0, 0f0),
        schwarzschild_radius=Float32(schwarzschild_radius),
        lensing_strength=Float32(lensing_strength),
        σ_a=σ_a,
        σ_s=σ_s,
        emission_scale=Float32(emission_scale),
        g=Float32(g),
        bounds=Hikari.Bounds3(Point3f(-coord_scale, -coord_scale, -coord_scale),
                              Point3f(coord_scale, coord_scale, coord_scale)),
        deflection_padding=Float32(deflection_padding)
    )

    volume_material = Hikari.MediumInterface(
        Hikari.ThinDielectricMaterial(eta=1.0f0),
        inside=spacetime,
        outside=nothing
    )

    ax = Scene(
        backgroundcolor=RGBf(0.0, 0.0, 0.0),
        size=resolution,
        lights=[
            EnvironmentLight(1f0, load(joinpath(@__DIR__, "starmap_2020_4k.exr")))
        ]
    )
    cam3d!(ax)

    # Medium boundary sphere
    mesh!(ax, Sphere(Point3f(0, 0, 0), Float32(boundary_radius));
        color=:black, visible=false,
        material=volume_material,
        transparency=true
    )

    # Event horizon (black sphere)
    mesh!(ax, Sphere(Point3f(0, 0, 0), Float32(schwarzschild_radius));
        color=RGBf(0, 0, 0), visible=false,
        material=Hikari.MatteMaterial(Kd=Hikari.RGBSpectrum(0f0, 0f0, 0f0))
    )

    # Camera: edge-on view for classic black hole look
    cam = ax.camera_controls
    cam.eyeposition[] = eyeposition
    cam.lookat[] = lookat
    cam.fov[] = Float32(fov)

    return ax
end

# ============================================================================
# Rendering
# ============================================================================

function render_scene(;
    device=DEVICE,
    resolution=(1600, 900),
    samples=1000,
    max_depth=100,
    output_path=joinpath(@__DIR__, "black_hole.png"),
)
    ax = create_black_hole_scene(resolution=resolution)
    GC.gc(true)
    integrator = Hikari.VolPath(samples=samples, max_depth=max_depth)
    sensor = Hikari.FilmSensor(iso=30, white_balance=5500)
    @time result = Makie.colorbuffer(ax;
        backend=TraceMakie, device=device, integrator=integrator, sensor=sensor,
    )
    save(output_path, result)
    @info "Saved → $output_path"
    return result
end

function render_interactive(;
    device=DEVICE,
    resolution=(800, 450),
)
    ax = create_black_hole_scene(resolution=resolution)
    sensor = Hikari.FilmSensor(iso=30, white_balance=5500)
    TraceMakie.interactive_window(ax; device=device, sensor=sensor, integrator=Hikari.VolPath(samples=1, max_depth=100))
    display(ax; backend=GLMakie, update=false)
end

# render_scene()
