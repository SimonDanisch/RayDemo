include(joinpath(@__DIR__, "..", "common", "common.jl"))
using Ark
using GeometryBasics
using LinearAlgebra

struct Pos
    p::Point3f
end
struct Vel
    v::Vec3f
end
struct UpdateStep
    step::Int
end

struct Neighbors
    n::Vector{Entity}
end
Neighbors() = Neighbors(Entity[])

# ============================================================================
# Resources
# ============================================================================

struct WorldBounds
    lo::Point3f
    hi::Point3f
end

mutable struct Tick
    tick::Int
end

# 3-D spatial hash for fast neighbour look-ups
struct Grid3D
    cells::Array{Vector{Entity},3}
    dims::NTuple{3,Int}
    cell_size::Float32
    origin::Point3f
end

function Grid3D(bounds::WorldBounds, cell::Float32)
    ext = bounds.hi - bounds.lo
    nx = max(1, ceil(Int, ext[1] / cell))
    ny = max(1, ceil(Int, ext[2] / cell))
    nz = max(1, ceil(Int, ext[3] / cell))
    Grid3D([Entity[] for _ in 1:nx, _ in 1:ny, _ in 1:nz],
        (nx, ny, nz), cell, bounds.lo)
end

function grid_cell(g::Grid3D, p::Point3f)
    ix = clamp(floor(Int, (p[1] - g.origin[1]) / g.cell_size) + 1, 1, g.dims[1])
    iy = clamp(floor(Int, (p[2] - g.origin[2]) / g.cell_size) + 1, 1, g.dims[2])
    iz = clamp(floor(Int, (p[3] - g.origin[3]) / g.cell_size) + 1, 1, g.dims[3])
    (ix, iy, iz)
end

# ============================================================================
# Minimal System / Scheduler
# ============================================================================

abstract type System end
initialize!(::System, ::World) = nothing
sys_update!(::System, ::World) = nothing

mutable struct Scheduler{S<:Tuple}
    world::World
    systems::S
end

function initialize!(s::Scheduler)
    add_resource!(s.world, Tick(0))
    for sys in s.systems
        initialize!(sys, s.world)
    end
end

function step!(s::Scheduler)
    for sys in s.systems
        sys_update!(sys, s.world)
    end
    get_resource(s.world, Tick).tick += 1
end

# ============================================================================
# System: initialisation
# ============================================================================

struct BoidsInit <: System
    count::Int
end

function initialize!(s::BoidsInit, world::World)
    bounds = get_resource(world, WorldBounds)
    centre = (bounds.lo + bounds.hi) / 2
    r = 0.3f0 * minimum(bounds.hi - bounds.lo)

    new_entities!(world, s.count,
        (Pos, Vel, Neighbors, UpdateStep),
    ) do (_, pos, vel, nbr, upd)
        for i in eachindex(pos)
            θ = rand(Float32) * 2f0π
            φ = acos(1f0 - 2f0rand(Float32))
            ρ = r * cbrt(rand(Float32))
            pos[i] = Pos(centre + Point3f(ρ * sin(φ) * cos(θ),
                ρ * sin(φ) * sin(θ),
                ρ * cos(φ)))
            θv = rand(Float32) * 2f0π
            φv = acos(1f0 - 2f0rand(Float32))
            sp = 0.3f0 + rand(Float32) * 0.3f0
            vel[i] = Vel(Vec3f(sp * sin(φv) * cos(θv),
                sp * sin(φv) * sin(θv),
                sp * cos(φv)))
            nbr[i] = Neighbors()
            upd[i] = UpdateStep(rand(0:29))
        end
    end
end

# ============================================================================
# System: neighbour search (staggered over 30 ticks)
# ============================================================================

struct BoidsNeighbors <: System
    max_dist::Float32
end

function initialize!(s::BoidsNeighbors, world::World)
    add_resource!(world, Grid3D(get_resource(world, WorldBounds), s.max_dist))
end

function sys_update!(s::BoidsNeighbors, world::World)
    tick = get_resource(world, Tick).tick
    grid = get_resource(world, Grid3D)
    md2 = s.max_dist * s.max_dist

    for c in grid.cells
        empty!(c)
    end

    for (entities, positions) in Query(world, (Pos,))
        for i in eachindex(entities, positions)
            ix, iy, iz = grid_cell(grid, positions[i].p)
            push!(grid.cells[ix, iy, iz], entities[i])
        end
    end

    for (entities, positions, neighbors, updates) in Query(world, (Pos, Neighbors, UpdateStep))
        for i in eachindex(entities, positions, neighbors, updates)
            tick % 30 != updates[i].step && continue
            p = positions[i].p
            me = entities[i]
            nn = neighbors[i].n
            empty!(nn)

            ix, iy, iz = grid_cell(grid, p)
            for dx in max(ix - 1, 1):min(ix + 1, grid.dims[1]),
                dy in max(iy - 1, 1):min(iy + 1, grid.dims[2]),
                dz in max(iz - 1, 1):min(iz + 1, grid.dims[3])

                for other in grid.cells[dx, dy, dz]
                    other == me && continue
                    op, = get_components(world, other, (Pos,))
                    d = p - op.p
                    dot(d, d) <= md2 && push!(nn, other)
                end
            end
        end
    end
end

# ============================================================================
# System: movement (separation / alignment / cohesion + boundary avoidance)
# ============================================================================

Base.@kwdef struct BoidsMovement <: System
    separation_weight::Float32 = 0.08f0
    separation_dist::Float32 = 3.0f0
    alignment_weight::Float32 = 0.04f0
    cohesion_weight::Float32 = 0.002f0
    min_speed::Float32 = 0.3f0
    max_speed::Float32 = 0.7f0
    margin::Float32 = 5.0f0
    margin_strength::Float32 = 0.06f0
end

function sys_update!(s::BoidsMovement, world::World)
    bounds = get_resource(world, WorldBounds)
    sep_dist2 = s.separation_dist^2

    for (_, positions, velocities, neighbors) in Query(world, (Pos, Vel, Neighbors))
        for i in eachindex(positions, velocities, neighbors)
            p = positions[i].p
            v = velocities[i].v
            nn = neighbors[i].n

            separation = Vec3f(0)
            avg_pos = Vec3f(0)
            avg_vel = Vec3f(0)

            for n in nn
                op, ov = get_components(world, n, (Pos, Vel))
                d = p - op.p
                d2 = dot(d, d)
                if d2 <= sep_dist2 && d2 > 0
                    separation += d / sqrt(d2)
                end
                avg_pos += Vec3f(op.p...)
                avg_vel += ov.v
            end

            nv = v
            if !isempty(nn)
                nf = Float32(length(nn))
                avg_pos /= nf
                avg_vel /= nf
                nv += separation * s.separation_weight
                nv += (avg_vel - v) * s.alignment_weight
                nv += (avg_pos - Vec3f(p...)) * s.cohesion_weight
            end

            # soft boundary avoidance
            for dim in 1:3
                lo_edge = bounds.lo[dim] + s.margin
                hi_edge = bounds.hi[dim] - s.margin
                if p[dim] < lo_edge
                    f = 1f0 - (p[dim] - bounds.lo[dim]) / s.margin
                    nv += Vec3f(ntuple(j -> j == dim ? s.margin_strength * f * f : 0f0, 3)...)
                elseif p[dim] > hi_edge
                    f = 1f0 - (bounds.hi[dim] - p[dim]) / s.margin
                    nv += Vec3f(ntuple(j -> j == dim ? -s.margin_strength * f * f : 0f0, 3)...)
                end
            end

            # clamp speed
            spd = norm(nv)
            if spd > 0
                nv = nv * clamp(spd, s.min_speed, s.max_speed) / spd
            end

            velocities[i] = Vel(nv)
            positions[i] = Pos(p + nv)
        end
    end
end

# ============================================================================
# Scene helpers
# ============================================================================

function collect_positions(world::World)
    out = Point3f[]
    for (_, pos) in Query(world, (Pos,))
        for p in pos
            push!(out, p.p)
        end
    end
    out
end

"""
    create_scene(; kwargs...) -> NamedTuple

Build the Ark world, run a warm-up, and create a Makie scene with gold boid
spheres.  Returns `(; scene, world, scheduler, mplot)`.
"""
function create_scene(;
    n_boids=500,
    bounds_min=Point3f(-20, -20, 0),
    bounds_max=Point3f(20, 20, 25),
    resolution=(1280, 720),
    warmup_steps=60,
)

    # ---- ECS ----------------------------------------------------------
    world = World(Pos, Vel, Neighbors, UpdateStep)
    add_resource!(world, WorldBounds(bounds_min, bounds_max))

    scheduler = Scheduler(world, (
        BoidsInit(n_boids),
        BoidsNeighbors(8f0),
        BoidsMovement(),
    ))
    initialize!(scheduler)

    for _ in 1:warmup_steps
        step!(scheduler)
    end

    positions = collect_positions(world)

    # ---- Makie scene --------------------------------------------------
    # No AmbientLight — gives clean black background with VolPath.
    # High RGB values needed: VolPath's spectral pipeline applies photometric
    # normalization (scale ≈ 1/10567), and lights are far from the boid cloud.
    scene = Scene(; size=resolution, camera=cam3d!,
        lights=[
            PointLight(RGBf(4000, 3500, 2500), Point3f(10, -20, 40)),   # warm key
            PointLight(RGBf(1200, 1200, 1500), Point3f(45, 35, 25)),    # cool fill
            PointLight(RGBf(600, 600, 900), Point3f(-30, 10, 30)),   # rim
        ],
    )

    cam = cameracontrols(scene)
    cam.eyeposition[] = Vec3f(42, 32, 10)
    cam.lookat[] = Vec3f(0, 0, 12)
    cam.upvector[] = Vec3f(0, 0, 1)
    cam.fov[] = 48f0
    update_cam!(scene, cam)

    # gold boid spheres
    mplot = meshscatter!(scene, positions;
        marker=Sphere(Point3f(0), 1f0),
        markersize=Vec3f(0.4),
        material=Hikari.Gold(roughness=0.25),
    )

    (; scene, world, scheduler, mplot)
end

# ============================================================================
# Rendering
# ============================================================================

function render_scene(;
    device=DEVICE,
    resolution=(1280, 720),
    samples=16,
    max_depth=6,
    output_path=joinpath(@__DIR__, "boids_flocking.png"),
)
    s = create_scene(; n_boids=500, warmup_steps=120, resolution=resolution)

    integrator = Hikari.VolPath(samples=samples, max_depth=max_depth)
    sensor = Hikari.FilmSensor(; iso=200, exposure_time=1.0, white_balance=0)

    img = colorbuffer(s.scene;
        device=device, integrator=integrator,
        tonemap=:aces, exposure=1.8f0, gamma=2.2f0, sensor=sensor,
    )
    save(output_path, img)
    @info "Saved → $output_path"
    return img
end

# ============================================================================
# Animation
# ============================================================================

function render_video(;
    device=DEVICE,
    resolution=(1280, 720),
    samples=8,
    max_depth=6,
    nframes=300,
    output_path=joinpath(@__DIR__, "boids_flocking.mp4"),
    n_boids=500,
    framerate=30,
)
    s = create_scene(; n_boids, resolution)

    integrator = Hikari.VolPath(samples=samples, max_depth=max_depth)
    sensor = Hikari.FilmSensor(; iso=200, exposure_time=1.0, white_balance=0)

    # Warm up
    colorbuffer(s.scene;
        device=device, integrator=integrator,
        tonemap=:aces, exposure=1.8f0, gamma=2.2f0, sensor=sensor,
    )

    Makie.record_longrunning(s.scene, output_path, 1:nframes;
        framerate=framerate, device=device, integrator=integrator,
        tonemap=:aces, exposure=1.8f0, gamma=2.2f0, sensor=sensor) do frame
        step!(s.scheduler)
        Makie.update!(s.mplot; arg1=collect_positions(s.world))
        println("Frame $frame/$nframes")
    end
    @info "Saved → $output_path"
end

# render_scene()
# render_video()
