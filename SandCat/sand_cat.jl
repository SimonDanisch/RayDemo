# sand_cat.jl
#
# Sand-cat simulation demo.  Position-based-dynamics granular simulation of
# sand grains falling onto Makie's cat.obj, with grain-grain and grain-mesh
# collisions resolved via inline lava_ray_query_* intrinsics against a hardware
# TLAS.
#
# Build incrementally:
#   1. Static scene + render          (this commit)
#   2. Gravity + ground collision     (CPU, debug-only kernel surfaces)
#   3. Grain-mesh ray-cast collision  (compute kernel against cat triangle BLAS)
#   4. Grain-grain PBD via rayQuery   (compute kernel against grain AABB BLAS)
#   5. Render to MP4 (RayMakie)
#
# Run via:
#   julia> include("/sim/Programmieren/VulkanDev/RayDemo/SandCat/sand_cat.jl")

using Lava
using Raycore
using GeometryBasics
using GeometryBasics: Point3f, Vec3f, GLTriangleFace
using LinearAlgebra: I
using StaticArrays: SMatrix
using Random
using FileIO
using Makie
using Makie: RGBf

const Mat4f = SMatrix{4, 4, Float32, 16}

# Internal helpers, kept close to the entry point so the demo is one short read.
const SAND_CAT_DIR = @__DIR__
include(joinpath(SAND_CAT_DIR, "scene.jl"))
include(joinpath(SAND_CAT_DIR, "physics.jl"))

"""
    SandCatScene

Holds GPU-side state for the demo.  Built once, consumed by `step!`.
"""
mutable struct SandCatScene
    # Geometry
    cat_mesh::GeometryBasics.Mesh
    cat_bbox::Tuple{Point3f, Point3f}
    radius::Float32
    floor_y::Float32

    # Per-grain state (CPU-side authoritative copy; GPU buffers refilled per step)
    positions::Vector{Point3f}
    velocities::Vector{Vec3f}

    # Pre-computed per-triangle geometric normals (CPU + GPU)
    tri_normals::Vector{Vec3f}
    tri_normals_gpu::Lava.LavaArray{Vec3f, 1}

    # GPU acceleration structures
    tlas::Lava.HWTLAS
    grain_blas::Union{Nothing, Lava.LavaBLAS}
    grain_handle::Union{Nothing, Raycore.TLASHandle}
    cat_handle::Union{Nothing, Raycore.TLASHandle}

    # Configuration
    backend::Lava.LavaBackend
end

"""
    build_scene(; n_grains=1_000_000, radius=0.0008f0, spawn=:cat_shape, seed=42)
        -> SandCatScene

Construct the demo scene.  Two spawn modes:

- `:cat_shape` (default) — sample grain positions inside the cat's volume via
  parity ray-casting against a Raycore software TLAS, so the initial state IS
  a sand statue of the cat which collapses under gravity.  The cat mesh is NOT
  added to the HWTLAS (grains start embedded inside it; ray-querying against
  the cat would just hit immediately every step).  `tri_normals` is a dummy
  one-element vector that the kernel reads but never uses (`hit_cat` stays
  false because no instances are tagged with mask `0x01`).
- `:cube` — fall sand from a cube above the cat, with cat triangle BLAS in the
  TLAS so grains bounce off it.  Older spawn mode used for early bring-up.
"""
function build_scene(; n_grains::Int = 1_000_000,
                     radius::Float32 = 0.0008f0,
                     spawn::Symbol = :cat_shape,
                     seed::Int = 42,
                     jitter::Float32 = 0.3f0)
    cat_mesh = load_cat_mesh()
    bb_min, bb_max = cat_bounds(cat_mesh)
    # Round up to a multiple of the workgroup size (64): kernels run with
    # one-thread-per-grain and the SPIR-V emitter requires a branch-free path
    # to the first ray query, so we cannot use a per-thread in_range guard.
    n_grains = ((n_grains + 63) ÷ 64) * 64
    floor_y = bb_min[2] - 0.05f0          # below the cat's feet

    if spawn === :cat_shape
        # Sample inside the cat volume via parity ray-casting.
        cat_sw_tlas = Raycore.TLAS([cat_mesh], (mi, ti) -> UInt32(0))
        positions = sample_inside_mesh(cat_sw_tlas, bb_min, bb_max, n_grains; seed)
        # Pad to multiple of 64 in the rare case rejection produced fewer
        # points than the rounded-up target.
        if length(positions) < n_grains
            pad = fill(positions[end], n_grains - length(positions))
            positions = vcat(positions, pad)
        end
        velocities = fill(Vec3f(0f0, 0f0, 0f0), n_grains)

        # Cat is NOT in the TLAS for cat-shape spawn; tri_normals is a dummy.
        tri_normals = [Vec3f(0f0, 1f0, 0f0)]
        tri_normals_gpu = LavaArray(tri_normals)

        backend = Lava.LavaBackend()
        aabbs = grain_aabbs(positions, radius)
        grain_blas = Lava.as_build() do ctx
            Lava.build_blas_aabb(ctx, aabbs)
        end
        tlas = Lava.HWTLAS(backend)
        grain_handle = push!(tlas, grain_blas, Mat4f(I);
                             instance_id = UInt32(1), instance_mask = UInt8(0x02))
        Raycore.sync!(tlas)
        return SandCatScene(
            cat_mesh, (bb_min, bb_max), radius, floor_y,
            positions, velocities,
            tri_normals, tri_normals_gpu,
            tlas, grain_blas, grain_handle, nothing,
            backend,
        )
    elseif spawn === :cube
        tri_normals = triangle_normals(cat_mesh)
        tri_normals_gpu = LavaArray(tri_normals)

        # Cube above the cat at near close-packing density; cell = 2.2*radius.
        cell = 2.2f0 * radius
        side = ceil(Int, cbrt(Float32(n_grains)))
        spawn_extent = cell * Float32(side)
        cat_cx = (bb_min[1] + bb_max[1]) * 0.5f0
        cat_cz = (bb_min[3] + bb_max[3]) * 0.5f0
        region_min = Point3f(cat_cx - spawn_extent * 0.5f0,
                             bb_max[2] + 0.4f0,
                             cat_cz - spawn_extent * 0.5f0)
        region_max = Point3f(cat_cx + spawn_extent * 0.5f0,
                             bb_max[2] + 0.4f0 + spawn_extent,
                             cat_cz + spawn_extent * 0.5f0)
        rng = MersenneTwister(seed)
        cell_extent = (region_max - region_min)[1] / cbrt(Float32(n_grains))
        j = min(jitter * cell_extent, cell_extent * 0.49f0)
        positions = spawn_grain_grid(; n_grains, region_min, region_max, jitter=j, rng)
        velocities = fill(Vec3f(0f0, 0f0, 0f0), n_grains)

        backend = Lava.LavaBackend()
        aabbs = grain_aabbs(positions, radius)
        grain_blas = Lava.as_build() do ctx
            Lava.build_blas_aabb(ctx, aabbs)
        end
        tlas = Lava.HWTLAS(backend)
        cat_handle = push!(tlas, cat_mesh, Mat4f(I);
                           instance_id = UInt32(0), instance_mask = UInt8(0x01))
        grain_handle = push!(tlas, grain_blas, Mat4f(I);
                             instance_id = UInt32(1), instance_mask = UInt8(0x02))
        Raycore.sync!(tlas)
        return SandCatScene(
            cat_mesh, (bb_min, bb_max), radius, floor_y,
            positions, velocities,
            tri_normals, tri_normals_gpu,
            tlas, grain_blas, grain_handle, cat_handle,
            backend,
        )
    else
        error("build_scene: spawn must be :cat_shape or :cube, got $spawn")
    end
end

"""
    rebuild_grain_blas!(scene)

Rebuild the grain AABB BLAS from `scene.positions` and swap it into the TLAS,
replacing the previous grain instance.  After this the TLAS is `sync!`'d and
ready for the next compute dispatch.
"""
function rebuild_grain_blas!(scene::SandCatScene)
    aabbs = grain_aabbs(scene.positions, scene.radius)
    new_blas = Lava.as_build() do ctx
        Lava.build_blas_aabb(ctx, aabbs)
    end

    # Drop the old grain instance, push the new BLAS, sync.
    Raycore.delete!(scene.tlas, scene.grain_handle)
    scene.grain_blas = new_blas
    scene.grain_handle = push!(scene.tlas, new_blas, Mat4f(I);
                               instance_id = UInt32(1),
                               instance_mask = UInt8(0x02))
    Raycore.sync!(scene.tlas)
    return scene
end

mutable struct PingState
    pos_in::Lava.LavaArray{Point3f, 1}
    pos_out::Lava.LavaArray{Point3f, 1}
    vel_in::Lava.LavaArray{Vec3f, 1}
    vel_out::Lava.LavaArray{Vec3f, 1}
end

"""
    make_ping_state(scene) -> PingState

Allocate the GPU buffers for the ping-pong PBD step.
"""
function make_ping_state(scene::SandCatScene)
    pos_in  = LavaArray(scene.positions)
    vel_in  = LavaArray(scene.velocities)
    pos_out = LavaArray(copy(scene.positions))
    vel_out = LavaArray(copy(scene.velocities))
    return PingState(pos_in, pos_out, vel_in, vel_out)
end

"""
    step!(scene, dt, ping_state; substeps=1) -> nothing

Advance the simulation by `dt` seconds, optionally subdivided into `substeps`
GPU dispatches of `pbd_step_kernel`.  Each substep reads from `pos_in`/`vel_in`
and writes `pos_out`/`vel_out`, swaps roles, copies positions back to the CPU,
and rebuilds the grain AABB BLAS so the next substep sees up-to-date neighbours.

Substepping is required when grains are small relative to per-frame
displacement: at 5mm grains and dt=1/60, a falling grain at terminal velocity
~6 m/s moves ~100mm per step, i.e. 20 grain diameters, which tunnels right
through the pile and prevents stacking.  Substepping shrinks the effective dt
and BLAS-rebuild interval so neighbours are caught before they pass through
each other.
"""
function step!(scene::SandCatScene, dt::Float32, ping_state::PingState; substeps::Int = 1)
    bb_min, bb_max = scene.cat_bbox
    pad = 0.4f0
    x_min = bb_min[1] - pad
    x_max = bb_max[1] + pad
    z_min = bb_min[3] - pad
    z_max = bb_max[3] + pad

    n = UInt32(length(scene.positions))
    bq = scene.backend.bq
    sub_dt = dt / Float32(substeps)

    @inbounds for _ in 1:substeps
        Lava.lava_launch!(bq, pbd_step_kernel,
            ping_state.pos_in, ping_state.vel_in,
            ping_state.pos_out, ping_state.vel_out,
            scene.tri_normals_gpu,
            scene.floor_y, x_min, x_max, z_min, z_max,
            scene.radius, sub_dt, n;
            ndrange = Int(n), workgroup_size = (64, 1, 1),
            tlas = scene.tlas)
        Lava.vk_flush!(bq)

        # Swap roles for next substep
        ping_state.pos_in, ping_state.pos_out = ping_state.pos_out, ping_state.pos_in
        ping_state.vel_in, ping_state.vel_out = ping_state.vel_out, ping_state.vel_in

        # Mirror positions to CPU and rebuild the grain BLAS for the next substep.
        copyto!(scene.positions, Array(ping_state.pos_in))
        rebuild_grain_blas!(scene)
    end
    return nothing
end

"""
    summarize(scene)

Print a one-line summary of the scene state, useful for sanity-checking
without dumping the giant `vk_context` to stdout.
"""
function summarize(scene::SandCatScene)
    bb_min, bb_max = scene.cat_bbox
    pmn = reduce((a, b) -> Point3f(min.(a, b)), scene.positions; init=Point3f(Inf32))
    pmx = reduce((a, b) -> Point3f(max.(a, b)), scene.positions; init=Point3f(-Inf32))
    println("SandCatScene:")
    println("  cat   bbox min=", bb_min, " max=", bb_max)
    println("  grain bbox min=", pmn,    " max=", pmx)
    println("  n_grains=", length(scene.positions),
            " radius=", scene.radius,
            " floor_y=", scene.floor_y)
    println("  TLAS  n_geometries=", Raycore.n_geometries(scene.tlas),
            " n_instances=", Raycore.n_instances(scene.tlas))
    return nothing
end

"""
    grain_palette(n; seed=11) -> Vector{RGBf}

Per-grain warm-beige color with small hue jitter.  Computed once at scene
setup; the Makie color observable doesn't update during the simulation.
"""
function grain_palette(n::Integer; seed::Int = 11)
    rng = MersenneTwister(seed)
    base_r = 0.85f0; base_g = 0.78f0; base_b = 0.55f0
    out = Vector{RGBf}(undef, n)
    @inbounds for i in 1:n
        jr = 0.85f0 + 0.30f0 * rand(rng, Float32)
        jg = 0.85f0 + 0.30f0 * rand(rng, Float32)
        jb = 0.85f0 + 0.30f0 * rand(rng, Float32)
        out[i] = RGBf(base_r * jr, base_g * jg, base_b * jb)
    end
    return out
end

"""
    run_demo(; n_grains=50_000, radius=0.008f0, n_frames=240,
             resolution=(1920, 1080), backend=:raymakie,
             samples_per_pixel=16, max_depth=5,
             out_dir=...,
             out=joinpath(out_dir, "sand_cat_v2.mp4"))

End-to-end driver: build the scene, step the PBD simulation, render one
RayMakie (or GLMakie) frame per step, encode an MP4 with ffmpeg.  Returns
the path to the encoded video.
"""
function run_demo(; n_grains::Int = 50_000,
                  radius::Float32 = 0.008f0,
                  n_frames::Int = 240,
                  resolution::Tuple{Int,Int} = (1920, 1080),
                  backend::Symbol = :raymakie,
                  samples_per_pixel::Int = 16,
                  max_depth::Int = 5,
                  out_dir::AbstractString = joinpath(SAND_CAT_DIR, "out"),
                  out::AbstractString = joinpath(out_dir, "sand_cat_v2.mp4"))

    # Skip backend activation if backend == :preloaded.  This avoids the
    # world-age error you get when GLMakie/RayMakie is loaded via @eval inside
    # this function and then immediately used to save frames.  Caller
    # activates the backend at top level first.
    if backend !== :preloaded
        activate_backend(Val(backend); samples_per_pixel, max_depth)
    end

    scene = build_scene(; n_grains, radius)
    ping  = make_ping_state(scene)
    println("SandCat: ", length(scene.positions), " grains, radius=", scene.radius)

    # Build the Makie scene.  Use a top-level Scene with lights set at
    # construction time, so RayMakie picks them up before init.
    lights = [
        Makie.DirectionalLight(RGBf(2.2, 2.1, 2.0), Vec3f(-0.5, -1.0, -0.3)),
        Makie.AmbientLight(RGBf(0.25, 0.27, 0.32)),
    ]
    mscene = Makie.Scene(; size=resolution,
                         backgroundcolor = RGBf(0.05, 0.06, 0.08),
                         camera = Makie.cam3d!,
                         lights = lights)

    # Ground plane for visual context (matches the simulated floor_y).
    floor_y = scene.floor_y
    cat_min, cat_max = scene.cat_bbox
    ground_pad = 1.5f0
    ground = Rect3f(Vec3f(cat_min[1] - ground_pad, floor_y - 0.02f0, cat_min[3] - ground_pad),
                    Vec3f((cat_max[1] - cat_min[1]) + 2f0 * ground_pad, 0.02f0,
                          (cat_max[3] - cat_min[3]) + 2f0 * ground_pad))
    mesh!(mscene, ground; color = RGBf(0.55, 0.50, 0.42))

    # Cat: matte off-white plaster look.
    mesh!(mscene, scene.cat_mesh; color = RGBf(0.92, 0.91, 0.88))

    # Per-grain warm-beige palette with hue jitter.  Color observable is set
    # once at scene setup; positions update each frame.
    positions_obs = Observable(copy(scene.positions))
    colors = grain_palette(length(scene.positions))
    # Cube marker (matches the AABB primitive used in physics).  At sub-pixel
    # rasterized sizes spheres vs cubes are indistinguishable, but the cube
    # mesh has 8 verts vs Makie's default sphere's tens of vertices, so it
    # renders faster at 1M+ instance counts.
    cube_marker = Rect3f(Vec3f(-1, -1, -1), Vec3f(2, 2, 2))
    meshscatter!(mscene, positions_obs;
                 marker     = cube_marker,
                 markersize = scene.radius,
                 color      = colors)

    # Camera: 3/4 view, slightly elevated, framed on cat.
    Makie.update_cam!(mscene,
                      Vec3f(2.5, 1.5, 2.5),
                      Vec3f(0.0, 0.3, 0.0),
                      Vec3f(0, 1, 0))

    frames_dir = joinpath(out_dir, "frames_v2")
    rm(frames_dir; force=true, recursive=true)
    mkpath(frames_dir)

    t0 = time()
    for f in 1:n_frames
        ts = time()
        step!(scene, Float32(1/60), ping)
        positions_obs[] = copy(scene.positions)
        frame_path = joinpath(frames_dir, "frame_$(lpad(f, 4, '0')).png")
        save(frame_path, mscene)
        println("frame ", f, "/", n_frames,
                "  step+render=", round(time() - ts; digits=2), "s")
    end
    println("Total simulation+render: ", round(time() - t0; digits=1), "s")

    mkpath(dirname(out))
    cmd = `ffmpeg -y -framerate 60 -i $(joinpath(frames_dir, "frame_%04d.png"))
           -c:v libx264 -pix_fmt yuv420p -crf 16 $out`
    run(pipeline(cmd; stdout=devnull, stderr=devnull))
    println("Wrote ", out, " (", round(filesize(out)/1024/1024; digits=2), " MiB)")
    return out
end

# Lazily load the requested rendering backend.
function activate_backend(::Val{:raymakie}; samples_per_pixel::Int = 16, max_depth::Int = 5)
    @eval Main begin
        using RayMakie
        using Hikari
        RayMakie.activate!(integrator = RayMakie.VolPath(samples = $samples_per_pixel,
                                                        max_depth = $max_depth))
    end
    return nothing
end

function activate_backend(::Val{:glmakie}; kwargs...)
    @eval Main begin
        using GLMakie
        GLMakie.activate!()
    end
    return nothing
end
