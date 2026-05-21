# demo_render_mvp.jl
#
# MVP render demo: 1M randomly positioned, randomly rotated cubes rendered via
# Hikari path tracing using the GPU-resident pipeline built in P3.4a-d.
#
# Validates the full P1 + P2 + P3 stack end-to-end.
# No physics, no simulation -- random positions/rotations on GPU, render, save PNG.
#
# Run via:
#   julia> include("/sim/Programmieren/VulkanDev/RayDemo/SandCat/demo_render_mvp.jl")
#   julia> result = main(; n_grains=1000)    # smoke test (1k)
#   julia> result = main(; n_grains=1_000_000)  # full demo (1M)

using Lava
using Raycore
using Hikari
using RayMakie
using GeometryBasics
using GeometryBasics: Point3f, Vec3f, Vec4f, Point2f
using ImageCore: RGB
using Adapt
import KernelAbstractions
using Random
using FileIO

function build_demo(; n_grains::Int = 1_000_000,
                    radius::Float32 = 0.005f0,
                    seed::Int = 42)
    rng = MersenneTwister(seed)

    # 1. GPU-resident grain state.
    #    Random positions in a 1m^3 cube, random unit quaternions.
    cpu_positions = [Point3f(rand(rng, Float32) * 1f0 - 0.5f0,
                              rand(rng, Float32) * 1f0 - 0.5f0,
                              rand(rng, Float32) * 1f0 - 0.5f0) for _ in 1:n_grains]

    # Random unit quaternion via Marsaglia-style sampling: 4 normal samples normalized.
    cpu_quats = Vector{Vec4f}(undef, n_grains)
    for i in 1:n_grains
        q = Vec4f(randn(rng, Float32), randn(rng, Float32),
                  randn(rng, Float32), randn(rng, Float32))
        n = sqrt(q[1]*q[1] + q[2]*q[2] + q[3]*q[3] + q[4]*q[4])
        cpu_quats[i] = Vec4f(q[1]/n, q[2]/n, q[3]/n, q[4]/n)
    end

    positions = Lava.LavaArray(cpu_positions)
    rotations = Lava.LavaArray(cpu_quats)

    # 2. Build Hikari scene with hw_accel=true so meshscatter_create_gpu! works.
    backend = Lava.LavaBackend()
    scene = Hikari.Scene(; backend=backend, hw_accel=true)

    # Lights: spectral VolPath uses photometric normalization internally.
    # DirectionalLight(::RGB, ::Vec3f) and AmbientLight(::RGB) accept any RGB type.
    push!(scene, Hikari.DirectionalLight(RGB{Float32}(2.5f0, 2.4f0, 2.3f0),
                                          Vec3f(-0.4f0, -1.0f0, -0.3f0)))
    push!(scene, Hikari.AmbientLight(RGB{Float32}(0.3f0, 0.3f0, 0.35f0)))

    # 3. Register a shared Diffuse material for all grain instances.
    #    push!(scene, material) returns a UInt32 index into scene.media_interfaces.
    #    This mi_idx is passed to meshscatter_create_gpu! so every instance has a
    #    valid material and path-traced hits produce non-zero radiance.
    grain_material = Hikari.Diffuse(Kd=Hikari.RGBSpectrum(0.8f0, 0.7f0, 0.55f0))
    mi_idx = push!(scene, grain_material)

    # 4. Cube marker mesh: half-extents = 1 so scale=radius gives actual radius.
    cube = GeometryBasics.normal_mesh(
        GeometryBasics.Rect3f(Vec3f(-1f0, -1f0, -1f0), Vec3f(2f0, 2f0, 2f0)))

    # 5. Register the GPU-resident grain batch via the new path.
    #    instance_mask=0x04: rendering mask (used by VolPath cull_mask=0x04).
    #    mi_idx: all instances share the same Diffuse material registered above.
    println("Building BLAS + TLAS for ", n_grains, " grains...")
    t0 = time()
    robj = RayMakie.meshscatter_create_gpu!(
        scene, nothing, cube,
        positions, rotations,
        radius, UInt8(0x04);
        mi_idx=mi_idx)
    println("  BLAS+TLAS build: ", round((time() - t0) * 1000; digits=1), " ms")

    return (; scene, positions, rotations, cube, robj, n_grains, radius, backend)
end

function render_demo(demo;
                     resolution::Tuple{Int,Int} = (1920, 1080),
                     samples_per_pixel::Int = 16,
                     out::AbstractString = joinpath(@__DIR__, "out", "render_mvp.png"))
    backend = demo.backend

    # Camera: look at the unit cube of grains from a 3/4 angle.
    film = Hikari.Film(Point2f(Float32(resolution[1]), Float32(resolution[2]));
        filter=Hikari.LanczosSincFilter(Point2f(1f0), 3f0),
        crop_bounds=Hikari.Bounds2(Point2f(0f0), Point2f(1f0)),
        diagonal=1f0, scale=1f0)

    # Adapt film buffers to GPU.
    gpu_film = Adapt.adapt(backend, film)

    camera = Hikari.PerspectiveCamera(
        Point3f(2.0f0, 1.5f0, 2.0f0),
        Point3f(0.0f0, 0.0f0, 0.0f0),
        gpu_film;
        up  = Vec3f(0f0, 1f0, 0f0),
        fov = 45f0)

    # VolPath with cull_mask=0x04: only instances with mask & 0x04 != 0 are visible.
    # This is the mask used in meshscatter_create_gpu! above.
    vp = Hikari.VolPath(samples=samples_per_pixel, max_depth=4,
                        hw_accel=true, cull_mask=UInt32(0x04))

    println("Rendering ", demo.n_grains, " grains @ ", resolution[1], "x", resolution[2],
            " ", samples_per_pixel, " spp...")
    t0 = time()
    # vp(scene, film, camera) runs all samples + postprocess!.
    vp(demo.scene, gpu_film, camera)
    KernelAbstractions.synchronize(backend)
    t = time() - t0
    println("Render took ", round(t * 1000; digits=1), " ms")

    # Save the image: film.postprocess is RGBA{Float32} (display-ready).
    # FileIO.save needs a colorant array -- convert to RGB{N0f8} for PNG.
    img = Array(gpu_film.postprocess)
    mkpath(dirname(out))
    save(out, img)
    println("Wrote ", out, "  (", round(filesize(out)/1024; digits=1), " KiB)")

    return (; render_ms=t*1000, out_path=out)
end

function main(; n_grains::Int = 1_000_000,
               radius::Float32 = 0.005f0,
               resolution::Tuple{Int,Int} = (1920, 1080),
               samples_per_pixel::Int = 16,
               out::AbstractString = joinpath(@__DIR__, "out", "render_mvp.png"))
    demo = build_demo(; n_grains, radius)
    return render_demo(demo; resolution, samples_per_pixel, out)
end
