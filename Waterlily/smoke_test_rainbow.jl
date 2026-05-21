# smoke_test_rainbow.jl — exercises the moving-dolphin update path at low cost.
#
# Why: the rainbow_glow scene uses every tricky update path RayMakie has —
# mesh swap (mesh!), volumetric medium swap (MediumInterface in mesh!), every
# frame.  Running it at 160×90 / spp=2 / 5 frames takes ~seconds per backend
# and surfaces leaks, use-after-free, batch.bq desync, etc.

using Revise, Dates
using StaticArrays, GeometryBasics, Meshing, JLD2, FileIO, Colors
using Makie, RayMakie, Hikari, Raycore, Lava, LinearAlgebra, Statistics

# Pull in dolphin.jl helpers (compute_vorticity, set_panel_cam!, etc.) and the
# rainbow_glow panel/medium builders.  We re-include the helpers here rather
# than running render_dolphin_rainbow_glow.jl wholesale — it has top-level
# colorbuffer + record at full settings that we don't want to trigger.
include("./dolphin.jl")

function sdf_to_tris(sdf::Array{Float32,3}; sigma::Float32=1.0f0)
    s = sigma > 0f0 ? Meshing.smooth_sdf(sdf; sigma=sigma) : sdf
    ranges = range.((0f0, 0f0, 0f0), size(s))
    pts, fcs, nms = Meshing.isosurface_normals(s,
        Meshing.MarchingCubes(iso=0f0, reduceverts=true), ranges...)
    GeometryBasics.Mesh(Point3f.(pts), GLTriangleFace.(fcs); normal=Vec3f.(nms))
end

function load_smoke_step(path)
    jldopen(path, "r") do f
        u    = copy(f["u"])
        sdf  = copy(f["sdf"])
        tris = sdf_to_tris(sdf)
        (u=u, sdf=sdf, U_mag=1f0, body_mesh=tris, body_scale=1f0,
         t=Float32(f["t"]), sim_t=Float32(f["sim_t"]))
    end
end

function build_rainbow_glow_medium(vc;
        emission_scale::Float32=2.5f0,
        extinction_intensity::Float32=0.6f0,
        colormap_sym=:turbo,
        colorrange=(0.015f0, 0.13f0))
    nx, ny, nz = size(vc)
    cmap = Makie.to_colormap(colormap_sym)
    cmin, cmax = Float32(colorrange[1]), Float32(colorrange[2])
    crange = cmax - cmin
    crange < 1f-10 && (crange = 1f0)
    σ_a = Array{Hikari.RGBSpectrum, 3}(undef, nx, ny, nz)
    σ_s = Array{Hikari.RGBSpectrum, 3}(undef, nx, ny, nz)
    Le  = Array{Hikari.RGBSpectrum, 3}(undef, nx, ny, nz)
    @inbounds for i in eachindex(vc)
        d = Float32(vc[i])
        t = clamp((d - cmin) / crange, 0f0, 1f0)
        col = Makie.interpolated_getindex(cmap, t)
        r = Float32(col.r); g = Float32(col.g); b = Float32(col.b)
        t2 = t * t; t4 = t2 * t2
        w_emit = t4 * t2; w_scat = t
        σ_s[i] = Hikari.RGBSpectrum(r * 1.5f0 * w_scat, g * 1.5f0 * w_scat, b * 1.5f0 * w_scat)
        σ_a[i] = Hikari.RGBSpectrum(0.4f0 * w_scat, 0.4f0 * w_scat, 0.4f0 * w_scat)
        Le[i]  = Hikari.RGBSpectrum(r * 2f0 * w_emit, g * 2f0 * w_emit, b * 2f0 * w_emit)
    end
    bounds = Raycore.Bounds3(Point3f(1f0, 1f0, 1f0),
                              Point3f(Float32(nx), Float32(ny), Float32(nz)))
    Hikari.RGBGridMedium(σ_a_grid=σ_a, σ_s_grid=σ_s, Le_grid=Le,
        sigma_scale=extinction_intensity, Le_scale=emission_scale,
        g=0.6f0, bounds=bounds, majorant_res=Vec{3, Int64}(16, 16, 16))
end

function panel_rainbow_glow!(ax::LScene, dmesh, vc, cam;
        emission_scale=0.5f0, extinction_intensity=0.02f0,
        colormap_sym=:turbo, colorrange=(0.015f0, 0.13f0))
    dolphin = mesh!(ax, dmesh; material=Hikari.CoatedDiffuse(
        reflectance=(0.30f0, 0.32f0, 0.38f0), roughness=0.05f0, eta=1.5f0))
    medium = build_rainbow_glow_medium(vc;
        emission_scale, extinction_intensity, colormap_sym, colorrange)
    nx, ny, nz = size(vc)
    cube = GeometryBasics.normal_mesh(Rect3f(
        Vec3f(1f0, 1f0, 1f0),
        Vec3f(Float32(nx - 1), Float32(ny - 1), Float32(nz - 1))))
    boundary = Hikari.Dielectric(index=1.0f0, roughness=0.0f0)
    vol = mesh!(ax, cube; material=Hikari.MediumInterface(boundary; inside=medium))
    set_panel_cam!(ax, cam)
    return (dolphin=dolphin, vol=vol)
end

function update_rainbow_glow!(plots, vc;
        emission_scale=0.5f0, extinction_intensity=0.02f0,
        colormap_sym=:turbo, colorrange=(0.015f0, 0.13f0))
    new_medium = build_rainbow_glow_medium(vc;
        emission_scale, extinction_intensity, colormap_sym, colorrange)
    boundary = Hikari.Dielectric(index=1.0f0, roughness=0.0f0)
    Makie.update!(plots.vol; material=Hikari.MediumInterface(boundary; inside=new_medium))
    return nothing
end

# ── Smoke test driver ──────────────────────────────────────────────────────
function run_smoke(; hw_accel::Bool, spp=2, res=(160, 90), n_frames=5,
                     max_depth::Int=4,
                     steps_dir=joinpath(@__DIR__, "dolphin_smoke_L256_steps"),
                     outpath::Union{Nothing, String}=nothing)
    println("\n=== smoke run: hw_accel=$hw_accel spp=$spp res=$res frames=$n_frames ===")
    GC.gc(true); sleep(0.2)
    base_bufs = length(Lava.LIVE_BUFFERS)
    base_bytes = Lava.GPU_LIVE_BYTES[]
    println("baseline: bufs=$base_bufs  GPU=$(round(base_bytes/1e9, digits=2))GB")

    jld_frames = sort(filter(startswith("frame_"), readdir(steps_dir)))
    @assert length(jld_frames) >= n_frames "not enough frame files"

    step1 = load_smoke_step(joinpath(steps_dir, jld_frames[1]))
    dmesh = step1.body_mesh
    vc = compute_vorticity(step1, 10f0; transition=2f0)

    sx, sy, sz = size(step1.u)[1:3]
    scale = Float32(sx / 96)
    body_centre = let v = dmesh.position
        Vec3f(mean(p[1] for p in v), mean(p[2] for p in v), mean(p[3] for p in v))
    end
    cam = (eye    = Vec3f(843.27, -153.6, 238.31) .* scale,
           lookat = body_centre + Vec3f(0, 30, 0),
           up     = Vec3f(-0.17, 0.06, 0.98),
           fov    = 10.0)

    sun_dir = normalize(Vec3f(-0.4, -0.15, 0.9))
    sunsky = Makie.SunSkyLight(sun_dir;
        intensity=1.5f0, turbidity=3.0f0,
        ground_albedo=RGBf(0.10, 0.12, 0.15), ground_enabled=false)
    bcx = Float32(sx / 2); bcy = Float32(sy / 2); bcz = Float32(sz / 2)
    lights = [
        sunsky,
        Makie.PointLight(RGBf(18000, 14000, 9000), Vec3f(bcx - 40, bcy - 80, bcz + 80) .* scale),
        Makie.PointLight(RGBf(1500, 2500, 4500),   Vec3f(bcx + 60, bcy + 40, bcz - 40) .* scale),
    ]

    fig = Figure(size=res; backgroundcolor=RGBf(0,0,0), figure_padding=0)
    ax = LScene(fig[1,1]; show_axis=false,
                scenekw=(lights=lights, backgroundcolor=RGBf(0,0,0)))
    plots = panel_rainbow_glow!(ax, dmesh, vc, cam;
        emission_scale=0.5f0, extinction_intensity=0.02f0)

    integrator = Hikari.VolPath(; samples=spp, max_depth=max_depth, hw_accel=hw_accel)
    RayMakie.activate!(; tonemap=:aces, gamma=2.2f0, exposure=1f0, denoise=false)

    timings = Float64[]
    bufs_per_frame = Int[]
    # Use Makie.record_longrunning when an outpath is given — that's the
    # resumable canonical recorder (frame PNGs in a sibling `_frames` dir +
    # ffmpeg stitch at the end + skip already-rendered frames).  Without an
    # outpath we just colorbuffer per frame for the leak/perf measurement.
    if outpath === nothing
        for i in 1:n_frames
            if i > 1
                step = load_smoke_step(joinpath(steps_dir, jld_frames[i]))
                new_vc = compute_vorticity(step, 10f0; transition=2f0)
                Makie.update!(plots.dolphin; arg1=step.body_mesh)
                update_rainbow_glow!(plots, new_vc; emission_scale=0.5f0, extinction_intensity=0.02f0)
            end
            t0 = time()
            img = colorbuffer(fig; backend=RayMakie, integrator=integrator, update=false)
            push!(timings, time() - t0)
            push!(bufs_per_frame, length(Lava.LIVE_BUFFERS))
            gpu_mib = Lava.GPU_LIVE_BYTES[] >> 20
            println("  frame $i: $(round(timings[end], digits=2))s  $(size(img))  bufs=$(bufs_per_frame[end])  GPU=$(gpu_mib)MiB")
        end
    else
        Makie.record_longrunning(fig, outpath, 1:n_frames;
                                 framerate=30, update=false,
                                 backend=RayMakie, integrator=integrator) do i
            i == 1 && return nothing  # frame 1 already corresponds to construct_scene state
            step = load_smoke_step(joinpath(steps_dir, jld_frames[i]))
            new_vc = compute_vorticity(step, 10f0; transition=2f0)
            Makie.update!(plots.dolphin; arg1=step.body_mesh)
            update_rainbow_glow!(plots, new_vc; emission_scale=0.5f0, extinction_intensity=0.02f0)
            push!(bufs_per_frame, length(Lava.LIVE_BUFFERS))
            gpu_mib = Lava.GPU_LIVE_BYTES[] >> 20
            println("  frame $i: bufs=$(length(Lava.LIVE_BUFFERS))  GPU=$(gpu_mib)MiB")
        end
        println("\nencoded video: $outpath")
    end

    GC.gc(true); sleep(0.3); GC.gc(true)
    final_bufs = length(Lava.LIVE_BUFFERS)
    final_bytes = Lava.GPU_LIVE_BYTES[]
    println("after GC: bufs=$final_bufs (Δ=$(final_bufs-base_bufs))  GPU=$(round(final_bytes/1e9, digits=2))GB (Δ=$(round((final_bytes-base_bytes)/1e9, digits=2))GB)")
    return (; timings, bufs_per_frame, final_bufs, final_bytes)
end
