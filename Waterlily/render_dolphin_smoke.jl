# Render the dolphin-smoke video via dolphin.jl's panel_warm_hero! pipeline,
# feeding JLD2 sim frames produced by generate_dolphin_smoke.jl. Per-frame
# update path matches dolphin.jl's update_scene! (Makie.update! on arg1/arg4).
#
# This file is meant to be edited and run line-by-line in the REPL.
using Revise, Dates
using StaticArrays, GeometryBasics, Meshing, JLD2, FileIO, Colors
using Makie, RayMakie, Hikari, Raycore, Lava, LinearAlgebra, Statistics

include("./dolphin.jl")

function sdf_to_tris(sdf::Array{Float32,3}; sigma::Float32=1.0f0)
    # Gaussian-smooth the SDF before meshing to suppress grid-aligned banding;
    # reduceverts welds shared edge verts; isosurface_normals returns per-vertex
    # gradient normals (smooth, analytical) instead of face normals.
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
        (u=u, U_mag=1f0, body_mesh=tris, body_scale=1f0,
         t=Float32(f["t"]), sim_t=Float32(f["sim_t"]))
    end
end

# ---- render settings (edit these) ----
spp     = 10
res_w   = 480
res_h   = 270
nframes = 20

steps_dir = joinpath(@__DIR__, "dolphin_smoke_L256_steps")
video_out = joinpath(@__DIR__, "dolphin_smoke_warm_$(res_w)x$(res_h)_spp$(spp).mp4")
jld_frames = sort(filter(startswith("frame_"), readdir(steps_dir)))
n_to_render = min(nframes, length(jld_frames))
step1 = load_smoke_step(joinpath(steps_dir, jld_frames[1]))
GLMakie.mesh(step1.body_mesh; color=:white) # precompile mesh recipe before timing

dmesh = step1.body_mesh
vc = compute_vorticity(step1)
# Wrap in Observables so per-frame updates propagate via Makie's reactive
# path (the same path GLMakie uses successfully). Makie.update!(plot; arg1=…)
# was silently dropped by RayMakie's mesh recipe — confirmed via GLMakie diff.
dmesh_obs = Observable(dmesh)
vc_obs    = Observable(vc)

sx, sy, sz = size(step1.u)[1:3]
bcx = Float32(sx/2); bcy = Float32(sy/2); bcz = Float32(sz/2)
# cam3 from dolphin.jl:501 tuned for L=192 (domain 96×288×96). Scale to L=256
# (130×386×130).
scale = Float32(sx / 96)
# Aim lookat at the dolphin's actual body centre (from the extracted mesh)
# rather than the reference cam's offset-downstream point — centres the
# dolphin in the frame.
body_centre = let v = dmesh.position
    Vec3f(mean(p[1] for p in v), mean(p[2] for p in v), mean(p[3] for p in v))
end
cam_warm = (eye    = Vec3f(843.27, -153.6, 238.31) .* scale,
            lookat = body_centre + Vec3f(0, 30, 0),    # +Y = downstream; pushes body left, exposes wake on right
            up     = Vec3f(-0.17, 0.06, 0.98),
            fov    = 10.0)

# Warm point-light cluster (dolphin.jl:527-532) + add a SunSkyLight for a
# soft global ambient.
sun_dir = normalize(Vec3f(-0.4, -0.15, 0.9))
sunsky = Makie.SunSkyLight(sun_dir;
    intensity=0.8f0, turbidity=3.0f0,
    ground_albedo=RGBf(0.15, 0.18, 0.22), ground_enabled=false)
lights_warm = [
    sunsky,
    Makie.PointLight(RGBf(16000,12000,7000), Vec3f(bcx-50, bcy-90, bcz+60) .* scale),
    Makie.PointLight(RGBf(1500,2500,4000),   Vec3f(bcx+60, bcy+40, bcz+25) .* scale),
    Makie.PointLight(RGBf(1000,900,500),     Vec3f(bcx,    bcy+100, bcz+50) .* scale),
    Makie.PointLight(RGBf(1000,1200,2000),   Vec3f(bcx,    bcy-20, bcz-30) .* scale),
]

fig = Figure(size=(res_w, res_h); backgroundcolor=RGBf(0.012, 0.02, 0.05), figure_padding=0)
ax = LScene(fig[1, 1]; show_axis=false,
            scenekw=(lights=lights_warm, backgroundcolor=RGBf(0.012, 0.02, 0.05)))
panel_warm_hero!(ax, dmesh_obs, vc_obs, cam_warm)

RayMakie.activate!(; tonemap=:aces, gamma=2.2f0, exposure=1f0, denoise=false)
integrator = Hikari.VolPath(; samples=spp, max_depth=8, hw_accel=false)

# ---- record ----
t_start = time()
Makie.record_longrunning(fig, video_out, 1:n_to_render;
                         framerate=30, update=true, backend=RayMakie, integrator=integrator) do i
    if i == 1
        return nothing
    end
    step = load_smoke_step(joinpath(steps_dir, jld_frames[i]))
    dmesh_obs[] = step.body_mesh
    vc_obs[]    = compute_vorticity(step)
    elapsed = time() - t_start
    eta = elapsed / i * (n_to_render - i)
    println("[$(Dates.now())] frame $i/$n_to_render  sim_t=$(step.sim_t)  elapsed=$(round(elapsed,digits=1))s  eta=$(round(eta,digits=1))s")
end
println("[$(Dates.now())] DONE → $video_out")
