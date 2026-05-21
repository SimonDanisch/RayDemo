# Dolphin rainbow-glow stress test. Renders the undulating dolphin through
# a vorticity volume that *emits* light per-voxel (real volumetric glow via
# `RGBGridMedium.Le_grid`), colored by a turbo/rainbow colormap. HW RT on.
#
# Template: render_dolphin_smoke.jl (same sim-step dir + Observable-driven
# per-frame updates). What's new here:
#   * panel_rainbow_glow! — emissive RGB grid inside a near-invisible
#     dielectric boundary (NullMaterial breaks HW RT's shadow-ray transit).
#   * build_rainbow_glow_medium — maps vorticity via Makie colormap to a
#     per-voxel Le_grid (emission) and matching σ_s_grid (scattering),
#     both modulated by (density)^2 so only the bright cores glow.
#
# This file is meant to be edited and run line-by-line in the REPL.
using Revise, Dates
using StaticArrays, GeometryBasics, Meshing, JLD2, FileIO, Colors
using Makie, RayMakie, Hikari, Raycore, Lava, LinearAlgebra, Statistics

include("./dolphin.jl")

# ── sim-step loading (shared shape with render_dolphin_smoke.jl) ────────────

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
        # sdf is carried so `compute_vorticity(data, crop)` can SDF-mask the body
        # region out of the smoke volume.
        (u=u, sdf=sdf, U_mag=1f0, body_mesh=tris, body_scale=1f0,
         t=Float32(f["t"]), sim_t=Float32(f["sim_t"]))
    end
end

# ── Rainbow-glow medium builder ─────────────────────────────────────────────

"""
    build_rainbow_glow_medium(vc; emission_scale, extinction_intensity,
                              colormap_sym, colorrange) -> RGBGridMedium

One rainbow voxel grid. For each voxel:
  * `t = clamp((vc - cmin) / (cmax - cmin), 0, 1)`  (vorticity → density)
  * `color = colormap(t)` (turbo by default — perceptually uniform rainbow)
  * `w_emit = t^4`  (sparse — only the vortex-core tips emit light)
  * `w_scat = sqrt(t)`  (shallow — whole wake participates as colored fog)
  * `Le_grid[i]  = color * w_emit`         (sparse emission)
  * `σ_s_grid[i] = color * 0.3 * w_scat`   (voluminous scattering)
  * `σ_a_grid[i] = 0.05 * w_scat`          (mild self-shadow)

Effect: bright glowing vortex cores embedded in a translucent coloured fog,
rather than the whole volume being uniformly emissive. `Le_scale` and
`sigma_scale` are exposed as global multipliers.
"""
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
        t2 = t * t
        t4 = t2 * t2
        w_emit = t4 * t2          # t^6 — only the very vortex-core peaks glow
        w_scat = t                # linear — density proportional to vorticity
        # Scattering gives the body + colour of the smoke; absorption gives
        # the depth cue (backside darkens, lit side bright).
        σ_s[i] = Hikari.RGBSpectrum(r * 1.5f0 * w_scat, g * 1.5f0 * w_scat, b * 1.5f0 * w_scat)
        σ_a[i] = Hikari.RGBSpectrum(0.4f0 * w_scat, 0.4f0 * w_scat, 0.4f0 * w_scat)
        Le[i]  = Hikari.RGBSpectrum(r * 2f0 * w_emit, g * 2f0 * w_emit, b * 2f0 * w_emit)
    end

    bounds = Raycore.Bounds3(
        Point3f(1f0, 1f0, 1f0),
        Point3f(Float32(nx), Float32(ny), Float32(nz)))
    # Forward-scattering phase function (Henyey-Greenstein g=0.6) — classic
    # cloud/smoke look, backlit edges brighten naturally.
    return Hikari.RGBGridMedium(
        σ_a_grid=σ_a, σ_s_grid=σ_s, Le_grid=Le,
        sigma_scale=extinction_intensity, Le_scale=emission_scale,
        g=0.6f0, bounds=bounds, majorant_res=Vec{3, Int64}(16, 16, 16))
end

# ── Rainbow-glow panel (dolphin mesh + emissive volume) ─────────────────────

function panel_rainbow_glow!(ax::LScene, dmesh, vc, cam;
        emission_scale::Float32=2.5f0,
        extinction_intensity::Float32=0.6f0,
        colormap_sym=:turbo,
        colorrange=(0.015f0, 0.13f0))

    # Mid-grey coated-diffuse body — dark enough that the rainbow-coloured wake
    # still reads as the subject, bright enough that the dolphin itself is
    # visible through the residual smoke haze.
    dolphin = mesh!(ax, dmesh; material=Hikari.CoatedDiffuse(
        reflectance=(0.30f0, 0.32f0, 0.38f0), roughness=0.05f0, eta=1.5f0))

    medium = build_rainbow_glow_medium(vc;
        emission_scale, extinction_intensity, colormap_sym, colorrange)
    nx, ny, nz = size(vc)
    cube = GeometryBasics.normal_mesh(Rect3f(
        Vec3f(1f0, 1f0, 1f0),
        Vec3f(Float32(nx - 1), Float32(ny - 1), Float32(nz - 1))))

    boundary = Hikari.Dielectric(index=1.0f0, roughness=0.0f0)
    vol = mesh!(ax, cube; material=Hikari.MediumInterface(
        boundary; inside=medium))

    set_panel_cam!(ax, cam)
    return (dolphin=dolphin, vol=vol)
end

# ── Per-frame medium update ────────────────────────────────────────────────

function update_rainbow_glow!(plots, vc;
        emission_scale::Float32=2.5f0,
        extinction_intensity::Float32=0.6f0,
        colormap_sym=:turbo,
        colorrange=(0.015f0, 0.13f0))
    new_medium = build_rainbow_glow_medium(vc;
        emission_scale, extinction_intensity, colormap_sym, colorrange)
    boundary = Hikari.Dielectric(index=1.0f0, roughness=0.0f0)
    Makie.update!(plots.vol; material=Hikari.MediumInterface(
        boundary; inside=new_medium))
    return nothing
end

# ---- render settings (edit these) ----
spp        = 250
res_w      = 480*2
res_h      = 270*2
nframes    = 10
hw_accel   = false
max_depth  = 10

steps_dir  = joinpath(@__DIR__, "dolphin_smoke_L256_steps")
video_out  = joinpath(@__DIR__, "dolphin_rainbow_glow_$(res_w)x$(res_h)_spp$(spp).mp4")
jld_frames = sort(filter(startswith("frame_"), readdir(steps_dir)))
n_to_render = min(nframes, length(jld_frames))

step1 = load_smoke_step(joinpath(steps_dir, jld_frames[1]))
GLMakie.mesh(step1.body_mesh; color=:white) # precompile mesh recipe before timing

dmesh = step1.body_mesh
# body_margin=10 creates a 10-voxel zone around the dolphin where vorticity is
# forced to zero (no smoke), then a 2-voxel ramp before full field kicks in.
# This prevents bright shear-layer smoke from occluding the body along camera
# rays, without needing to bound the medium to a subregion.
vc = compute_vorticity(step1, 10f0; transition=2f0)

# Camera scaled from dolphin.jl's cam3 (L=192 domain) to this L=256 sim.
sx, sy, sz = size(step1.u)[1:3]
scale = Float32(sx / 96)
body_centre = let v = dmesh.position
    Vec3f(mean(p[1] for p in v), mean(p[2] for p in v), mean(p[3] for p in v))
end
cam = (eye    = Vec3f(843.27, -153.6, 238.31) .* scale,
       lookat = body_centre + Vec3f(0, 30, 0),
       up     = Vec3f(-0.17, 0.06, 0.98),
       fov    = 10.0)

# Bright sun-sky illuminates the smoke (scattering is now the main effect).
# A warm key light + cool fill give the smoke its volumetric shading.
sun_dir = normalize(Vec3f(-0.4, -0.15, 0.9))
sunsky = Makie.SunSkyLight(sun_dir;
    intensity=1.5f0, turbidity=3.0f0,
    ground_albedo=RGBf(0.10, 0.12, 0.15), ground_enabled=false)
bcx = Float32(sx / 2); bcy = Float32(sy / 2); bcz = Float32(sz / 2)
lights = [
    sunsky,
    # Warm key — catches the upper side of the smoke, shadows on the underside.
    Makie.PointLight(RGBf(18000, 14000, 9000), Vec3f(bcx - 40, bcy - 80, bcz + 80) .* scale),
    # Cool fill — prevents the shadow side from going pure black.
    Makie.PointLight(RGBf(1500, 2500, 4500),   Vec3f(bcx + 60, bcy + 40, bcz - 40) .* scale),
]

fig = Figure(size=(res_w, res_h);
             backgroundcolor=RGBf(0.0, 0.0, 0.0), figure_padding=0)
ax = LScene(fig[1, 1]; show_axis=false,
            scenekw=(lights=lights, backgroundcolor=RGBf(0.0, 0.0, 0.0)))
plots = panel_rainbow_glow!(ax, dmesh, vc, cam;
    emission_scale=0.5f0, extinction_intensity=0.02f0)

RayMakie.activate!(; tonemap=:aces, gamma=2.2f0, exposure=1f0, denoise=false)
integrator = Hikari.VolPath(; samples=500, max_depth=12, hw_accel=hw_accel)
println("[$(Dates.now())] integrator: hw_accel=$hw_accel  max_depth=$max_depth")
colorbuffer(fig; integrator=integrator)

# ---- record ----
Makie.record_longrunning(fig, video_out, 1:n_to_render; framerate=30, update=false, backend=RayMakie, integrator=integrator) do i
    i == 1 && return nothing
    step = load_smoke_step(joinpath(steps_dir, jld_frames[i]))
    new_vc = compute_vorticity(step, 10f0; transition=2f0)
    Makie.update!(plots.dolphin; arg1=step.body_mesh)
    update_rainbow_glow!(plots, new_vc;
        emission_scale=0.5f0, extinction_intensity=0.02f0)
end
