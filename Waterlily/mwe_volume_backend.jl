# MWE: Lava vs ROCBackend volume rendering
#
# Isolates the volume-rendering path to compare Hikari.VolPath on
# Lava.LavaBackend (HW RT) vs AMDGPU.ROCBackend (SW BVH) with the same
# density field, lights, camera, and material parameters that panel_blue_volume
# and panel_warm_hero in dolpin2.jl use.
#
# Writes: mwe_volume_lava.png, mwe_volume_rocm.png

using Lava, AMDGPU
using GLMakie, RayMakie, Hikari
using GeometryBasics, FileIO, Colors

# --- Synthetic density field -------------------------------------------------
# Elongated gaussian blob — stand-in for the vorticity volume. Values are on the
# same scale (~0–0.15) as the real `vc` after sqrt+clamp in compute_vorticity.
nx, ny, nz = 96, 200, 96
vc = Array{Float32}(undef, nx, ny, nz)
cx, cy, cz = nx/2, ny/3, nz/2
@inbounds for k in 1:nz, j in 1:ny, i in 1:nx
    r2 = ((i-cx)/30f0)^2 + ((j-cy)/60f0)^2 + ((k-cz)/30f0)^2
    # add a wake-like tail in +y
    tail = max(0f0, 1f0 - ((i-cx)^2 + (k-cz)^2) / 500f0) * max(0f0, (j-cy)/ny) * 0.08f0
    vc[i,j,k] = clamp(0.13f0 * exp(-r2) + tail, 0f0, 0.15f0)
end
println("vc extrema: ", extrema(vc))

# --- Scene builder (same lights / material / integrator as panel_blue_volume) ---
function build_scene()
    bcx, bcy, bcz = Float32(cx), Float32(cy), Float32(cz)
    s = Scene(size=(640, 480);
        lights=[
            Makie.PointLight(RGBf(8000,12000,18000), Vec3f(bcx-40,bcy-80,bcz+50)),
            Makie.PointLight(RGBf(2000,3000,5000),   Vec3f(bcx+50,bcy+30,bcz+30)),
            Makie.PointLight(RGBf(1000,1500,2500),   Vec3f(bcx,bcy-20,bcz-30)),
        ],
        backgroundcolor=RGBf(0.02, 0.04, 0.12))
    cam3d!(s)

    # placeholder body (single sphere) so the scene has a non-volume surface
    mesh!(s, Sphere(Point3f(cx, cy-40, cz), 8f0);
        material=Hikari.CoatedDiffuse(reflectance=(0.30f0,0.35f0,0.45f0),
                                      roughness=0.02f0, eta=1.5f0))

    volume!(s, 1f0..Float32(nx), 1f0..Float32(ny), 1f0..Float32(nz), vc;
        colormap=[RGBA(0,0,0,0),              RGBA(0,0,0,0),
                  RGBA(0.01,0.02,0.08,0.005), RGBA(0.03,0.08,0.25,0.015),
                  RGBA(0.08,0.2,0.55,0.04),   RGBA(0.15,0.38,0.8,0.08),
                  RGBA(0.25,0.55,0.92,0.16),  RGBA(0.4,0.72,0.98,0.3),
                  RGBA(0.6,0.88,1.0,0.48)],
        colorrange=(0.015f0, 0.13f0),
        material=(extinction_scale=14f0, asymmetry_g=0.6f0, single_scatter_albedo=0.3f0))

    update_cam!(s,
        Vec3f(bcx+90, bcy-55, bcz+35),      # eye
        Vec3f(bcx-5,  bcy+20, bcz-3),       # lookat
        Vec3f(0, 0, 1))                     # up
    return s
end

# --- Render helper -----------------------------------------------------------
function render_with(device, tag; spp=128)
    hw = device isa Lava.LavaBackend
    s = build_scene()
    println("Rendering with $(tag) (hw_accel=$hw)...")
    img = @time colorbuffer(s;
        device=device,
        integrator=Hikari.VolPath(; samples=spp, max_depth=10, hw_accel=hw),
        tonemap=:aces, gamma=2.2f0, exposure=2.2f0)
    out = joinpath(@__DIR__, "mwe_volume_$(tag).png")
    save(out, img)
    println("  wrote $out")
    return img
end

RayMakie.activate!()
img_lava = render_with(Lava.LavaBackend(),   "lava"; spp=128)
img_rocm = render_with(AMDGPU.ROCBackend(), "rocm"; spp=128)

# Side-by-side
h, w = size(img_lava)
combined = fill(RGBf(0.05, 0.05, 0.05), h, 2w + 4)
combined[:, 1:w]            .= img_lava
combined[:, w+5:2w+4]       .= img_rocm
save(joinpath(@__DIR__, "mwe_volume_compare.png"), combined)
println("compare: mwe_volume_compare.png")
