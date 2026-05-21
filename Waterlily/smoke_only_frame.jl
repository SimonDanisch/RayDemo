# One-off: render a single frame of just the WARM (yellow) volume panel.
# Cam3 + lights_warm kept EXACTLY as in construct_scene — no tweaks.
using Pkg; Pkg.activate("/sim/Programmieren/VulkanDev")
include("/sim/Programmieren/VulkanDev/RayDemo/Waterlily/dolphin.jl")
using Lava, FileIO

const OUT_PATH = joinpath(@__DIR__, "smoke_single.png")
const STEP = 100
const SPP  = 10
const PANEL_SIZE = (960, 540)

integrator = Hikari.VolPath(; samples=SPP, max_depth=8, hw_accel=true)
RayMakie.activate!(; integrator, tonemap=:aces, gamma=2.2f0, exposure=2.0f0, denoise=false)

data = load_step(STEP)
dmesh = dolphin_body_mesh(data)
vc = compute_vorticity(data)
sx, sy, sz, _ = size(data.u)
bcx = Float32(sx / 2); bcy = Float32(sy / 2); bcz = Float32(sz / 2)

# cam3 + lights_warm copied verbatim from construct_scene.
cam = (eye=Vec3f(843.27, -153.6, 238.31), lookat=Vec3f(41.73, 138.05, 79.23),
       up=Vec3f(-0.17, 0.06, 0.98), fov=10.0)

lights_warm = [
    Makie.PointLight(RGBf(16000,12000,7000), Vec3f(bcx-50, bcy-90, bcz+60)),
    Makie.PointLight(RGBf(1500,2500,4000),   Vec3f(bcx+60, bcy+40, bcz+25)),
    Makie.PointLight(RGBf(1000,900,500),     Vec3f(bcx,    bcy+100, bcz+50)),
    Makie.PointLight(RGBf(1000,1200,2000),   Vec3f(bcx,    bcy-20, bcz-30)),
]

fig = Figure(size=PANEL_SIZE; backgroundcolor=RGBf(0f0, 0f0, 0f0), figure_padding=0)
ax  = LScene(fig[1, 1]; show_axis=false,
             scenekw=(lights=lights_warm, backgroundcolor=RGBf(0.012, 0.02, 0.05)))
p = panel_warm_hero!(ax, dmesh, vc, cam)

println("rendering at spp=$SPP…")
t0 = time()
img = Makie.colorbuffer(fig; update=false)
println("  render: $(round(time()-t0, digits=1)) s  size=$(size(img))")
save(OUT_PATH, img)
println("  saved $OUT_PATH ($(filesize(OUT_PATH) ÷ 1024) KiB)")
