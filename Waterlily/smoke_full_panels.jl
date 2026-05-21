# Render ONE frame of the complete 4-panel construct_scene at low spp.
# Used to answer: does the blue (top-right) panel look the same as my
# standalone single-panel render with cam2 defaults? If so, my standalone
# is correct and the reference user sees is just what cam2 produces. If
# not, something about the single-panel extract differs from the full
# panel context and needs to be fixed.
using Pkg; Pkg.activate("/sim/Programmieren/VulkanDev")
include("/sim/Programmieren/VulkanDev/RayDemo/Waterlily/dolphin.jl")
using Lava, FileIO

const OUT_PATH = joinpath(@__DIR__, "full4_single.png")
const STEP = 100
const SPP  = 10
const PANEL_SIZE = (960, 540)

integrator = Hikari.VolPath(; samples=SPP, max_depth=8, hw_accel=true)
RayMakie.activate!(; integrator, tonemap=:aces, gamma=2.2f0, exposure=2.0f0, denoise=false)

data = load_step(STEP)
fig, plots, sd = construct_scene(; panel_size=PANEL_SIZE, data=data)
println("rendering 4-panel at spp=$SPP …")
t0 = time()
img = Makie.colorbuffer(fig; update=false)
println("  render: $(round(time()-t0, digits=1)) s  size=$(size(img))")
save(OUT_PATH, img)
println("  saved $OUT_PATH ($(filesize(OUT_PATH) ÷ 1024) KiB)")

# Also crop + save just the blue (top-right) panel subregion so we can
# compare pixel-for-pixel with the standalone smoke_single.png.
# Makie's colorbuffer returns (H, W) matrix with row 1 at the TOP — so the
# top-right quadrant is rows 1..H/2, cols W/2+1..W.
H, W = size(img)
top_right = img[1:H÷2, W÷2+1:end]
save(joinpath(@__DIR__, "full4_blue_crop.png"), top_right)
println("  saved blue-panel crop ($(filesize(joinpath(@__DIR__, \"full4_blue_crop.png\")) ÷ 1024) KiB)")
