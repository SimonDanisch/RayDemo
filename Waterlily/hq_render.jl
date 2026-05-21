using Pkg
Pkg.activate("/sim/Programmieren/VulkanDev")
using Dates

include("dolphin.jl")

println("=" ^ 60)
println("HQ video render — ", Dates.now())
println("=" ^ 60)
flush(stdout)

# HQ: 450 frames, 100 spp, 960×540 panels, max_depth=8
# `hq_video` already runs GC.gc(true) per frame to bound RAM+VRAM growth.
# Makie.record_longrunning writes frames into `_frames/` alongside the mp4,
# so the run is resumable on crash.

outpath = joinpath(@__DIR__, "dolphin_hq.mp4")
t0 = time()
try
    hq_video(; n_frames=450, spp=100, panel_size=(960, 540), outpath)
    elapsed = time() - t0
    println("=" ^ 60)
    println("DONE in ", round(elapsed, digits=1), " s = ", round(elapsed/60, digits=1), " min")
    println("Output: ", outpath)
    println("=" ^ 60)
catch e
    elapsed = time() - t0
    println("=" ^ 60)
    println("FAILED after ", round(elapsed, digits=1), " s")
    println("Error: ", sprint(showerror, e, catch_backtrace()))
    println("=" ^ 60)
    rethrow()
end
