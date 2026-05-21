# Full warm-volume video — 450 frames, spp=500, 960×540, cam3 verbatim.
# Resumable: Makie.record_longrunning writes frame_NNNN.png next to the mp4
# and skips existing ones; a first_missing scan avoids re-running
# update_scene! for cached frames on restarts.
using Pkg; Pkg.activate("/sim/Programmieren/VulkanDev")
include("/sim/Programmieren/VulkanDev/RayDemo/Waterlily/dolphin.jl")
using Lava, FileIO, Dates

const OUT_PATH   = joinpath(@__DIR__, "smoke_warm_hq.mp4")
const N_FRAMES   = 450
const SPP        = 500
const PANEL_SIZE = (960, 540)

println("=" ^ 60)
println("WARM-volume HQ video — ", Dates.now())
println("=" ^ 60)
flush(stdout)

integrator = Hikari.VolPath(; samples=SPP, max_depth=8, hw_accel=true)
RayMakie.activate!(; integrator, tonemap=:aces, gamma=2.2f0, exposure=2.0f0, denoise=false)

step1 = load_step(1)
dmesh_init  = dolphin_body_mesh(step1)
vc_init = compute_vorticity(step1)
sx, sy, sz, _ = size(step1.u)
bcx = Float32(sx / 2); bcy = Float32(sy / 2); bcz = Float32(sz / 2)

cam = (eye=Vec3f(843.27, -153.6, 238.31), lookat=Vec3f(41.73, 138.05, 79.23),
       up=Vec3f(-0.17, 0.06, 0.98), fov=10.0)

lights_warm = [
    Makie.PointLight(RGBf(16000,12000,7000), Vec3f(bcx-50, bcy-90, bcz+60)),
    Makie.PointLight(RGBf(1500,2500,4000),   Vec3f(bcx+60, bcy+40, bcz+25)),
    Makie.PointLight(RGBf(1000,900,500),     Vec3f(bcx,    bcy+100, bcz+50)),
    Makie.PointLight(RGBf(1000,1200,2000),   Vec3f(bcx,    bcy-20, bcz-30)),
]

fig = Figure(size=PANEL_SIZE; backgroundcolor=RGBf(0f0,0f0,0f0), figure_padding=0)
ax  = LScene(fig[1, 1]; show_axis=false,
             scenekw=(lights=lights_warm, backgroundcolor=RGBf(0.012, 0.02, 0.05)))
p = panel_warm_hero!(ax, dmesh_init, vc_init, cam)

# Resume support: skip update_scene!-equivalent work for frames already on
# disk (record_longrunning still iterates but also skips the colorbuffer).
frame_dir = joinpath(@__DIR__, "smoke_warm_hq_frames")
nd = max(4, length(string(N_FRAMES)))
first_missing = let m = N_FRAMES + 1
    isdir(frame_dir) || mkpath(frame_dir)
    for i in 1:N_FRAMES
        if !isfile(joinpath(frame_dir, "frame_$(lpad(i, nd, '0')).png"))
            m = i; break
        end
    end
    m
end
println("resume-aware: first_missing = $first_missing")
flush(stdout)

t0 = time()
try
    Makie.record_longrunning(fig, OUT_PATH, 1:N_FRAMES; framerate=30, update=false) do i
        i < first_missing && return
        step = load_step(i)
        Makie.update!(p.dolphin; arg1=dolphin_body_mesh(step))
        new_vc = compute_vorticity(step)
        Makie.update!(p.vol; arg4=new_vc)
        GC.gc(true)
        gpu = Lava.GPU_LIVE_BYTES[] >> 20
        println("frame $i/$N_FRAMES  gpu=$(gpu)MiB  t=$(round(time()-t0, digits=1))s")
        flush(stdout)
    end
    println("=" ^ 60)
    println("DONE in ", round(time()-t0, digits=1), " s = ", round((time()-t0)/60, digits=1), " min")
    println("Output: ", OUT_PATH)
    println("=" ^ 60)
catch e
    println("=" ^ 60)
    println("FAILED after ", round(time()-t0, digits=1), " s")
    println("Error: ", sprint(showerror, e, catch_backtrace()))
    println("=" ^ 60)
    rethrow()
end
