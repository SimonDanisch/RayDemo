# Serialize the L=256 dolphin sim state per-frame. Mirrors mooving-dolphin.jl:
# L=256, sim_step!(sim, 1) warmup, then 200 frames at dt_sim=0.01 from t=1 to t=3.
# Per-frame JLD2 holds velocity field, body SDF, vorticity magnitude, time.
# Resumes: existing frame files are skipped (sim still steps through them).
#
# Usage:
#   julia --project=/sim/Programmieren/VulkanDev RayDemo/Waterlily/generate_dolphin_smoke.jl
using Dates
println("[$(Dates.now())] starting")

using WaterLily, StaticArrays, BiotSavartBCs, WaterLilyMeshBodies, Lava
using JLD2

# ── sim setup ───────────────────────────────────────────────────────────────
# Mirrors mooving-dolphin.jl exactly: L=256, sim_step!(sim,1) warmup, then viz!(duration=2, step=0.01)
# → 200 frames spanning sim_time=1→3, dt=0.01 each.
const L_SIM   = 256
const WARMUP  = 1.0f0
const N_FRAMES = 201   # range(1, 3; step=0.01) is 201 values (inclusive both ends)
const DT_SIM_PER_FRAME = 0.01f0
const OUTDIR = joinpath(@__DIR__, "dolphin_smoke_L$(L_SIM)_steps")
isdir(OUTDIR) || mkpath(OUTDIR)

function dolphin_sim(; L=L_SIM, Re=1e6, U=1, A=0.1, St=0.3, k=5.3, mem=Lava.LavaArray, T=Float32)
    mesh_path = joinpath(@__DIR__, "low_poly_dolphin.stl")
    probe = MeshBody(mesh_path)
    lo, up = probe.bvh.nodes[1].lo, probe.bvh.nodes[1].up
    scale = T(L / maximum(up .- lo))
    center = scale * SVector{3}(lo .+ up) / 2
    a = T(L * A); ω = T(π * St * U / a); k = T(k)
    @inline s(x) = clamp((x[2] + L ÷ 2) / L, 0, 1)
    @inline amp(x) = (1 + 9 * s(x)^3) / 10
    function map(x, t)
        x -= SA[L÷4, L÷2+5, L÷4] - center
        x + a * amp(x) * sin(k * s(x) - ω * t) * SA[0, 0, 1]
    end
    sz = (L ÷ 2, 3L ÷ 2, L ÷ 2)
    BiotSimulation(sz, (0, U, 0), L; ν=U * L / Re, mem, T,
        body=MeshBody(mesh_path; scale, map, boundary=true, mem, size=sz))
end

println("[$(Dates.now())] building BiotSimulation L=$L_SIM …")
sim = dolphin_sim()
println("[$(Dates.now())] domain=$(size(sim.flow.σ))  stepping to warmup=$WARMUP …")
t_warm = @elapsed sim_step!(sim, WARMUP; verbose=false)
println("[$(Dates.now())] warmup done in $(round(t_warm,digits=1))s")

# ── dump N_FRAMES frames ────────────────────────────────────────────────────
function snapshot!(sim)
    # Body SDF first (uses σ), then vorticity (also uses σ).
    WaterLily.measure_sdf!(sim.flow.σ, sim.body, WaterLily.time(sim))
    sdf = Array(sim.flow.σ[WaterLily.inside(sim.flow.σ)])
    WaterLily.@inside sim.flow.σ[I] = WaterLily.ω_mag(I, sim.flow.u)
    vort = Array(sim.flow.σ[WaterLily.inside(sim.flow.σ)])
    u_cpu = Array(sim.flow.u)
    return (u=u_cpu, sdf=sdf, vort=vort,
            t=Float32(WaterLily.time(sim)),
            sim_t=Float32(WaterLily.sim_time(sim)))
end

target = Float64(WARMUP)
t_start = time()
for f in 1:N_FRAMES
    if f > 1
        target += Float64(DT_SIM_PER_FRAME)
        sim_step!(sim, Float32(target); verbose=false)
    end
    path = joinpath(OUTDIR, "frame_$(lpad(f,4,'0')).jld2")
    if isfile(path)
        println("[$(Dates.now())] frame $f/$N_FRAMES already on disk, skipping")
        continue
    end
    snap = snapshot!(sim)
    jldsave(path; snap.u, snap.sdf, snap.vort, snap.t, snap.sim_t, L=L_SIM)
    elapsed = time() - t_start
    eta = elapsed / f * (N_FRAMES - f)
    println("[$(Dates.now())] frame $f/$N_FRAMES  sim_t=$(round(snap.sim_t,digits=2))  phys_t=$(round(snap.t,digits=1))  elapsed=$(round(elapsed/60,digits=1))min  eta=$(round(eta/60,digits=1))min")
end
println("[$(Dates.now())] DONE. Files in $OUTDIR")
