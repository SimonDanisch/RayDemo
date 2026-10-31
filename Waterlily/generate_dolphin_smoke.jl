# Serialize the L=256 dolphin sim state per-frame. Mirrors mooving-dolphin.jl:
# L=256, sim_step!(sim, 1) warmup, then 200 frames at dt_sim=0.01 from t=1 to t=3.
# Per-frame JLD2 holds velocity field, body SDF, vorticity magnitude, time.
# Resumes: existing frame files are skipped (sim still steps through them).
#
# Usage:
#   julia --project=/sim/Programmieren/VulkanDev RayDemo/Waterlily/generate_dolphin_smoke.jl
using Dates
println("[$(Dates.now())] starting")

using WaterLily, StaticArrays, BiotSavartBCs, WaterLilyMeshBodies, Lava, CUDA
using JLD2
import KernelAbstractions as K

# ── sim setup ───────────────────────────────────────────────────────────────
# Mirrors mooving-dolphin.jl exactly: L=256, sim_step!(sim,1) warmup, then viz!(duration=2, step=0.01)
# → 200 frames spanning sim_time=1→3, dt=0.01 each.
const L_SIM   = 256
const WARMUP  = 0.001f0
const N_FRAMES = 201   # range(1, 3; step=0.01) is 201 values (inclusive both ends)
const DT_SIM_PER_FRAME = 0.01f0
mesh_path = joinpath(@__DIR__, "low_poly_dolphin.stl")


function dolphin_sim(; L=L_SIM, Re=1e6, U=1, A=0.1, St=0.3, k=5.3, mem=Lava.LavaArray, T=Float32)
    mesh_path = "/sim/Programmieren/VulkanDev/RayDemo/Waterlily/low_poly_dolphin.stl"
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
dev = Lava.LavaBackend()
println("[$(Dates.now())] building BiotSimulation L=$L_SIM …")
sim = dolphin_sim(mem=Lava.LavaArray, T=Float32);
println("[$(Dates.now())] domain=$(size(sim.flow.σ))  stepping to warmup=$WARMUP …")
t_warm = @elapsed sim_step!(sim, WARMUP; verbose=false, max_steps=1)
println("[$(Dates.now())] warmup done in $(round(t_warm,digits=1))s")

begin
    target = Float64(WARMUP)
    t_start = time()
    for f in 1:N_FRAMES
        target += DT_SIM_PER_FRAME
        if f > 1
            @time begin
                sim_step!(sim, Float32(target); max_steps=1, verbose=false)
                K.synchronize(dev)
            end
        end
        elapsed = time() - t_start
        eta = elapsed / f * (N_FRAMES - f)
        println("[$(Dates.now())] frame $f/$N_FRAMES  t=$(round(target,digits=3))  elapsed=$(round(elapsed,digits=1))s  eta=$(round(eta,digits=1))s")
    end
end

func1(a, b) = a + b
func1(a::Int, b::Float32) = "other implementation"
func1(a::Float64, b::Float64) = "other implementation"

push!(::Array, x::Int)
push!(::MyArray, x::Int)

function myfunc(x::AnyArray)
    push!(x, 1)
end
