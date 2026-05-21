# sim_regen_compare.jl — re-run the dolphin WaterLily simulation on Lava and
# compare numerically against the cached `dolphin_smoke_L256_steps/` frames.
#
# This stresses Lava under a totally different workload than rendering: the
# WaterLily pressure-Poisson solver + advection runs many compute kernels
# (KA.@kernel, AcceleratedKernels reductions, BiotSavart BCs).  If anything
# drifts vs. the cached reference (or crashes) we know it's a Lava-side
# regression in the compute kernel path.
#
# Compares per frame:
#   * `u`     — full velocity field (4D Float32, ghost cells included)
#   * `sdf`   — body signed-distance function (3D Float32)
#   * `vort`  — vorticity magnitude (3D Float32)
#
# Uses the same sim setup as `generate_dolphin_smoke.jl`: L=256, warmup=1.0,
# 200 frames at dt=0.01.  Comparison is element-wise: max abs diff + mean
# abs diff per frame, with a tolerance of `atol`.

using Dates
println("[$(Dates.now())] starting sim regen + cache compare")

using WaterLily, StaticArrays, BiotSavartBCs, WaterLilyMeshBodies, Lava
using JLD2, Statistics

const L_SIM    = 256
const WARMUP   = 1.0f0
const DT_PER_F = 0.01f0
const CACHE_DIR = joinpath(@__DIR__, "dolphin_smoke_L$(L_SIM)_steps")

# Same dolphin_sim helper as generate_dolphin_smoke.jl (mirrored here so this
# script is self-contained).
function dolphin_sim(; L=L_SIM, Re=1e6, U=1, A=0.1, St=0.3, k=5.3,
                       mem=Lava.LavaArray, T=Float32)
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

function snapshot!(sim)
    WaterLily.measure_sdf!(sim.flow.σ, sim.body, WaterLily.time(sim))
    sdf = Array(sim.flow.σ[WaterLily.inside(sim.flow.σ)])
    WaterLily.@inside sim.flow.σ[I] = WaterLily.ω_mag(I, sim.flow.u)
    vort = Array(sim.flow.σ[WaterLily.inside(sim.flow.σ)])
    u_cpu = Array(sim.flow.u)
    return (; u=u_cpu, sdf, vort,
              t=Float32(WaterLily.time(sim)),
              sim_t=Float32(WaterLily.sim_time(sim)))
end

function compare_arr(name, a, b)
    @assert size(a) == size(b) "$name size mismatch: $(size(a)) vs $(size(b))"
    da = Float64.(abs.(a .- b))
    return (max=maximum(da), mean=mean(da))
end

function regen_and_compare(; n_frames::Int=200, atol::Float32=1f-4,
                              first_check::Int=1)
    @assert isdir(CACHE_DIR) "missing cache dir $CACHE_DIR"

    println("[$(Dates.now())] building BiotSimulation L=$L_SIM …")
    sim = dolphin_sim()
    println("[$(Dates.now())] warmup to t=$WARMUP …")
    t_warm = @elapsed sim_step!(sim, WARMUP; verbose=false)
    println("[$(Dates.now())] warmup done in $(round(t_warm,digits=1))s")

    target = Float64(WARMUP)
    diffs = NamedTuple[]
    failed_frames = Int[]
    t_start = time()
    for f in 1:n_frames
        if f > 1
            target += Float64(DT_PER_F)
            sim_step!(sim, Float32(target); verbose=false)
        end
        f < first_check && continue
        snap = snapshot!(sim)
        ref_path = joinpath(CACHE_DIR, "frame_$(lpad(f, 4, '0')).jld2")
        isfile(ref_path) || (println("  frame $f: no cache file, skipping"); continue)
        ref = jldopen(ref_path, "r") do file
            (u=copy(file["u"]), sdf=copy(file["sdf"]), vort=copy(file["vort"]),
             t=Float32(file["t"]), sim_t=Float32(file["sim_t"]))
        end
        du = compare_arr("u",    snap.u,    ref.u)
        ds = compare_arr("sdf",  snap.sdf,  ref.sdf)
        dv = compare_arr("vort", snap.vort, ref.vort)
        push!(diffs, (; f, sim_t=snap.sim_t, du, ds, dv))
        ok = du.max <= atol && ds.max <= atol && dv.max <= atol
        ok || push!(failed_frames, f)
        if f % 20 == 0 || !ok
            elapsed = time() - t_start
            println("  frame $f sim_t=$(round(snap.sim_t,digits=2))  ",
                    "u(max=$(round(du.max,sigdigits=3)) mean=$(round(du.mean,sigdigits=3)))  ",
                    "sdf(max=$(round(ds.max,sigdigits=3)) mean=$(round(ds.mean,sigdigits=3)))  ",
                    "vort(max=$(round(dv.max,sigdigits=3)) mean=$(round(dv.mean,sigdigits=3)))  ",
                    ok ? "OK" : "FAIL",
                    "  elapsed=$(round(elapsed/60, digits=1))min")
        end
    end

    println("\n[$(Dates.now())] regen+compare done.")
    println("  frames compared: $(length(diffs))")
    if !isempty(diffs)
        ums = [d.du.max  for d in diffs]; sm = [d.ds.max for d in diffs]; vm = [d.dv.max for d in diffs]
        println("  u    max-of-max  $(round(maximum(ums),sigdigits=3))   median $(round(sort(ums)[end÷2+1],sigdigits=3))")
        println("  sdf  max-of-max  $(round(maximum(sm),sigdigits=3))   median $(round(sort(sm)[end÷2+1],sigdigits=3))")
        println("  vort max-of-max  $(round(maximum(vm),sigdigits=3))   median $(round(sort(vm)[end÷2+1],sigdigits=3))")
    end
    if isempty(failed_frames)
        println("  ✓ ALL $(length(diffs)) frames within atol=$atol")
    else
        println("  ✗ $(length(failed_frames)) frames > atol=$atol: first 5 = $(failed_frames[1:min(5,end)])")
    end
    return (; diffs, failed_frames)
end
