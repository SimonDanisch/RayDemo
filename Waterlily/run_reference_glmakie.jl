using Pkg; Pkg.activate("/sim/Programmieren/VulkanDev")

using WaterLily, StaticArrays, BiotSavartBCs, WaterLilyMeshBodies

function dolphin(;L=64, Re=1e6, U=1, A=0.1, St=0.3, k=5.3, mem=Array, T=Float32)
    mesh_path = joinpath(dirname(pathof(WaterLilyMeshBodies)), "..", "example", "LowPolyDolphin.stl")
    probe = MeshBody(mesh_path)
    lo, up = probe.bvh.nodes[1].lo, probe.bvh.nodes[1].up
    scale = T(L / maximum(up .- lo))
    center = scale * SVector{3}(lo .+ up) / 2

    a = T(L * A); ω = T(π * St * U / a); k = T(k)
    @inline s(x) = clamp((x[2] + L÷2) / L, 0, 1)
    @inline amp(x) = (1 + 9 * s(x)^3) / 10
    function map(x, t)
        x -= SA[L÷4, L÷2+5, L÷4] - center
        x + a * amp(x) * sin(k * s(x) - ω * t) * SA[0, 0, 1]
    end

    size = (L÷2, 3L÷2, L÷2)
    BiotSimulation(size, (0, U, 0), L; ν = U * L / Re, mem, T,
        body = MeshBody(mesh_path; scale, map, boundary=true, mem, size))
end

using GLMakie, Meshing, AMDGPU
Makie.inline!(false)
println("GLMakie active: ", Makie.current_backend())
println("AMDGPU device: ", AMDGPU.device())

dolphin_sim = dolphin(L=256, mem=AMDGPU.ROCArray)
println("sim built, warmup to t=1 ...")
t0 = time()
sim_step!(dolphin_sim, 1; verbose=true)
println("warmup done in $(round(time()-t0, digits=1))s")

println("running viz! with duration=2, step=0.01, framerate=60 → dolphin_glmakie.mp4")
t0 = time()
viz!(dolphin_sim, body2mesh=true, body_color=:snow2,
    azimuth=-0.5, fig_size=(1280, 720), hidedecorations=true,
    duration=2, step=0.01, video=joinpath(@__DIR__, "dolphin_glmakie.mp4"),
    colormap=:ocean, colorrange=(0.15, 0.5), algorithm=:mip)
println("viz! done in $(round(time()-t0, digits=1))s")
