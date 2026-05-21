using WaterLily, StaticArrays, BiotSavartBCs, WaterLilyMeshBodies
function dolphin(; L=64, Re=1e6, U=1, A=0.1, St=0.3, k=5.3, mem=Array, T=Float32)
    # Get bounding box, scale and center
    mesh_path = joinpath(@__DIR__, "low_poly_dolphin.stl")
    probe = MeshBody(mesh_path)
    lo, up = probe.bvh.nodes[1].lo, probe.bvh.nodes[1].up
    scale = T(L / maximum(up .- lo))
    center = scale * SVector{3}(lo .+ up) / 2

    # undulating mapping
    a = T(L * A)
    ω = T(π * St * U / a)
    k = T(k)  # motion parameters
    @inline s(x) = clamp((x[2] + L ÷ 2) / L, 0, 1) # normalized lengthwise coordinate
    @inline amp(x) = (1 + 9 * s(x)^3) / 10       # amplitude envelope function
    function map(x, t)
        x -= SA[L÷4, L÷2+5, L÷4] - center
        x + a * amp(x) * sin(k * s(x) - ω * t) * SA[0, 0, 1]
    end

    # create simulation with mesh body
    size = (L ÷ 2, 3L ÷ 2, L ÷ 2)
    BiotSimulation(size, (0, U, 0), L; ν=U * L / Re, mem, T,
        body=MeshBody(mesh_path; scale, map, boundary=true, mem, size))
end

using GLMakie, Meshing, Lava

Makie.inline!(false)

dolphin_sim = dolphin(L=256, mem=Lava.LavaArray);
sim_step!(dolphin_sim, 1; verbose=true)
viz!(dolphin_sim, body2mesh=true, body_color=:snow2,
    azimuth=-0.5, fig_size=(1280, 720), hidedecorations=true,
    duration=2, step=0.01, video="dolphin.mp4",
    colormap=:ocean, colorrange=(0.15, 0.5), algorithm=:mip)
