# Reference render of the same JLD2 sim frames via GLMakie, to check whether
# the saved SDFs actually carry the per-frame body deformation (decoupled from
# any RayMakie/Hikari mesh-update peculiarity).
#
# Uses Observable-based mesh/volume updates — exactly viz!'s pattern.
using GeometryBasics, Meshing, JLD2, FileIO, Colors
using Makie, GLMakie

function sdf_to_mesh(sdf::Array{T,3}) where T
    ranges = range.((0, 0, 0), size(sdf))
    points, faces = Meshing.isosurface(sdf, Meshing.MarchingCubes(iso=0), ranges...)
    GeometryBasics.Mesh(Point3.(points), GLTriangleFace.(faces))
end

function main()
    res_w   = length(ARGS) ≥ 1 ? parse(Int, ARGS[1]) : 480
    res_h   = length(ARGS) ≥ 2 ? parse(Int, ARGS[2]) : 270
    nframes = length(ARGS) ≥ 3 ? parse(Int, ARGS[3]) : typemax(Int)

    steps_dir = joinpath(@__DIR__, "dolphin_smoke_L256_steps")
    video_out = joinpath(@__DIR__, "dolphin_smoke_glmakie_$(res_w)x$(res_h).mp4")
    jld_frames = sort(filter(startswith("frame_"), readdir(steps_dir)))
    n_to_render = min(nframes, length(jld_frames))
    println("GLMakie render: $n_to_render frames @ $(res_w)x$(res_h)")

    first_state = load(joinpath(steps_dir, jld_frames[1]))
    σ_obs    = Observable(first_state["vort"]::Array{Float32,3})
    body_obs = Observable(sdf_to_mesh(first_state["sdf"]::Array{Float32,3}))

    fig = Figure(size=(res_w, res_h))
    ax = Axis3(fig[1,1]; aspect=:data, azimuth=-0.5, elevation=π/8)
    hidedecorations!(ax)
    ax.xspinesvisible = false; ax.yspinesvisible = false; ax.zspinesvisible = false

    volume!(ax, σ_obs; colormap=:ocean, colorrange=(0.15, 0.5),
            algorithm=:mip)
    mesh!(ax, body_obs; color=:snow2)

    GLMakie.activate!()
    Makie.record(fig, video_out, 1:n_to_render; framerate=30) do i
        state = load(joinpath(steps_dir, jld_frames[i]))
        σ_obs[]    = state["vort"]::Array{Float32,3}
        body_obs[] = sdf_to_mesh(state["sdf"]::Array{Float32,3})
    end
    println("DONE → $video_out")
end

main()
