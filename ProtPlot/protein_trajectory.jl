include("../common/common.jl")
using ProtPlot

basedir = joinpath(@__DIR__, "Trajectories")

function render_video(;
    device=DEVICE,
    resolution=nothing,
    samples=100,
    max_depth=5,
    nframes=nothing,
    output_path=joinpath(@__DIR__, "protein_trajectory.mp4"),
)
    i = 15
    integrator = Hikari.VolPath(samples=samples, max_depth=max_depth)
    sensor = Hikari.PixelSensor(iso=100, whitebalance=6500)
    integrator.sensor = sensor
    dir = "samp_$(string(i, pad=5))"
    tracks = reverse([r for r in readdir(joinpath(basedir, dir), join=true) if r[end] != 'e'])
    set_theme!(lights=[Makie.SunSkyLight(Vec3f(0.4, -0.3, 0.7); intensity=1.0f0, turbidity=3.0f0, ground_enabled=false)])
    @time ProtPlot.animate_molecule_dir(
        output_path,
        tracks
    )
    @info "Saved → $output_path"
end

# render_video()
