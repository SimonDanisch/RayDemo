include("../common/common.jl")
using GeometryBasics, PlantGeom, XPalm, XPalm.VPalm
import PlantGeom: symbol, descendants
import Hikari.TextureNodes as TN

leaf_material = Hikari.Principled(base_color = (0.14, 0.34, 0.04), roughness = 0.22, subsurface_weight = 0.6, thin_wall = true)
stem_material = Hikari.CoatedDiffuse(reflectance = (0.55f0, 0.35f0, 0.2f0), roughness = 0.4f0, ior = 1.5f0, thickness = 0.001f0)

file = joinpath(dirname(dirname(pathof(XPalm))), "test", "references", "vpalm-parameter_file.yml")
mtg = build_mockup(read_parameters(file); merge_scale = :leaflet)

PlantGeom.traverse!(mtg) do node
    if symbol(node) == :Petiole
        segs = descendants(node, symbol = [:PetioleSegment, :RachisSegment])
        colormap = cgrad([colorant"sienna", colorant"burlywood"], length(segs), scale = :log2)
        for (i, seg) in enumerate(segs)
            seg[:color_type] = colormap[i]
        end
    elseif symbol(node) in (:Leaflet, :Leaf)
        node[:color_type] = nothing
    end
end

const dune_turn = deg2rad(-15)

smooth_foot(h, a) = h >= a ? h : h <= -a ? zero(h) : (h + a)^2 / (4a)
crest_y(x) = -2 + 0.5 * (-x - 15) + 5 * sin(x / 13)
crest_height(x) = x < -14 ? 0.55 * (-x - 14)^1.1 : -0.5 * (x + 14)

function dune_height(x, y; windward_slope = tand(13), slip_slope = tand(33), crest_rounding = 4, foot_blend = 6)
    H = crest_height(x)
    d = crest_y(x) - y
    windward = H - windward_slope * (sqrt(d^2 + crest_rounding^2) - crest_rounding)
    h = d > 0 ? min(windward, H - slip_slope * d) : windward
    undulation = 0.15 * sin(x / 9) * sin(y / 11)
    return smooth_foot(h - undulation, foot_blend) + undulation
end

function turned_dune_height(x, y; setback = 4)
    s, c = sincos(dune_turn)
    return dune_height(c * x - s * y + setback, s * x + c * y)
end

function dune_mesh(; xs = range(-260, 30, step = 0.3), ys = range(-260, 200, step = 0.3))
    points = [Point3f(x, y, turned_dune_height(x, y)) for x in xs, y in ys]
    index(i, j) = i + (j - 1) * length(xs)
    faces = GLTriangleFace[]
    for j in 1:length(ys)-1, i in 1:length(xs)-1
        push!(faces, GLTriangleFace(index(i, j), index(i + 1, j), index(i + 1, j + 1)),
                     GLTriangleFace(index(i, j), index(i + 1, j + 1), index(i, j + 1)))
    end
    return normal_mesh(GeometryBasics.Mesh(vec(points), faces))
end

"""
Sand with wind ripples that fork and merge where two half-wavelength shifted ripple fields meet,
smooth on the slip face, and grainy close to the camera.
"""
function sand_material(; color = (0.78, 0.55, 0.34), ripple_period = 0.6, ripple_depth = 0.025, defect_scale = 0.2,
                       grain_scale = 60, grain_depth = 0.004, speckle = 0.15, grain_fade = (6, 20), ripple_fade = (200, 600))
    p = TN.mapping(TN.xyz(); rotation = (0, 0, 0.25 + dune_turn))
    ripple(harmonic, phase) = TN.wave(p; direction = :y, scale = harmonic * 0.314 / ripple_period, phase,
                                      distortion = 6harmonic, detail = 2, detail_scale = 0.4).fac
    skewed(shift) = ripple(1, shift) + 0.3 * ripple(2, π / 2 + 2shift)
    patches = TN.smoothstep(0.4, 0.6, TN.noise(TN.mapping(TN.xyz(); location = (0, 0, 20)); scale = defect_scale, detail = 1).fac)
    ripples = TN.math(:smooth_max, skewed(0) * TN.math(:minimum, 1, 2 - 2patches),
                                   skewed(π) * TN.math(:minimum, 1, 2patches), 0.1)
    windward = TN.smoothstep(-0.3, -0.12, TN.normal().y)

    # sub-pixel bumps alias into streaks, so both fade out with distance
    camera_distance = TN.vector_math(:length, TN.texture_coordinate().camera)
    near_ripples = 1 - TN.smoothstep(ripple_fade..., camera_distance)
    near_grains = 1 - TN.smoothstep(grain_fade..., camera_distance)
    grains = TN.noise(TN.mapping(TN.xyz(); location = (0, 0, 10)); scale = grain_scale, detail = 4, roughness = 0.6).fac
    height = ripple_depth * windward * near_ripples * ripples + grain_depth * near_grains * grains

    speckled = TN.color_ramp(TN.noise(TN.xyz(); scale = 2grain_scale, detail = 2).fac,
                             0.3 => (1 - speckle) .* color, 0.7 => (1 + speckle) .* color)
    return Hikari.Diffuse(; color = speckled, normal = TN.bump(height; distance = 1))
end

const dune = dune_mesh()

function create_scene(; resolution = (1800, 1200))
    sun = Makie.SunSkyLight(Vec3f(0.628, 0.777, 0.087); intensity = 1f0, turbidity = 3f0, ground_enabled = false)
    scene = Scene(; size = resolution, lights = [sun])
    cam3d!(scene)
    leaflet_plot = plantviz!(scene, mtg, symbol = "Leaflet")
    stem_plot = plantviz!(scene, mtg, color = :color_type, symbol = ["RachisSegment", "PetioleSegment", "Internode"])
    base_plot = plantviz!(scene, mtg, color = :color_type, symbol = "Leaf")
    leaflet_plot.plots[1].material = leaf_material
    stem_plot.plots[1].material = stem_material
    base_plot.plots[1].material = stem_material
    foreach(p -> Makie.rotate!(p, Vec3f(0, 0, 1), deg2rad(-15)), (leaflet_plot, stem_plot, base_plot))

    sand = sand_material()
    mesh!(scene, dune; material = sand)
    mesh!(scene, Rect3f(Vec3f(-200_000, -200_000, -0.3), Vec3f(400_000, 400_000, 0.1)); material = sand)

    cam = scene.camera_controls
    cam.eyeposition[] = Vec3f(10, -6, 1.6)
    cam.lookat[] = Vec3f(-10, 2, 7)
    cam.upvector[] = Vec3f(0, 0, 1)
    cam.fov[] = 55f0
    update_cam!(scene, cam)
    return scene
end

function render_scene(;
    device = DEVICE,
    resolution = (1800, 1200),
    samples = 1000,
    max_depth = 30,
    output_path = joinpath(@__DIR__, "plants.png"),
)
    scene = create_scene(; resolution)
    @time result = Makie.colorbuffer(scene; device, samples, max_depth, hw_accel = true,
                                     tonemap = :aces, exposure = 2.2f0, denoise = true, update = false)
    isnothing(output_path) || save(output_path, result)
    @info "Saved → $output_path"
    return result
end

function render_interactive(; device = DEVICE, resolution = (900, 600))
    scene = create_scene(; resolution)
    RayMakie.interactive_window(scene; device)
    display(scene; backend = GLMakie, update = false)
end

# render_scene()
