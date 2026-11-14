include("../common/common.jl")
using ProtPlot, ProteinChains, GeometryBasics, LinearAlgebra

structure = pdb"1HQK"

# ProtPlot's `ribbon!` recipe does not run on this Makie, so the ribbon mesh is built with ProtPlot's own functions.
struct RibbonShape
    attributes::NamedTuple
end

function ProtPlot.get_surface_segment(r::RibbonShape, ::Val{ProtPlot.COIL}, range::UnitRange{Int}, backbone::Array{<:Real, 3})
    adjusted = max(1, range.start - 1):min(size(backbone, 3), range.stop + 1)
    return ProtPlot.coil_surface(r.attributes, adjusted, @view(backbone[:, 2, :])), adjusted
end
ProtPlot.get_surface_segment(r::RibbonShape, ::Val{ProtPlot.HELIX}, range::UnitRange{Int}, backbone::Array{<:Real, 3}) =
    ProtPlot.helix_surface(r.attributes, range, @view(backbone[:, 2, :])), range
ProtPlot.get_surface_segment(r::RibbonShape, ::Val{ProtPlot.STRAND}, range::UnitRange{Int}, backbone::Array{<:Real, 3}) =
    ProtPlot.strand_surface(r.attributes, range, @view(backbone[:, 2, :]), ProtPlot.ASS.get_oxygen_positions(backbone)), range

function ribbon_mesh(structure)
    attributes = (coil_diameter = 0.5, coil_spline_quality = 20, coil_slice_quality = 20,
                  helix_width = 2.2, helix_thickness = 0.5, helix_spline_quality = 20, helix_slice_quality = 20,
                  strand_width = 2.2, strand_thickness = 0.5, strand_spline_quality = 20,
                  strand_arrow_head_length = 5, strand_arrow_head_width = 3.5)
    shape = RibbonShape(map(Ref, attributes))
    backbones = convert.(Array{Float64, 3}, Makie.convert_arguments(ProtPlot.Ribbon, structure)[1])
    vertices, faces, colors = Point3f[], GLTriangleFace[], RGBAf[]
    for (backbone, ss) in zip(backbones, ProtPlot.ASS.assign_secondary_structure(backbones))
        color = ProtPlot.resolve_colormap(range(0, 1, size(backbone, 3)), :jet, (0, 1))
        for r in ProtPlot.get_subchain_ranges(backbone)
            length(r) > 1 && ProtPlot.add_segments!(vertices, faces, colors, shape, backbone[:, :, r], ss[r], color[r])
        end
    end
    return normal_mesh(GeometryBasics.Mesh(vertices, faces)), colors
end

const eye, target, up = Point3f(165.43, -10, 233.86), Point3f(148.04, 0, 126.64), Vec3f(0, 1, 0)

"""A square of side `side` at `offset` (right, up, towards the camera) from the target, facing it."""
function light_panel(offset, side)
    forward = normalize(target - eye)
    right = normalize(cross(forward, up))
    center = target + offset[1] * right + offset[2] * up - offset[3] * forward
    n = normalize(target - center)
    u = normalize(cross(n, up))
    v = cross(u, n)
    corners = [center + side / 2 * (a * u + b * v) for (a, b) in ((-1, -1), (1, -1), (1, 1), (-1, 1))]
    return normal_mesh(GeometryBasics.Mesh(corners, [GLTriangleFace(1, 2, 3), GLTriangleFace(1, 3, 4)]))
end

const key_and_rim_lights = [((250, 250, 250), 375, 40), ((-350, 0, -250), 300, 40), ((350, -100, -250), 300, 40)]

function create_scene(; resolution = (1500, 1500))
    scene = Scene(size = resolution)
    cam3d!(scene)
    ribbon, colors = ribbon_mesh(structure)
    mesh!(scene, ribbon; color = colors, material = Hikari.Glass())

    backdrop = normal_mesh(Tessellation(Sphere(target, 2000f0), 64))
    mesh!(scene, backdrop; material = Hikari.Emission(color = (0.03, 0.03, 0.03), strength = 1, two_sided = true))
    for (offset, side, strength) in key_and_rim_lights
        mesh!(scene, light_panel(offset, side); material = Hikari.Emission(; color = (1, 1, 1), strength, two_sided = true),
              ray_visibility = Hikari.Visibility(camera = false))
    end

    cam = scene.camera_controls
    cam.eyeposition[] = eye
    cam.lookat[] = target
    cam.upvector[] = up
    cam.fov[] = 45
    Makie.update_cam!(scene, cam)
    return scene
end

function render_scene(;
    device = DEVICE,
    resolution = (1500, 1500),
    samples = 1000,
    max_depth = 128, # light inside the thin glass tubes needs many bounces, or the helix ends turn black
    output_path = joinpath(@__DIR__, "protein_glass.png"),
)
    scene = create_scene(; resolution)
    @time result = Makie.colorbuffer(scene; device, samples, max_depth, hw_accel = true,
                                     tonemap = :aces, exposure = 0.6f0, denoise = true, update = false)
    isnothing(output_path) || save(output_path, result)
    @info "Saved → $output_path"
    return result
end

function render_interactive(; device = DEVICE, resolution = (800, 800))
    scene = create_scene(; resolution)
    RayMakie.interactive_window(scene; device)
    display(scene; backend = GLMakie, update = false)
end
render_scene()
