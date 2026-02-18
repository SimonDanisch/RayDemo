# CMS Detector — Ray-Traced Visualization
#
# Loads the CMS particle detector from GDML via Geant4.jl, applies a quadrant cut
# to reveal internal structure, and renders with physically-based materials.

include("../common/common.jl")
using Geant4
using Geant4.SystemOfUnits
using GeometryBasics
using Colors

# Load quadrant cut and scene construction utilities
include("../common/cms_utils.jl")

# Increase tessellation quality for curved surfaces
HepPolyhedron!SetNumberOfRotationSteps(128)

# ============================================================================
# Load the CMS detector from GDML
# ============================================================================

detname = "cms2018"

gdml_path = joinpath(dirname(pathof(Geant4)), "..", "docs", "src", "examples", "$(detname).gdml")

detector = G4JLDetectorGDML(gdml_path; validate_schema=false)
app = G4JLApplication(; detector=detector, physics_type=FTFP_BERT)
configure(app)
initialize(app)

world = GetWorldVolume()

# ============================================================================
# Collect meshes with quadrant cut
# ============================================================================

println("Collecting detector meshes with quadrant cut...")
@time lv_meshes, glass_meshes = collect_detector_meshes(world; maxlevel=5, cut_quadrant=true)

# ============================================================================
# Render scene
# ============================================================================
function create_trace_scene(lv_meshes; glass_meshes=nothing, resolution=(800, 1080))
    color_groups = group_meshes_by_color(lv_meshes)
    println("Grouped into $(length(color_groups)) unique color groups")

    fig = Figure(size=resolution)
    ax = LScene(fig[1, 1]; show_axis=false, scenekw=(;
        lights=[
            PointLight(RGBf(30, 20, 15), Vec3f(1, 1, 0)),
            Makie.SunSkyLight(Vec3f(3, 2, 8); intensity=1.0f0, turbidity=3.0f0, ground_enabled=false),
        ]
    ))

    # Build single MetaMesh with per-view Hikari materials
    all_points = Point3f[]
    all_normals = Vec3f[]
    all_faces = TriangleFace{Int}[]
    views = UnitRange{UInt32}[]
    material_names = String[]
    materials_dict = Dict{String, Any}()
    mat_idx = 0

    function add_group!(meshes_vec, mat::Hikari.Material, name::String)
        merged = merge(meshes_vec)
        offset = length(all_points)
        append!(all_points, GeometryBasics.coordinates(merged))
        append!(all_normals, GeometryBasics.normals(merged))
        face_start = length(all_faces) + 1
        for f in GeometryBasics.faces(merged)
            push!(all_faces, TriangleFace{Int}(f[1] + offset, f[2] + offset, f[3] + offset))
            if f isa QuadFace
                push!(all_faces, TriangleFace{Int}(f[1] + offset, f[3] + offset, f[4] + offset))
            end
        end
        face_end = length(all_faces)
        push!(views, UInt32(face_start):UInt32(face_end))
        push!(material_names, name)
        materials_dict[name] = mat
    end

    # Opaque meshes — apply color merge to match individual mesh! behavior
    for (key, meshes) in color_groups
        r, g, b, a = key
        a < 0.1 && continue
        mat_idx += 1
        color_rgb = RGB(r, g, b)
        mat = color_to_material((color_rgb, Float64(a)))
        color_tex = Hikari.ConstTexture(RayMakie.to_spectrum(color_rgb))
        mat = RayMakie.merge_color_with_material(color_tex, mat)
        add_group!(meshes, mat, "opaque_$mat_idx")
    end

    # Glass meshes for cut-out wedge — tinted glass like old path
    if glass_meshes !== nothing
        glass_groups = group_meshes_by_color(glass_meshes)
        for (key, meshes) in glass_groups
            r, g, b, a = key
            a < 0.1 && continue
            mat_idx += 1
            color_tex = Hikari.ConstTexture(RayMakie.to_spectrum(RGB(r, g, b)))
            mat = RayMakie.merge_color_with_material(color_tex, Hikari.GlassMaterial())
            add_group!(meshes, mat, "glass_$mat_idx")
        end
    end

    inner_mesh = GeometryBasics.Mesh(all_points, all_faces; views=views, normal=all_normals)
    detector_mesh = GeometryBasics.MetaMesh(inner_mesh;
        material_names=material_names,
        materials=materials_dict,
    )
    println("Combined MetaMesh: $(length(all_points)) vertices, $(length(all_faces)) faces, $(length(views)) material groups")
    mesh!(ax, detector_mesh)

    # Display case box, oriented so back wall faces the camera
    # Camera is at [12, 12, -6], so forward direction is (1,1,0)/√2
    inv_sqrt2 = 1f0 / sqrt(2f0)
    fwd   = Vec3f(inv_sqrt2, inv_sqrt2, 0)    # toward camera
    right = Vec3f(inv_sqrt2, -inv_sqrt2, 0)    # perpendicular in XY
    s  = 7f0   # half-width of box
    sh = 20f0  # ceiling height

    box_material = Hikari.MatteMaterial()
    function make_quad(p1, p2, p3, p4)
        pts = Point3f[p1, p2, p3, p4]
        faces = TriangleFace{Int}[(1,2,3), (1,3,4)]
        return GeometryBasics.Mesh(pts, faces)
    end
    corner(f, r, z) = Point3f(f * fwd + r * right + z * Vec3f(0,0,1))

    # Back wall (normal = +fwd, into scene)
    mesh!(ax, make_quad(
        corner(-s, -s, -s), corner(-s, s, -s),
        corner(-s, s, sh), corner(-s, -s, sh)
    ); color=:gray, material=box_material)
    # Left wall (normal = +right, into scene)
    mesh!(ax, make_quad(
        corner(-s, -s, -s), corner(-s, -s, sh),
        corner(s, -s, sh), corner(s, -s, -s)
    ); color=:gray, material=box_material)
    # Right wall (normal = -right, into scene)
    mesh!(ax, make_quad(
        corner(-s, s, -s), corner(s, s, -s),
        corner(s, s, sh), corner(-s, s, sh)
    ); color=:gray, material=box_material)
    # Floor (normal = +up, into scene)
    mesh!(ax, make_quad(
        corner(-s, -s, -s), corner(-s, s, -s),
        corner(s, s, -s), corner(s, -s, -s)
    ); color=:gray, material=box_material)
    # Ceiling (normal = -up, into scene)
    mesh!(ax, make_quad(
        corner(-s, -s, sh), corner(s, -s, sh),
        corner(s, s, sh), corner(-s, s, sh)
    ); color=:gray, material=box_material)

    return fig, ax
end

function render_scene(;
    device=DEVICE,
    resolution=(800, 1080),
    samples=1000,
    max_depth=6,
    output_path=joinpath(@__DIR__, "cms_detector.png"),
)
    fig_trace, ax_trace = create_trace_scene(lv_meshes; glass_meshes=glass_meshes, resolution=resolution)
    # Camera coords scaled to match normalized scene (~10-unit cube).
    # Original coords were for raw Geant4 mm; scale ≈ 10/30000.
    set_camera!(ax_trace;
        eyeposition = [12, 12, -6],
        lookat = [-0.90, -0.9, 0],
        upvector = [0.0, 0.0, 1.0],
        fov = 45.0
    )

    integrator = Hikari.VolPath(samples=samples, max_depth=max_depth)
    sensor = Hikari.FilmSensor(iso=100, white_balance=6500)

    println("Rendering with GPU, $samples samples...")
    @time result = Makie.colorbuffer(ax_trace.scene;
        device=device,
        integrator=integrator,
        tonemap=:aces,
        sensor=sensor,
        update=false
    )

    save(output_path, result)
    @info "Saved → $output_path"
    return result
end
