# Killeroo Gold Scene - PBRT-v4 Port
# Port of pbrt-v4-scenes/killeroos/killeroo-gold.pbrt to RayMakie
#
# This scene showcases:
# - Loop subdivision surface parsing and tessellation
# - Gold conductor material with physically-based optical properties
# - Area lighting approximation

include("../common/common.jl")
using GeometryBasics, LinearAlgebra

# Scene data lives alongside this script (geometry/killeroo3.pbrt)
const KILLEROO_DIR = @__DIR__
@assert isdir(joinpath(KILLEROO_DIR, "geometry")) "geometry/ not found in $KILLEROO_DIR"

# =============================================================================
# PBRT LoopSubdiv Parser
# =============================================================================

function parse_loopsubdiv_pbrt(filepath::String)
    content = read(filepath, String)
    levels_match = match(r"\"integer levels\"\s*\[\s*(\d+)\s*\]", content)
    levels = levels_match !== nothing ? parse(Int, levels_match.captures[1]) : 3

    p_match = match(r"\"point3 P\"\s*\[([\s\S]*?)\](?=\s*\"integer)", content)
    p_match === nothing && error("Could not find point3 P in file")
    p_nums = [parse(Float32, m.match) for m in eachmatch(r"-?[\d.]+(?:e[+-]?\d+)?", p_match.captures[1])]
    points = [Point3f(p_nums[i], p_nums[i+1], p_nums[i+2]) for i in 1:3:length(p_nums)]

    idx_match = match(r"\"integer indices\"\s*\[([\s\S]*?)\]", content)
    idx_match === nothing && error("Could not find integer indices in file")
    indices = [parse(Int, m.match) for m in eachmatch(r"\d+", idx_match.captures[1])]
    faces = [(indices[i]+1, indices[i+1]+1, indices[i+2]+1) for i in 1:3:length(indices)]

    return points, faces, levels
end

# =============================================================================
# Loop Subdivision Algorithm
# =============================================================================

loop_beta(valence::Int) = valence == 3 ? 3f0 / 16f0 : 3f0 / (8f0 * valence)

function build_adjacency(vertices::Vector{Point3f}, faces::Vector{NTuple{3,Int}})
    n = length(vertices)
    neighbors = [Set{Int}() for _ in 1:n]
    for (v1, v2, v3) in faces
        push!(neighbors[v1], v2, v3)
        push!(neighbors[v2], v1, v3)
        push!(neighbors[v3], v1, v2)
    end
    return neighbors
end

function find_boundary_vertices(vertices::Vector{Point3f}, faces::Vector{NTuple{3,Int}})
    edge_count = Dict{Tuple{Int,Int}, Int}()
    for (v1, v2, v3) in faces
        for (a, b) in [(v1,v2), (v2,v3), (v3,v1)]
            edge = minmax(a, b)
            edge_count[edge] = get(edge_count, edge, 0) + 1
        end
    end
    boundary = falses(length(vertices))
    for ((a, b), count) in edge_count
        count == 1 && (boundary[a] = true; boundary[b] = true)
    end
    return boundary
end

function subdivide_once(vertices::Vector{Point3f}, faces::Vector{NTuple{3,Int}})
    neighbors = build_adjacency(vertices, faces)
    boundary = find_boundary_vertices(vertices, faces)

    new_vertices = Vector{Point3f}()

    for (i, v) in enumerate(vertices)
        neighs = collect(neighbors[i])
        valence = length(neighs)
        if boundary[i]
            boundary_neighs = filter(n -> boundary[n], neighs)
            if length(boundary_neighs) >= 2
                bn = boundary_neighs[1:2]
                new_p = 0.75f0 * v + 0.125f0 * vertices[bn[1]] + 0.125f0 * vertices[bn[2]]
            else
                new_p = v
            end
        else
            b = loop_beta(valence)
            ring_sum = sum(vertices[n] for n in neighs)
            new_p = (1f0 - valence * b) * v + b * ring_sum
        end
        push!(new_vertices, new_p)
    end

    edge_vertex_map = Dict{Tuple{Int,Int}, Int}()
    edge_faces = Dict{Tuple{Int,Int}, Vector{Int}}()

    for (fi, (v1, v2, v3)) in enumerate(faces)
        for (a, b) in [(v1,v2), (v2,v3), (v3,v1)]
            edge = minmax(a, b)
            if !haskey(edge_faces, edge)
                edge_faces[edge] = Int[]
            end
            push!(edge_faces[edge], fi)
        end
    end

    for (edge, adj_faces) in edge_faces
        a, b = edge
        if length(adj_faces) == 1
            new_p = 0.5f0 * (vertices[a] + vertices[b])
        else
            f1, f2 = adj_faces[1], adj_faces[2]
            opp1 = setdiff(faces[f1], (a, b))[1]
            opp2 = setdiff(faces[f2], (a, b))[1]
            new_p = 0.375f0 * (vertices[a] + vertices[b]) + 0.125f0 * (vertices[opp1] + vertices[opp2])
        end
        push!(new_vertices, new_p)
        edge_vertex_map[edge] = length(new_vertices)
    end

    new_faces = Vector{NTuple{3,Int}}()
    for (v1, v2, v3) in faces
        e12 = edge_vertex_map[minmax(v1, v2)]
        e23 = edge_vertex_map[minmax(v2, v3)]
        e31 = edge_vertex_map[minmax(v3, v1)]
        push!(new_faces, (v1, e12, e31))
        push!(new_faces, (v2, e23, e12))
        push!(new_faces, (v3, e31, e23))
        push!(new_faces, (e12, e23, e31))
    end

    return new_vertices, new_faces
end

function loop_subdivide(vertices::Vector{Point3f}, faces::Vector{NTuple{3,Int}}, levels::Int)
    v, f = vertices, faces
    for i in 1:levels
        v, f = subdivide_once(v, f)
    end
    return v, f
end

function to_mesh(vertices::Vector{Point3f}, faces::Vector{NTuple{3,Int}})
    face_indices = [TriangleFace{Int}(f[1], f[2], f[3]) for f in faces]
    GeometryBasics.normal_mesh(GeometryBasics.Mesh(vertices, face_indices))
end

# =============================================================================
# Scene Creation
# =============================================================================

function load_killeroo_mesh(; levels=3)
    filepath = joinpath(KILLEROO_DIR, "geometry", "killeroo3.pbrt")
    points, faces, _ = parse_loopsubdiv_pbrt(filepath)
    subdiv_verts, subdiv_faces = loop_subdivide(points, faces, levels)
    return to_mesh(subdiv_verts, subdiv_faces)
end

function create_scene(; resolution=(684, 513), subdivision_levels=3)
    println("Loading killeroo mesh ($(subdivision_levels) subdivision levels)...")
    killeroo_mesh = load_killeroo_mesh(; levels=subdivision_levels)
    println("  $(length(coordinates(killeroo_mesh))) vertices")

    scene = Scene(size=resolution; lights=Makie.AbstractLight[], ambient=RGBf(0.02, 0.02, 0.02))
    cam3d!(scene)

    # Camera: LookAt 200 250 70   0 33 -50   0 0 1
    update_cam!(scene, Vec3f(200, 250, 70), Vec3f(0, 33, -50), Vec3f(0, 0, 1))
    scene.camera_controls.fov[] = 38.0

    # Gold conductor material (pbrt: roughness 0.002)
    gold_material = Hikari.Gold(roughness=0.002f0)
    mesh!(scene, killeroo_mesh; material=gold_material)

    # Ground planes
    ground_material = Hikari.Diffuse(Kd=(0.5f0, 0.5f0, 0.5f0))
    ground_z = -140f0
    ground_size = 400f0
    mesh!(scene, Rect3f(Vec3f(-ground_size, -ground_size, ground_z - 0.1f0),
                        Vec3f(2*ground_size, 2*ground_size, 0.2f0)); material=ground_material)
    mesh!(scene, Rect3f(Vec3f(-ground_size, -ground_size, ground_z),
                        Vec3f(2*ground_size, 0.2f0, 1000f0)); material=ground_material)
    mesh!(scene, Rect3f(Vec3f(-ground_size, -ground_size, ground_z),
                        Vec3f(0.2f0, 2*ground_size, 1000f0)); material=ground_material)

    # Lights
    push_light!(scene, Makie.PointLight(RGBf(5000, 5000, 5000), Vec3f(0, 0, 800)))
    push_light!(scene, Makie.DirectionalLight(RGBf(0.2, 0.2, 0.2), Vec3f(-200, -250, -70)))

    return scene
end

# =============================================================================
# Standardized RayDemo render_scene API
# =============================================================================

function render_scene(;
    device=DEVICE,
    resolution=(684, 513),
    samples=32,
    max_depth=8,
    output_path=joinpath(@__DIR__, "output", "killeroo_gold.png"),
    hw_accel=false,
)
    scene = create_scene(; resolution=resolution)
    GC.gc(true)
    sensor = Hikari.FilmSensor(; iso=100, white_balance=5500)
    RayMakie.activate!(; device=device, sensor=sensor, exposure=1.0f0, tonemap=:aces, gamma=2.2f0)
    integrator = Hikari.VolPath(; samples=samples, max_depth=max_depth, hw_accel=hw_accel)
    @time img = colorbuffer(scene; backend=RayMakie, integrator=integrator)
    mkpath(dirname(output_path))
    save(output_path, img)
    @info "Saved → $output_path"
    return img
end

# render_scene()
