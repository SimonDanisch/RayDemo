# Geant4 CMS Detector Utilities
#
# Provides functions to collect meshes from a Geant4 detector hierarchy
# with an optional quadrant cut (removes x>0, y>0 region) using G4SubtractionSolid,
# plus material mapping and scene construction for TraceMakie rendering.

using Geant4
using Geant4.SystemOfUnits
using GeometryBasics
using Colors
using Rotations
using StaticArrays
using LinearAlgebra

# Import Geant4 visualization types
using Geant4: GetSolid, GetVisAttributes, GetMaterial, GetNoDaughters, GetDaughter,
              IsReplicated, GetLogicalVolume, GetTranslation, GetRotation,
              GetPolyhedron, GetNoVertices, GetVertex, GetNoFacets, GetFacet, GetNormal,
              GetNextVertex, G4Point3D, G4Normal3D,
              GetColour, IsVisible, IsDaughtersInvisible, GetTotNbOfElectPerVolume,
              GetTotNbOfAtomsPerVolume, GetDensity, GetName, GetEntityType,
              GetReplicationData, G4IntersectionSolid

# ============================================================================
# Transformation3D - matches G4Vis convention
# ============================================================================

const Vector3 = SVector{3}

struct Transformation3D{T<:AbstractFloat}
    rotation::RotMatrix3{T}
    translation::Vector3{T}
end

Transformation3D{T}(rot::RotMatrix3{T}, trans::Vec3{T}) where T =
    Transformation3D{T}(rot, Vector3{T}(trans...))

Base.one(::Type{Transformation3D{T}}) where T =
    Transformation3D{T}(one(RotMatrix3{T}), Vector3{T}(0, 0, 0))

Base.isone(t::Transformation3D{T}) where T =
    isone(t.rotation) && iszero(t.translation)

# Composition: t1 * t2 gives (t1.rot * t2.rot, t2.rot^T * t1.trans + t2.trans)
Base.:*(t1::Transformation3D{T}, t2::Transformation3D{T}) where T =
    Transformation3D{T}(t1.rotation * t2.rotation, t2.rotation' * t1.translation + t2.translation)

# Inverse: R' and -R * T
Base.inv(t::Transformation3D{T}) where T =
    Transformation3D{T}(t.rotation', -t.rotation * t.translation)

# Apply transform to point: p * t = R^T * p + T (local to world)
apply_transform(t::Transformation3D{T}, p::Vec3{T}) where T =
    t.rotation' * p + t.translation

# Inverse transform: world to local
apply_inv_transform(t::Transformation3D{T}, p::Vec3{T}) where T =
    t.rotation * (p - t.translation)

# ============================================================================
# Color handling
# ============================================================================

const LVColor = Union{ColorTypes.Color, Tuple{ColorTypes.RGB, Float64}}

# Color lookup table by atomic number
const colZ = let c = fill(colorant"gray", 110)
    c[3]  = colorant"silver"       # Li
    c[4]  = colorant"slategray"    # Be
    c[5]  = colorant"gray30"       # B
    c[6]  = colorant"gray30"       # C
    c[7]  = colorant"skyblue"      # N
    c[8]  = colorant"white"        # O
    c[9]  = colorant"lightyellow"  # F
    c[10] = colorant"orange"       # Ne
    c[11] = colorant"silver"       # Na
    c[12] = colorant"silver"       # Mg
    c[13] = colorant"silver"       # Al
    c[14] = colorant"yellow"       # Si
    c[16] = colorant"yellowgreen"  # S
    c[20] = colorant"gold"         # Ca
    c[22] = colorant"lightgray"    # Ti
    c[24] = colorant"darkgray"     # Cr
    c[26] = colorant"orangered"    # Fe
    c[29] = colorant"blue"         # Cu
    c[47] = colorant"lightgray"    # Ag
    c[79] = colorant"gold"         # Au
    c[82] = colorant"lightyellow"  # Pb
    c
end

function default_color(mat)
    z = GetTotNbOfElectPerVolume(mat) / GetTotNbOfAtomsPerVolume(mat) |> round |> Int
    d = GetDensity(mat)
    return (colZ[z], d > 1g/cm3 ? 1.0 : d / (2g/cm3))
end

# ============================================================================
# Mesh collection with quadrant cut
# ============================================================================

const UnitOnAxis = [(1, 0, 0), (0, 1, 0), (0, 0, 1)]

# Debug counter for mesh processing
const _mesh_counter = Ref{Int}(0)

"""
    polyhedron_to_mesh(ph, t::Transformation3D{Float64}) -> Union{GeometryBasics.Mesh, Nothing}

Convert a Geant4 polyhedron to a GeometryBasics mesh with correct per-vertex normals.
Uses Geant4's GetNextVertex iterator which provides normals from FindNodeNormal,
correctly handling sharp edges by duplicating vertices where normals differ.

Returns nothing if the polyhedron is invalid or empty.
"""
function polyhedron_to_mesh(ph, t::Transformation3D{Float64})
    ph == C_NULL && return nothing

    nf = GetNoFacets(ph)
    nf <= 0 && return nothing

    # Debug: track mesh count
    _mesh_counter[] += 1
    if _mesh_counter[] % 100 == 0
        println("Processing mesh #$(_mesh_counter[]), nf=$nf")
    end

    # Fresh objects for iterator (don't reuse - might cause state issues)
    g4_vertex = G4Point3D()
    g4_normal = G4Normal3D()
    edge_flag = Ref{Int32}(0)

    # Collect vertices and normals per face (vertices duplicated at sharp edges)
    all_points = Point3{Float64}[]
    all_normals = Vec3{Float32}[]
    faces = Union{TriangleFace, QuadFace}[]

    face_indices = Int[]
    vertex_idx = 0
    faces_done = 0

    max_iter = min(nf * 5, 10000)  # Hard limit
    iter_count = 0
    GC.@preserve g4_vertex g4_normal begin
        for _ in 1:max_iter
            iter_count += 1
            result = GetNextVertex(ph, g4_vertex, edge_flag, g4_normal)

            # Transform vertex: p_world = R^T * p_local + translation
            p_local = Vec3{Float64}(g4_vertex[1], g4_vertex[2], g4_vertex[3])
            p_world = apply_transform(t, p_local)

            # Transform normal (rotation only, then normalize)
            n_local = Vec3{Float64}(g4_normal[1], g4_normal[2], g4_normal[3])
            n_world = t.rotation * n_local
            n_len = LinearAlgebra.norm(n_world)
            n_world = n_len > 0 ? n_world / n_len : Vec3{Float64}(0, 0, 1)

            # Add vertex with Geant4's computed normal
            vertex_idx += 1
            push!(all_points, Point3{Float64}(p_world...))
            push!(all_normals, Vec3{Float32}(n_world...))
            push!(face_indices, vertex_idx)

            if !result  # End of current face
                faces_done += 1
                n_verts = length(face_indices)
                if n_verts == 3
                    push!(faces, TriangleFace(face_indices...))
                elseif n_verts == 4
                    push!(faces, QuadFace(face_indices...))
                elseif n_verts > 4
                    # Fan triangulation for larger polygons
                    for i in 2:(n_verts-1)
                        push!(faces, TriangleFace(face_indices[1], face_indices[i], face_indices[i+1]))
                    end
                end
                empty!(face_indices)
                faces_done >= nf && break
            end
        end
    end

    # Debug: warn if we didn't process all faces
    if faces_done < nf
        @warn "polyhedron_to_mesh: only processed $faces_done/$nf faces (iter=$iter_count, max=$max_iter)"
    end

    isempty(faces) && return nothing
    return GeometryBasics.Mesh(all_points, faces; normal=all_normals)
end

"""
    collect_meshes_with_cut!(lv, lv_meshes, glass_meshes, cutter, cutter_offset, t, level, maxlevel)

Recursively collect meshes from a logical volume hierarchy, applying a quadrant cut
to remove the x>0, y>0 region in world coordinates. Also collects the cut-out pieces
for glass visualization.

# Arguments
- `lv::G4LogicalVolume`: The logical volume to process
- `lv_meshes::Dict`: Output dictionary mapping LogicalVolume -> (meshes, color, visible)
- `glass_meshes::Dict`: Output dictionary for cut-out pieces (same structure as lv_meshes)
- `cutter::G4VSolid`: The cutting solid (typically a large box)
- `cutter_offset::Vec3{Float64}`: Center of the cutter in world coordinates
- `t::Transformation3D{Float64}`: Accumulated transform from world to this volume
- `level::Int64`: Current recursion depth
- `maxlevel::Int64`: Maximum recursion depth
"""
function collect_meshes_with_cut!(
    lv::G4LogicalVolume,
    lv_meshes::Dict{G4LogicalVolume, Tuple{Vector{GeometryBasics.Mesh}, LVColor, Bool}},
    glass_meshes::Dict{G4LogicalVolume, Tuple{Vector{GeometryBasics.Mesh}, LVColor, Bool}},
    cutter::G4VSolid,
    cutter_offset::Vec3{Float64},
    t::Transformation3D{Float64},
    level::Int64,
    maxlevel::Int64
)
    solid = GetSolid(lv)
    g4vis = GetVisAttributes(lv)

    # Transform the cutter into the local coordinate system
    local_cutter_pos = apply_inv_transform(t, Vec3{Float64}(cutter_offset...))

    # Build G4RotationMatrix from accumulated rotation
    R = t.rotation
    rotColX = G4ThreeVector(R[1,1], R[2,1], R[3,1])
    rotColY = G4ThreeVector(R[1,2], R[2,2], R[3,2])
    rotColZ = G4ThreeVector(R[1,3], R[2,3], R[3,3])
    g4rot = G4RotationMatrix(rotColX, rotColY, rotColZ)
    local_cutter_transform = G4Transform3D(
        g4rot,
        G4ThreeVector(local_cutter_pos[1], local_cutter_pos[2], local_cutter_pos[3])
    )

    # Get the original solid's polyhedron
    ph_orig = GetPolyhedron(solid[])
    if ph_orig == C_NULL
        @goto after_mesh
    end

    nv_orig = GetNoVertices(ph_orig)
    if nv_orig <= 0
        @goto after_mesh
    end

    # Check if cutting is needed by examining vertices in world space
    # Use GC.@preserve to ensure solid stays alive while we access ph_orig
    needs_cut = false
    all_inside_cutter = true
    GC.@preserve solid for i in 1:nv_orig
        v = GetVertex(ph_orig, i)
        p_world = apply_transform(t, Vec3{Float64}(v[1], v[2], v[3]))
        in_cutter = p_world[1] > 0 && p_world[2] > 0
        if in_cutter
            needs_cut = true
        else
            all_inside_cutter = false
        end
    end

    # If all vertices are inside the cutter, solid is completely removed
    if all_inside_cutter
        @goto after_mesh
    end

    # Get color info for this logical volume
    color = g4vis != C_NULL ? convert(Tuple{RGB, Float64}, GetColour(g4vis)) : default_color(GetMaterial(lv))
    visible = g4vis != C_NULL ? IsVisible(g4vis) : color[2] >= 0.1

    # Choose which polyhedron to use for the main (subtracted) mesh
    local ph
    local cut_solid_ref = nothing  # Keep reference alive to prevent GC
    local intersect_solid_ref = nothing  # Keep reference alive for intersection solid
    if needs_cut
        cut_solid_ref = G4SubtractionSolid(
            "cut_$(GetName(solid[]))",
            solid,
            CxxPtr(cutter),
            local_cutter_transform
        )
        ph = GetPolyhedron(cut_solid_ref)

        # Also create the intersection (cut-out piece) for glass visualization
        intersect_solid_ref = G4IntersectionSolid(
            "glass_$(GetName(solid[]))",
            solid,
            CxxPtr(cutter),
            local_cutter_transform
        )
        glass_ph = GetPolyhedron(intersect_solid_ref)
        # Use GC.@preserve to ensure intersect_solid_ref stays alive while we access glass_ph
        glass_mesh = GC.@preserve intersect_solid_ref polyhedron_to_mesh(glass_ph, t)
        if glass_mesh !== nothing
            if haskey(glass_meshes, lv)
                push!(glass_meshes[lv][1], glass_mesh)
            else
                glass_meshes[lv] = ([glass_mesh], color, visible)
            end
        end
    else
        ph = ph_orig
    end

    # Create the main mesh from the (possibly subtracted) solid
    # Use GC.@preserve to ensure the solid owning ph stays alive:
    # - solid owns ph_orig (when needs_cut=false)
    # - cut_solid_ref owns ph (when needs_cut=true)
    m = GC.@preserve solid cut_solid_ref polyhedron_to_mesh(ph, t)
    if m !== nothing
        if haskey(lv_meshes, lv)
            push!(lv_meshes[lv][1], m)
        else
            lv_meshes[lv] = ([m], color, visible)
        end
    end

    @label after_mesh

    # Recurse to daughters
    level >= maxlevel && return
    g4vis != C_NULL && IsDaughtersInvisible(g4vis) && return

    for idx in 1:GetNoDaughters(lv)
        daughter = GetDaughter(lv, idx - 1)

        if IsReplicated(daughter)
            volume = GetLogicalVolume(daughter)
            axis = Ref{EAxis}(kYAxis)
            nReplicas = Ref{Int32}(0)
            width = Ref{Float64}(0.0)
            offset_val = Ref{Float64}(0.0)
            consuming = Ref{UInt8}(0)
            GetReplicationData(daughter, axis, nReplicas, width, offset_val, consuming)

            unitV = Vector3{Float64}(UnitOnAxis[axis[] + 1]...)
            for i in 1:nReplicas[]
                local_t = unitV * (-width[] * (nReplicas[] - 1) * 0.5 + (i - 1) * width[])
                child_transform = Transformation3D{Float64}(one(RotMatrix3{Float64}), local_t)
                new_t = child_transform * t
                collect_meshes_with_cut!(volume[], lv_meshes, glass_meshes, cutter, cutter_offset, new_t, level + 1, maxlevel)
            end
        else
            g4t = GetTranslation(daughter)
            g4r = GetRotation(daughter)

            local_rot = if g4r == C_NULL
                one(RotMatrix3{Float64})
            else
                RotMatrix3{Float64}(
                    xx(g4r[]), yx(g4r[]), zx(g4r[]),
                    xy(g4r[]), yy(g4r[]), zy(g4r[]),
                    xz(g4r[]), yz(g4r[]), zz(g4r[])
                )
            end
            local_trans = Vector3{Float64}(x(g4t), y(g4t), z(g4t))

            child_transform = Transformation3D{Float64}(local_rot, local_trans)
            new_t = child_transform * t
            volume = GetLogicalVolume(daughter)
            collect_meshes_with_cut!(volume[], lv_meshes, glass_meshes, cutter, cutter_offset, new_t, level + 1, maxlevel)
        end
    end
end

"""
    create_quadrant_cutter(size=50000.0; epsilon=0.001)

Create a box cutter for removing the x>0, y>0 quadrant.
Returns (cutter_solid, cutter_offset).

A small epsilon offset is added to avoid coincident surfaces with detector geometry,
which would cause "BooleanProcessor::createPolyhedron : too many edges" errors.
"""
function create_quadrant_cutter(size::Float64=50000.0; epsilon::Float64=0.001)
    cutter = G4Box("quadrant_cutter", size/2, size/2, size)
    # Add tiny offset to break surface coincidence at x=0, y=0 planes
    offset = Vec3{Float64}(size/2 - epsilon, size/2 - epsilon, 0)
    return cutter, offset
end

"""
    collect_detector_meshes(world; maxlevel=5, cut_quadrant=true)

Collect all meshes from a Geant4 detector world volume.

Returns (lv_meshes, glass_meshes) where each is a Dict mapping
LogicalVolume -> (meshes, color, visible). glass_meshes contains the
cut-out pieces for glass visualization.
"""
function collect_detector_meshes(world; maxlevel::Int=5, cut_quadrant::Bool=true)
    lv = GetLogicalVolume(world)[]
    lv_meshes = Dict{G4LogicalVolume, Tuple{Vector{GeometryBasics.Mesh}, LVColor, Bool}}()
    glass_meshes = Dict{G4LogicalVolume, Tuple{Vector{GeometryBasics.Mesh}, LVColor, Bool}}()

    cutter, cutter_offset = create_quadrant_cutter()
    collect_meshes_with_cut!(lv, lv_meshes, glass_meshes, cutter, cutter_offset,
                                one(Transformation3D{Float64}), 1, maxlevel)

    return lv_meshes, glass_meshes
end

# ============================================================================
# Material mapping from detector colors to Hikari materials
# ============================================================================

function color_to_material(color::Tuple{RGB, Float64})
    rgb = color[1]
    r, g, b = Float32(red(rgb)), Float32(green(rgb)), Float32(blue(rgb))

    # Map specific colors to appropriate materials
    if r > 0.8 && g < 0.4 && b < 0.4  # Orangered (Fe)
        return Hikari.CoatedDiffuseMaterial(reflectance=(r, g, b), roughness=0.15f0, eta=1.5f0)
    elseif r < 0.3 && g < 0.3 && b > 0.7  # Blue (Cu)
        return Hikari.Copper(roughness=0.1f0, reflectance=(0.5f0, 0.5f0, 1.0f0))
    elseif r > 0.8 && g > 0.8 && b < 0.3  # Yellow (Si)
        return Hikari.Gold(roughness=0.2f0, reflectance=(1.0f0, 1.0f0, 0.8f0))
    elseif r > 0.7 && g > 0.7 && b > 0.7  # White/silver
        return Hikari.Silver(roughness=0.1f0)
    elseif r > 0.5 && g > 0.5 && b > 0.5  # Gray
        return Hikari.Aluminum(roughness=0.15f0)
    elseif r < 0.4 && g > 0.6 && b > 0.8  # Skyblue (N)
        return Hikari.CoatedDiffuseMaterial(reflectance=(r, g, b), roughness=0.1f0, eta=1.5f0)
    else
        return Hikari.CoatedDiffuseMaterial(reflectance=(r, g, b), roughness=0.2f0, eta=1.5f0)
    end
end

color_to_material(color::ColorTypes.Color) =
    color_to_material((RGB(Float32(red(color)), Float32(green(color)), Float32(blue(color))), 1.0))

"""
Convert a color to a thin glass material for the cut-out wedge visualization.
"""
function color_to_glass_material(color::Tuple{RGB, Float64})
    # ThinDielectric passes light straight through (no refraction bending)
    return Hikari.ThinDielectric(eta=1.5f0)
end

color_to_glass_material(color::ColorTypes.Color) =
    color_to_glass_material((RGB(Float32(red(color)), Float32(green(color)), Float32(blue(color))), 1.0))

# ============================================================================
# Group meshes by color for efficient rendering
# ============================================================================

function group_meshes_by_color(lv_meshes; quantize_digits=2)
    color_groups = Dict{NTuple{4,Float32}, Vector{GeometryBasics.Mesh}}()

    for (lv, (meshes, color, visible)) in lv_meshes
        !visible && continue

        rgb, alpha = if color isa Tuple
            color[1], Float64(color[2])
        else
            color, 1.0
        end

        # Quantize color to reduce unique materials
        r = round(Float32(red(rgb)), digits=quantize_digits)
        g = round(Float32(green(rgb)), digits=quantize_digits)
        b = round(Float32(blue(rgb)), digits=quantize_digits)
        a = round(Float32(alpha), digits=quantize_digits)
        key = (r, g, b, a)

        if !haskey(color_groups, key)
            color_groups[key] = GeometryBasics.Mesh[]
        end
        append!(color_groups[key], meshes)
    end

    return color_groups
end

# ============================================================================
# GLMakie visualization
# ============================================================================

function draw_detector!(scene, lv_meshes; wireframe::Bool=false)
    for (lv, (meshes, color, visible)) in lv_meshes
        !visible && continue
        m = merge(meshes)
        if wireframe
            Makie.wireframe!(scene, m; linewidth=1)
        else
            Makie.mesh!(scene, m; color=color)
        end
    end
    return scene
end

# ============================================================================
# Create TraceMakie scene
# ============================================================================

function create_trace_scene(lv_meshes; glass_meshes=nothing, resolution=(800, 1080))
    color_groups = group_meshes_by_color(lv_meshes)
    println("Grouped into $(length(color_groups)) unique color groups")

    fig = Figure(size=resolution)
    ax = LScene(fig[1, 1]; show_axis=false, scenekw=(;
        backgroundcolor=RGBf(0.02, 0.02, 0.03),
        lights=[
            PointLight(RGBf(8, 8, 10), Vec3f(-8000, 3000, -5000)),
            AmbientLight(RGBf(0.15, 0.15, 0.18)),
        ]
    ))

    # Add meshes with materials
    for (key, meshes) in color_groups
        r, g, b, a = key
        a < 0.1 && continue

        merged = merge(meshes)
        color_rgb = RGB(r, g, b)
        mat = color_to_material((color_rgb, Float64(a)))
        mesh!(ax, merged; color=color_rgb, material=mat)
    end

    # Add glass meshes for the cut-out wedge
    if glass_meshes !== nothing
        glass_groups = group_meshes_by_color(glass_meshes)
        println("Grouped into $(length(glass_groups)) unique glass color groups")

        for (key, meshes) in glass_groups
            r, g, b, a = key
            a < 0.1 && continue

            merged = merge(meshes)
            color_rgb = RGB(r, g, b)
            mat = color_to_glass_material((color_rgb, Float64(a)))
            mesh!(ax, merged; color=color_rgb, material=mat)
        end
    end

    return fig, ax
end

# ============================================================================
# Set camera from show_cam output
# ============================================================================

function set_camera!(ax; eyeposition, lookat, upvector, fov=45.0)
    cam = ax.scene.camera_controls
    cam.eyeposition[] = Vec3f(eyeposition...)
    cam.lookat[] = Vec3f(lookat...)
    up = Vec3f(upvector...)
    cam.upvector[] = up / norm(up)
    cam.fov[] = Float32(fov)
    Makie.update_cam!(ax.scene, cam)
    return cam
end
