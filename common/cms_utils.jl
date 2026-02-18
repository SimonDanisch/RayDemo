# Geant4 CMS Detector Utilities
#
# Provides functions to collect meshes from a Geant4 detector hierarchy
# with an optional quadrant cut (removes x>0, y>0 region) using G4SubtractionSolid,
# plus material mapping and scene construction for RayMakie rendering.

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

Convert a Geant4 polyhedron to a GeometryBasics mesh with smooth per-vertex normals
and correct hard-edge handling.

Uses index-based GetVertex/GetFacet instead of the slow stateful GetNextVertex iterator.
Vertices at hard edges (crease angle > threshold) are duplicated so that each smooth
group gets its own normal. This prevents the renderer from flipping shading normals
due to winding/normal disagreement at sharp features.

Returns nothing if the polyhedron is invalid or empty.
"""
function polyhedron_to_mesh(ph, t::Transformation3D{Float64}; crease_cos::Float64=0.5)
    ph == C_NULL && return nothing

    nv = GetNoVertices(ph)
    nf = GetNoFacets(ph)
    (nv <= 0 || nf <= 0) && return nothing

    # Debug: track mesh count
    _mesh_counter[] += 1
    if _mesh_counter[] % 100 == 0
        println("Processing mesh #$(_mesh_counter[]), nv=$nv, nf=$nf")
    end

    # Collect all vertices at once (nv FFI calls)
    local_points = Vector{Vec3{Float64}}(undef, nv)
    world_points = Vector{Point3{Float64}}(undef, nv)
    for i in 1:nv
        v = GetVertex(ph, i)
        p_local = Vec3{Float64}(v[1], v[2], v[3])
        local_points[i] = p_local
        world_points[i] = Point3{Float64}(apply_transform(t, p_local)...)
    end

    # Pass 1: Collect face data (nf FFI calls for GetFacet + nf for GetNormal)
    # Store: face vertex indices, face normal (world space), vertex count per face
    face_verts = Vector{NTuple{4,Int32}}(undef, nf)
    face_nverts = Vector{Int}(undef, nf)
    face_normals_world = Vector{Vec3{Float64}}(undef, nf)
    # vertex → list of face indices
    vert_faces = [Int[] for _ in 1:nv]

    nodes_buf = Vector{Int32}(undef, 4)
    n_ref = Ref{Int32}(0)
    valid_faces = 0

    for i in 1:nf
        GetFacet(ph, i, n_ref, nodes_buf)
        nn = n_ref[]
        nn < 3 && continue
        valid_faces += 1

        g4n = GetNormal(ph, i)
        g4_normal = Vec3{Float64}(g4n[1], g4n[2], g4n[3])
        g4_world = t.rotation' * g4_normal
        g4_len = LinearAlgebra.norm(g4_world)
        if g4_len > 0
            g4_world = g4_world / g4_len
        end

        face_verts[valid_faces] = (nodes_buf[1], nodes_buf[2], nodes_buf[3], nodes_buf[4])
        face_nverts[valid_faces] = nn
        face_normals_world[valid_faces] = g4_world

        for j in 1:nn
            push!(vert_faces[nodes_buf[j]], valid_faces)
        end
    end

    valid_faces == 0 && return nothing

    # Pass 2: For each (vertex, face) pair, compute the smooth normal by averaging
    # normals from adjacent faces at that vertex within the crease angle.
    # If a vertex has multiple smooth groups, it gets duplicated.
    out_points = Point3{Float64}[]
    out_normals = Vec3{Float32}[]
    # Map (original_vertex, face_index) → new vertex index
    # We use a per-vertex group assignment to reuse indices within smooth groups.
    vert_new_idx = Vector{Vector{Tuple{Vec3{Float64}, Int}}}(undef, nv)
    for i in 1:nv
        vert_new_idx[i] = Tuple{Vec3{Float64}, Int}[]
    end

    function get_or_create_vertex(vi::Int32, face_idx::Int)
        fn = face_normals_world[face_idx]
        # Compute smooth normal for this vertex from compatible adjacent faces
        smooth_n = Vec3{Float64}(0, 0, 0)
        for adj_fi in vert_faces[vi]
            if LinearAlgebra.dot(face_normals_world[adj_fi], fn) >= crease_cos
                smooth_n += face_normals_world[adj_fi]
            end
        end
        sn_len = LinearAlgebra.norm(smooth_n)
        smooth_n = sn_len > 0 ? smooth_n / sn_len : fn

        # Check if we already have a vertex with this smooth normal
        for (existing_n, idx) in vert_new_idx[vi]
            if LinearAlgebra.dot(existing_n, smooth_n) > 0.999
                return idx
            end
        end

        # Create new vertex
        push!(out_points, world_points[vi])
        push!(out_normals, Vec3{Float32}(smooth_n...))
        new_idx = length(out_points)
        push!(vert_new_idx[vi], (smooth_n, new_idx))
        return new_idx
    end

    # Pass 3: Build faces with new vertex indices + winding check
    out_faces = Vector{Union{TriangleFace, QuadFace}}()
    sizehint!(out_faces, valid_faces)

    for fi in 1:valid_faces
        fv = face_verts[fi]
        nn = face_nverts[fi]

        # Get new vertex indices
        new_vi = ntuple(j -> get_or_create_vertex(fv[j], fi), nn)

        # Winding check: geometric normal from world-space points vs face normal
        p1 = out_points[new_vi[1]]
        p2 = out_points[new_vi[2]]
        p3 = out_points[new_vi[3]]
        e1 = Vec3{Float64}((p2 - p1)...)
        e2 = Vec3{Float64}((p3 - p1)...)
        geo_n = Vec3{Float64}(
            e1[2]*e2[3] - e1[3]*e2[2],
            e1[3]*e2[1] - e1[1]*e2[3],
            e1[1]*e2[2] - e1[2]*e2[1]
        )
        needs_flip = LinearAlgebra.dot(geo_n, face_normals_world[fi]) < 0

        if nn == 3
            if needs_flip
                push!(out_faces, TriangleFace(new_vi[1], new_vi[3], new_vi[2]))
            else
                push!(out_faces, TriangleFace(new_vi[1], new_vi[2], new_vi[3]))
            end
        elseif nn == 4
            if needs_flip
                push!(out_faces, QuadFace(new_vi[1], new_vi[4], new_vi[3], new_vi[2]))
            else
                push!(out_faces, QuadFace(new_vi[1], new_vi[2], new_vi[3], new_vi[4]))
            end
        end
    end

    isempty(out_faces) && return nothing
    return GeometryBasics.Mesh(out_points, out_faces; normal=out_normals)
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

    # Normalize all meshes to fit in a ~10-unit cube (preserving aspect ratio).
    # Geant4 coordinates can be very large (thousands of mm), which causes
    # Float32 precision issues in the ray tracer.
    normalize_meshes!(lv_meshes, glass_meshes)

    return lv_meshes, glass_meshes
end

"""
    normalize_meshes!(lv_meshes, glass_meshes)

Rescale all mesh vertices so the entire scene fits in a cube of roughly size 10,
centered at the origin. Preserves aspect ratio. Normals are unaffected since they
are unit vectors.
"""
function normalize_meshes!(lv_meshes, glass_meshes)
    # Pass 1: compute global bounding box across all meshes
    lo = Vec3f(Inf, Inf, Inf)
    hi = Vec3f(-Inf, -Inf, -Inf)
    for dict in (lv_meshes, glass_meshes)
        for (_, (meshes, _, _)) in dict
            for m in meshes
                for p in GeometryBasics.coordinates(m)
                    lo = Vec3f(min(lo[1], p[1]), min(lo[2], p[2]), min(lo[3], p[3]))
                    hi = Vec3f(max(hi[1], p[1]), max(hi[2], p[2]), max(hi[3], p[3]))
                end
            end
        end
    end

    extent = hi - lo
    max_extent = max(extent[1], extent[2], extent[3])
    if max_extent < 1f-6
        @warn "All meshes are degenerate, skipping normalization"
        return
    end

    scale = 10f0 / max_extent
    center = Point3f(((lo + hi) * 0.5f0)...)
    println("Normalizing meshes: bbox [$lo, $hi], center=$center, scale=$scale")

    # Pass 2: rescale all vertex positions in-place
    for dict in (lv_meshes, glass_meshes)
        for (_, (meshes, _, _)) in dict
            for (mi, m) in enumerate(meshes)
                old_pts = GeometryBasics.coordinates(m)
                new_pts = [Point3f(((Vec3f(p...) - Vec3f(center...)) * scale)...) for p in old_pts]
                # Rebuild mesh with new positions but same normals and faces
                meshes[mi] = GeometryBasics.Mesh(new_pts, GeometryBasics.faces(m);
                                                  normal=m.normal)
            end
        end
    end
end

# ============================================================================
# Material mapping from detector colors to Hikari materials
# ============================================================================

function color_to_material(color::Tuple{RGB, Float64})
    rgb = color[1]
    r, g, b = Float32(red(rgb)), Float32(green(rgb)), Float32(blue(rgb))

    # Map specific colors to appropriate materials
    if r > 0.8 && g < 0.4 && b < 0.4  # Orangered (Fe)
        return Hikari.Copper(roughness=0.01f0)
    elseif r < 0.3 && g < 0.3 && b > 0.7  # Blue (Cu)
        return Hikari.CoatedDiffuseMaterial(reflectance=(r, g, b), roughness=0.15f0, eta=1.5f0)
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
    return Hikari.Glass()
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
