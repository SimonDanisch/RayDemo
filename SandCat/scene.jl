# scene.jl
#
# Scene setup helpers for the sand-cat demo.  Loads cat.obj, computes its
# bounding box, and generates a 3D grid of grain positions above the cat.

using GeometryBasics
using GeometryBasics: Point3f, Vec3f
using FileIO
using Makie
using LinearAlgebra: cross, norm

"""
    load_cat_mesh() -> GeometryBasics.Mesh

Load Makie's bundled `cat.obj`.  The mesh is in its native units
(approximately 1 m tall, 1.3 m long, 0.34 m wide).
"""
function load_cat_mesh()
    return FileIO.load(Makie.assetpath("cat.obj"))
end

"""
    cat_bounds(mesh) -> (min, max)

Axis-aligned bounding box of a mesh.
"""
function cat_bounds(mesh)
    verts = GeometryBasics.coordinates(mesh)
    mn = reduce((a, b) -> Point3f(min.(a, b)), verts; init=Point3f(Inf32))
    mx = reduce((a, b) -> Point3f(max.(a, b)), verts; init=Point3f(-Inf32))
    return mn, mx
end

"""
    spawn_grain_grid(; n_grains, region_min, region_max, jitter=0f0, rng=Random.GLOBAL_RNG)
        -> Vector{Point3f}

Place `n_grains` points on a regular 3D grid covering the AABB defined by
`region_min`..`region_max`.  Optional jitter perturbs each point by a uniform
random offset in `[-jitter, jitter]^3`.  Trailing grid cells beyond
`n_grains` are dropped.
"""
function spawn_grain_grid(; n_grains::Int,
                          region_min::Point3f,
                          region_max::Point3f,
                          jitter::Float32 = 0f0,
                          rng = Random.GLOBAL_RNG)
    side = ceil(Int, cbrt(n_grains))
    extent = region_max - region_min
    step = Vec3f(extent[1] / side, extent[2] / side, extent[3] / side)

    out = Vector{Point3f}(undef, n_grains)
    k = 0
    for ix in 0:side-1, iy in 0:side-1, iz in 0:side-1
        k += 1
        k > n_grains && break
        base = region_min + Point3f((ix + 0.5f0) * step[1],
                                    (iy + 0.5f0) * step[2],
                                    (iz + 0.5f0) * step[3])
        if jitter > 0f0
            j = Vec3f(2f0 * rand(rng, Float32) - 1f0,
                      2f0 * rand(rng, Float32) - 1f0,
                      2f0 * rand(rng, Float32) - 1f0) * jitter
            base += j
        end
        out[k] = base
    end
    return out
end

"""
    grain_aabbs(positions, radius) -> Vector{Lava.AABB}

Build per-grain AABBs around each `Point3f` position.  Half-extent is `2*radius`,
NOT `radius`: the kernel's grain-grain neighbor query is a degenerate ray
(`t_min = t_max = 0`) that hits an AABB iff the query origin lies inside it.
For a query at grain `i`'s centre to find a neighbor `j` whose physical contact
condition is `|p_i - p_j| < 2r`, neighbor `j`'s AABB must extend `2r` along each
axis around `p_j`.  Sizing AABBs to half-extent `r` would only catch neighbors
within `r` along each axis (Chebyshev distance), missing roughly half the
overlapping pairs and making PBD unable to resolve overlaps; piles collapse
into a thin pancake instead of stacking.
"""
function grain_aabbs(positions::AbstractVector{Point3f}, radius::Float32)
    # Use 2.05*r so floating-point boundary cases (where the kernel's d² check
    # accepts neighbours at exactly d=2r) are not lost to a half-open AABB
    # containment test in the rayQuery driver.
    h = Vec3f(2.05f0 * radius, 2.05f0 * radius, 2.05f0 * radius)
    return Lava.AABB[Lava.AABB(p - h, p + h) for p in positions]
end

"""
    triangle_normals(mesh) -> Vector{Vec3f}

Pre-compute geometric (face) normals for each triangle in the mesh, indexed
by primitive index (matches the order of `GeometryBasics.faces(mesh)`).  This
is what the GPU kernel looks up via `lava_ray_query_get_primitive_index` to
get a real surface normal for grain-cat collision response.
"""
function triangle_normals(mesh)
    coords = GeometryBasics.coordinates(mesh)
    fs = GeometryBasics.faces(mesh)
    out = Vector{Vec3f}(undef, length(fs))
    @inbounds for (i, face) in enumerate(fs)
        v1 = Vec3f(coords[face[1]])
        v2 = Vec3f(coords[face[2]])
        v3 = Vec3f(coords[face[3]])
        n  = cross(v2 - v1, v3 - v1)
        ln = norm(n)
        out[i] = ln > 1f-12 ? n / ln : Vec3f(0f0, 1f0, 0f0)
    end
    return out
end

"""
    point_inside_mesh(tlas, p; max_hits=64) -> Bool

Jordan-curve / parity test for "is point `p` inside a closed triangle mesh
behind `tlas`?".  Casts a `+y` ray from `p` and counts triangle hits by
iterating `Raycore.closest_hit`, advancing `t_min` past each hit until the
ray exits.  An odd hit count means the point is inside.

Raycore exposes `closest_hit` and `any_hit` but no all-hits enumerator, so we
walk hits manually.  `max_hits` is a safety cap; for the cat mesh the longest
chord crosses ~6 surfaces.
"""
function point_inside_mesh(tlas, p::Point3f; max_hits::Int = 64)::Bool
    dir = Vec3f(0f0, 1f0, 0f0)
    t_curr = 1f-5
    count = 0
    for _ in 1:max_hits
        ray = Raycore.Ray(o = p, d = dir, t_min = t_curr, t_max = Inf32)
        hit, _, t, _, _ = Raycore.closest_hit(tlas, ray)
        hit || break
        count += 1
        t_curr = t + 1f-4
    end
    return isodd(count)
end

"""
    sample_inside_mesh(tlas, bb_min, bb_max, n_target; oversample=8f0, seed=11)
        -> Vector{Point3f}

Rejection sample `n_target` points uniformly inside the closed mesh wrapped by
`tlas`.  Generates `n_target * oversample` random candidates inside the AABB,
parallelises the parity test across `Threads.maxthreadid()` threads, and keeps
the inside ones.  Returns at most `n_target` positions; if too few survived
rejection (rare, only for very thin meshes) the caller can retry with a higher
oversample.

This is what makes the demo a "cat made of sand": every grain spawns inside
the cat's volume, so the initial state IS a sand statue of the cat which
collapses under gravity rather than a cube of sand falling onto it.
"""
function sample_inside_mesh(tlas, bb_min::Point3f, bb_max::Point3f, n_target::Int;
                             oversample::Float32 = 8.0f0,
                             seed::Int = 11)
    extent = bb_max - bb_min
    ncand = round(Int, n_target * oversample)
    out = Vector{Point3f}(undef, ncand)
    inside_flag = Vector{Bool}(undef, ncand)
    rngs = [MersenneTwister(seed + tid) for tid in 1:Threads.maxthreadid()]
    Threads.@threads for i in 1:ncand
        rng = rngs[Threads.threadid()]
        p = Point3f(bb_min[1] + rand(rng, Float32) * extent[1],
                    bb_min[2] + rand(rng, Float32) * extent[2],
                    bb_min[3] + rand(rng, Float32) * extent[3])
        out[i] = p
        inside_flag[i] = point_inside_mesh(tlas, p)
    end
    inside = out[inside_flag]
    return length(inside) >= n_target ? inside[1:n_target] : inside
end
