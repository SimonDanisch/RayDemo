# physics.jl
#
# Position-based-dynamics step for the sand-cat demo.  Single GPU compute
# kernel, one thread per grain.  Inline ray queries against the HWTLAS resolve
# both grain-mesh (cullMask=0x01) and grain-grain (cullMask=0x02) collisions.
#
# Note on the SPIR-V emitter: the per-function `OpVariable` for the ray query
# is allocated lazily on the first `lava_ray_query_init` call, but its
# declaration is only injected into the entry block's preamble.  Any
# conditional branch (bounds-checked array load, ternary, `if`, ...) on the
# straight-line path before that first call diverts the init into a non-entry
# block; the emitter then fails to inject the OpVariable and SPIR-V validation
# errors with "ID has not been defined".  Until that emitter limitation is
# resolved, this kernel keeps the entry block branch-free up to (and
# including) the first ray query: all bounds checks are masked with
# `@inbounds`, no `?:`, no `if`, and the launch must be sized so every thread
# corresponds to a valid grain (ndrange == n_grains).

using Lava
using Raycore
using GeometryBasics: Point3f, Vec3f
import LinearAlgebra: dot

# Kernel-side constants.
const GRAVITY        = Vec3f(0f0, -9.81f0, 0f0)
const RESTITUTION    = 0.05f0
const MU_FRICTION    = 0.7f0
const VEL_DAMPING    = 0.985f0
const PBD_ITERATIONS = 4

"""
    pbd_step_kernel(positions_in, velocities_in, positions_out, velocities_out,
                    tri_normals, floor_y, x_min, x_max, z_min, z_max,
                    radius, dt, n_grains)

One thread per grain.  Reads `positions_in[i]` and `velocities_in[i]`, writes
the post-step values into `positions_out[i]` and `velocities_out[i]`.
`tri_normals[k]` holds the geometric normal of cat triangle `k` (1-indexed).
"""
function pbd_step_kernel(
        positions_in::Lava.LavaDeviceArray{Point3f, 1},
        velocities_in::Lava.LavaDeviceArray{Vec3f, 1},
        positions_out::Lava.LavaDeviceArray{Point3f, 1},
        velocities_out::Lava.LavaDeviceArray{Vec3f, 1},
        tri_normals::Lava.LavaDeviceArray{Vec3f, 1},
        floor_y::Float32,
        x_min::Float32,
        x_max::Float32,
        z_min::Float32,
        z_max::Float32,
        radius::Float32,
        dt::Float32,
        n_grains::UInt32)

    i = Int(Lava.lava_global_invocation_id_x()) + 1
    @inbounds p_in = positions_in[i]
    @inbounds v_in = velocities_in[i]

    # ---------------- 1. Symplectic Euler prediction ----------------
    v_pred = Vec3f(v_in[1], v_in[2] + GRAVITY[2] * dt, v_in[3])
    dx     = Vec3f(v_pred[1] * dt, v_pred[2] * dt, v_pred[3] * dt)
    travel2 = dx[1] * dx[1] + dx[2] * dx[2] + dx[3] * dx[3]
    travel  = sqrt(travel2 + 1f-30)   # +eps to keep sqrt > 0

    # Direction is dx/travel; with the +1f-30 guard travel is always > 0 so
    # the division is unconditional.
    inv_t  = 1f0 / travel
    dnorm  = Vec3f(dx[1] * inv_t, dx[2] * inv_t, dx[3] * inv_t)

    # ---------------- 2. Trajectory ray cast (cat, mask 0x01) ----------------
    # First ray query: must lie on the entry block's straight-line path, see
    # the file-level note.
    ray_cat = Raycore.Ray(o=p_in, d=dnorm,
                           t_min=0f0, t_max=travel + radius)
    Lava.lava_ray_query_init(ray_cat; mask=UInt32(0x01))
    while Lava.lava_ray_query_proceed()
        Lava.lava_ray_query_confirm()
    end
    kind_cat = Lava.lava_ray_query_get_type(true)
    t_cat = Lava.lava_ray_query_get_t(true)
    prim_cat = Lava.lava_ray_query_get_primitive_index(true)

    hit_cat = kind_cat == UInt32(1)
    # Stop short of the surface by `radius`.  Use Float32 max() to avoid
    # introducing a branch.
    t_safe   = t_cat - radius
    t_safe   = max(t_safe, 0f0)
    # Look up the real per-triangle geometric normal.  The primitive index is
    # only meaningful when we actually hit the cat; clamp to 1 otherwise so
    # the unused load never reads out-of-bounds.  Flip the normal so it points
    # back into the grain (against the incoming ray).
    n_tri = UInt32(length(tri_normals))
    prim_safe = hit_cat ? (prim_cat + UInt32(1)) : UInt32(1)
    prim_safe = min(prim_safe, n_tri)
    @inbounds n_face_raw = tri_normals[Int(prim_safe)]
    facing = n_face_raw[1] * dnorm[1] + n_face_raw[2] * dnorm[2] + n_face_raw[3] * dnorm[3]
    flip = facing > 0f0 ? -1f0 : 1f0
    n_face = Vec3f(n_face_raw[1] * flip, n_face_raw[2] * flip, n_face_raw[3] * flip)
    vn       = v_pred[1] * n_face[1] + v_pred[2] * n_face[2] + v_pred[3] * n_face[3]
    v_tangent = Vec3f(v_pred[1] - vn * n_face[1],
                      v_pred[2] - vn * n_face[2],
                      v_pred[3] - vn * n_face[3])
    v_after_hit = Vec3f(v_tangent[1] * (1f0 - MU_FRICTION) + n_face[1] * (-RESTITUTION * vn),
                        v_tangent[2] * (1f0 - MU_FRICTION) + n_face[2] * (-RESTITUTION * vn),
                        v_tangent[3] * (1f0 - MU_FRICTION) + n_face[3] * (-RESTITUTION * vn))

    # Branchless select between ballistic step and stop-at-surface step.
    sel_hit = hit_cat ? 1f0 : 0f0
    adv = sel_hit * t_safe + (1f0 - sel_hit) * travel
    p = Point3f(p_in[1] + dnorm[1] * adv,
                p_in[2] + dnorm[2] * adv,
                p_in[3] + dnorm[3] * adv)
    v_pred = Vec3f(sel_hit * v_after_hit[1] + (1f0 - sel_hit) * v_pred[1],
                   sel_hit * v_after_hit[2] + (1f0 - sel_hit) * v_pred[2],
                   sel_hit * v_after_hit[3] + (1f0 - sel_hit) * v_pred[3])

    # ---------------- 3. Walls + floor (branchless clamp) ----------------
    px = clamp(p[1], x_min + radius, x_max - radius)
    py = max(p[2], floor_y + radius)
    pz = clamp(p[3], z_min + radius, z_max - radius)
    # Velocity damping on contact: project v_pred[2] up if pushed off floor.
    vy_damp_floor = py > floor_y + radius * 1.001f0 ? v_pred[2] : max(v_pred[2], -RESTITUTION * v_pred[2])
    p = Point3f(px, py, pz)
    v_pred = Vec3f(v_pred[1], vy_damp_floor, v_pred[3])

    # ---------------- 4. Grain-grain push-apart (mask 0x02) ----------------
    two_r  = 2f0 * radius
    two_r2 = two_r * two_r
    @inbounds for _iter in UInt32(1):UInt32(PBD_ITERATIONS)
        ray_g = Raycore.Ray(o=p, d=Vec3f(1f0, 0f0, 0f0),
                             t_min=0f0, t_max=0f0)
        Lava.lava_ray_query_init(ray_g; mask=UInt32(0x02))
        while Lava.lava_ray_query_proceed()
            kind = Lava.lava_ray_query_get_type(false)
            if kind == UInt32(1)
                j = Int(Lava.lava_ray_query_get_primitive_index(false)) + 1
                if j != i && j >= 1 && j <= Int(n_grains)
                    q = positions_in[j]
                    dxij = p[1] - q[1]
                    dyij = p[2] - q[2]
                    dzij = p[3] - q[3]
                    d2 = dxij * dxij + dyij * dyij + dzij * dzij
                    if d2 < two_r2 && d2 > 1f-12
                        d = sqrt(d2)
                        overlap = (two_r - d) * 0.5f0
                        inv_d = 1f0 / d
                        nx = dxij * inv_d
                        ny = dyij * inv_d
                        nz = dzij * inv_d
                        p = Point3f(p[1] + nx * overlap,
                                    p[2] + ny * overlap,
                                    p[3] + nz * overlap)
                        vn2 = v_pred[1] * nx + v_pred[2] * ny + v_pred[3] * nz
                        if vn2 < 0f0
                            # 1) Bounce: reverse normal component with restitution.
                            v_pred = Vec3f(v_pred[1] - vn2 * nx * (1f0 + RESTITUTION),
                                           v_pred[2] - vn2 * ny * (1f0 + RESTITUTION),
                                           v_pred[3] - vn2 * nz * (1f0 + RESTITUTION))
                            # 2) Static-ish friction: zero out the tangential
                            #    velocity entirely on contact.  Coulomb-style
                            #    kinetic friction (just damping tangential by a
                            #    fraction) lets grains keep sliding indefinitely
                            #    under any small push, so piles flatten into
                            #    pancakes.  Killing tangential velocity outright
                            #    on contact pins the grain in place once it
                            #    settles - approximates a high static-friction
                            #    coefficient and gives an angle of repose.
                            vn_after = v_pred[1] * nx + v_pred[2] * ny + v_pred[3] * nz
                            v_pred = Vec3f(vn_after * nx,
                                           vn_after * ny,
                                           vn_after * nz)
                        end
                    end
                end
            end
        end
    end

    # ---------------- 5. Velocity damping ----------------
    v_pred = Vec3f(v_pred[1] * VEL_DAMPING,
                   v_pred[2] * VEL_DAMPING,
                   v_pred[3] * VEL_DAMPING)

    @inbounds positions_out[i]  = p
    @inbounds velocities_out[i] = v_pred
    return nothing
end
