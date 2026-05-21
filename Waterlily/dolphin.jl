using WaterLily, StaticArrays, BiotSavartBCs, WaterLilyMeshBodies, Random
using Lava, KernelAbstractions
using GLMakie, RayMakie, Hikari, Raycore
using GeometryBasics, FileIO, Colors, ImageCore
using LinearAlgebra: norm, cross, normalize
using JLD2, Meshing

folder = joinpath(dirname(pathof(WaterLilyMeshBodies)), "..", "example")

# ============================================================================
# Simulation
# ============================================================================

"""
Animated dolphin sim. Body mesh undulates via a time-dependent `map(x, t)` that
adds a z-displacement sin-wave whose amplitude grows toward the tail. Returns
`(sim, inverse_map)` — `inverse_map(body_vertex, t)` gives the world-frame
position at time `t`, used when saving per-step deformed meshes for rendering.

Everything is Float32. A T kwarg parameterization would make the `map` closure
capture `Type{Float32}` (non-isbits) and the resulting kernel would fail to
compile on GPU — easier to just hardcode Float32.
"""
function dolphin(; L=192, Re=1e6, U=1, A=0.1, St=0.3, k=5.3, mem=Array)
    mesh_path = joinpath(folder, "LowPolyDolphin.stl")
    probe = MeshBody(mesh_path)
    lo, up = probe.bvh.nodes[1].lo, probe.bvh.nodes[1].up
    scale = Float32(L / maximum(up .- lo))
    center = scale * SVector{3}(lo .+ up) / 2

    # Pre-compute all motion constants as isbits scalars — captured by the
    # map closure, which runs on GPU (needs isbits captures only).
    #
    # ω: the physical Strouhal formula π*St*U/a gives ω ≈ 0.05 rad/tU, so one
    # tail beat takes ~130 tU — invisibly slow over our 9-tU video window.
    # Override for visualization: higher ω for visible tail motion. CFL in the
    # flow solver scales with peak tail velocity (a*ω), so we can't push ω too
    # high without the sim slowing by ~peak_vel/U. ω=0.5 gives ~0.7 beats over
    # the 9-tU window (one full "left → right → mostly back" swing), and peak
    # tail velocity a*ω = 9.6 units/tU ≈ 10× the freestream — the sim stays
    # stable and regen takes ~100 min instead of ~5 hours at ω=2.
    a  = Float32(L * A)
    ω  = 0.5f0
    kv = Float32(k)
    Lh = Float32(L ÷ 2)
    Lf = Float32(L)
    shift = SA{Float32}[L÷4, L÷2+5, L÷4] - center   # SVector{3,Float32}

    function map(x, t)
        x_sh = x - shift
        sx = clamp((x_sh[2] + Lh) / Lf, 0f0, 1f0)
        amp = (1f0 + 9f0 * sx^3) / 10f0
        x_sh + a * amp * sin(kv * sx - ω * t) * SA[0f0, 0f0, 1f0]
    end

    domain_size = (L ÷ 2, 3L ÷ 2, L ÷ 2)
    sim = BiotSimulation(domain_size, (0, U, 0), L; ν=U*L/Re, mem, T=Float32,
        body = MeshBody(mesh_path; scale, map, boundary=true, mem, size=domain_size))

    # Inverse map: body-frame mesh vertex v (scaled STL coords) → world-frame
    # position at time t. Closed-form because the z-wave in `map` depends only
    # on x_sh[2] (via `sx`), not x_sh[3] — so given v we know x_sh[2] = v[2]
    # and can solve for x_sh[3] directly.
    function inverse_map(v, t)
        sx = clamp((v[2] + Lh) / Lf, 0f0, 1f0)
        amp_val = (1f0 + 9f0 * sx^3) / 10f0
        wave = a * amp_val * sin(kv * sx - ω * t)
        SA{Float32}[v[1] + shift[1], v[2] + shift[2], v[3] - wave + shift[3]]
    end

    return sim, inverse_map
end

# ============================================================================
# Time-series step loading -- reads from the L=256 smoke sim cache.
# The cache is generated separately by `generate_dolphin_smoke.jl` (200 frames
# spanning sim_t = 1..3 at dt = 0.01).  Per-frame JLD2 holds u, sdf, vort.
# Derived fields (body_mesh from marching-cubes, dye from compute_pathlines_dye)
# are computed lazily and cached to a sister `_panels_cache/` dir to avoid
# re-running the ~10s pathline integration on every render.
# ============================================================================

const STEPS_DIR        = joinpath(@__DIR__, "dolphin_smoke_L256_steps")
const PANELS_CACHE_DIR = joinpath(@__DIR__, "dolphin_smoke_L256_panels_cache")

"""
Build a triangle-list body mesh from a body-SDF via marching cubes.  The smoke
sim stores the SDF (signed distance to the dolphin surface, in grid units) per
frame, so this is the cheap reconstruction of the deformed body each frame.
Each triangle is a 3×3 SMatrix (cols = vertex coords), matching the shape
`dolphin_body_mesh` expects from `data.body_mesh`.
"""
function body_mesh_from_sdf(sdf::Array{Float32,3})
    ranges = range.((1f0, 1f0, 1f0), Float32.(size(sdf)))
    pts, fcs, _ = Meshing.isosurface_normals(sdf,
        Meshing.MarchingCubes(iso=0f0, reduceverts=true), ranges...)
    tris = Vector{SMatrix{3,3,Float32,9}}(undef, length(fcs))
    @inbounds for (k, f) in enumerate(fcs)
        v1 = pts[f[1]]; v2 = pts[f[2]]; v3 = pts[f[3]]
        tris[k] = @SMatrix Float32[
            v1[1] v2[1] v3[1];
            v1[2] v2[2] v3[2];
            v1[3] v2[3] v3[3];
        ]
    end
    return tris
end

"""
Lazy load+cache for the per-frame derived panel fields (body_mesh, dye).
The smoke sim cache holds `u, sdf, vort`; everything else is reconstructed and
stashed in `PANELS_CACHE_DIR` so subsequent renders skip the ~10s dye work.
"""
function load_step(step::Int)
    smoke_path = joinpath(STEPS_DIR, "frame_$(lpad(step, 4, '0')).jld2")
    cache_path = joinpath(PANELS_CACHE_DIR, "panels_$(lpad(step, 4, '0')).jld2")
    isdir(PANELS_CACHE_DIR) || mkpath(PANELS_CACHE_DIR)

    # Smoke sim fields — copy out of mmap before the file closes.
    u, t = jldopen(smoke_path, "r") do f
        (copy(f["u"]), Float32(f["sim_t"]))
    end

    # Derived fields — load if cached, otherwise compute + persist.
    if isfile(cache_path)
        body_mesh, dye_r, dye_g, dye_b, dye_lo, dye_hi = jldopen(cache_path, "r") do f
            (copy(f["body_mesh"]),
             copy(f["dye_r"]), copy(f["dye_g"]), copy(f["dye_b"]),
             f["dye_lo"], f["dye_hi"])
        end
    else
        sdf = jldopen(smoke_path, "r") do f; copy(f["sdf"]); end
        body_mesh = body_mesh_from_sdf(sdf)
        data_for_dye = (u=u, U_mag=1f0, body_mesh=body_mesh, body_scale=1f0)
        bnds = body_bounds(data_for_dye)
        sx, sy, sz = size(u)[1:3]
        bcx, bcy, bcz = Float32(sx/2), Float32(sy/2), Float32(sz/2)
        dye_r, dye_g, dye_b, dye_lo, dye_hi =
            compute_pathlines_dye(data_for_dye, bnds, bcx, bcy, bcz)
        jldsave(cache_path; body_mesh, dye_r, dye_g, dye_b, dye_lo, dye_hi)
    end

    return (u=u, U_mag=1f0, body_mesh=body_mesh, body_scale=1f0,
            dye_r=dye_r, dye_g=dye_g, dye_b=dye_b,
            dye_lo=dye_lo, dye_hi=dye_hi, t=t)
end

# ============================================================================
# Mesh & velocity helpers
# ============================================================================

function dolphin_body_mesh(data)
    tris = data.body_mesh   # per-step world-frame deformed triangles (3×3 columns=vertices)
    n = length(tris)

    # Merge coincident vertices for smooth normals. Key on the raw vertex
    # coords (rounded to 1e-3 units) so shared edges share a vertex index and
    # normal_mesh can compute area-weighted normals.
    unique_verts = Point3f[]
    vert_map = Dict{Tuple{Int,Int,Int}, Int}()
    indices = Vector{Int}(undef, n * 3)
    for (i, tri) in enumerate(tris)
        for j in 1:3
            v = Point3f(tri[1,j], tri[2,j], tri[3,j])
            key = (round(Int, tri[1,j] * 1000), round(Int, tri[2,j] * 1000), round(Int, tri[3,j] * 1000))
            idx = get!(vert_map, key) do
                push!(unique_verts, v)
                length(unique_verts)
            end
            indices[(i-1)*3 + j] = idx
        end
    end

    faces = [GLTriangleFace(indices[(i-1)*3+1], indices[(i-1)*3+2], indices[(i-1)*3+3]) for i in 1:n]
    return GeometryBasics.normal_mesh(GeometryBasics.Mesh(unique_verts, faces))
end

"""
Callable struct for the perturbation-velocity sampler. Stably typed (fields
are concrete `Array{Float32,4}` + `Float32`) so `update!(streamplot; arg1=vf)`
across frames doesn't change the closure type — Makie's ComputePipeline locks
arg types at plot creation, so anonymous closures break animation.

Subtypes `Function` so `streamplot!`'s `convert_arguments(::Type{<:StreamPlot},
f::Function, ...)` dispatches; without it Makie raises "Result needs to have
same length" because the recipe falls through to a generic conversion path.
"""
mutable struct VelocityField <: Function
    u::Array{Float32,4}
    U_free::Float32
end

function (vf::VelocityField)(pos)
    u = vf.u
    sx, sy, sz, _ = size(u)
    x, y, z = Float32(pos[1]), Float32(pos[2]), Float32(pos[3])
    (x < 1 || x >= sx || y < 1 || y >= sy || z < 1 || z >= sz) && return Vec3f(0, 0, 0)
    ix = clamp(floor(Int, x), 1, sx-1)
    iy = clamp(floor(Int, y), 1, sy-1)
    iz = clamp(floor(Int, z), 1, sz-1)
    fx, fy, fz = x - ix, y - iy, z - iz
    v = Vec3f(0, 0, 0)
    @inbounds for d in 1:3
        c000 = u[ix,iy,iz,d]; c100 = u[ix+1,iy,iz,d]
        c010 = u[ix,iy+1,iz,d]; c110 = u[ix+1,iy+1,iz,d]
        c001 = u[ix,iy,iz+1,d]; c101 = u[ix+1,iy,iz+1,d]
        c011 = u[ix,iy+1,iz+1,d]; c111 = u[ix+1,iy+1,iz+1,d]
        val = c000*(1-fx)*(1-fy)*(1-fz) + c100*fx*(1-fy)*(1-fz) +
              c010*(1-fx)*fy*(1-fz) + c110*fx*fy*(1-fz) +
              c001*(1-fx)*(1-fy)*fz + c101*fx*(1-fy)*fz +
              c011*(1-fx)*fy*fz + c111*fx*fy*fz
        adj = d == 2 ? val - vf.U_free : val
        v = setindex(v, adj, d)
    end
    return v
end

function make_velocity_func(data; subtract_freestream=true)
    U_free = subtract_freestream ? Float32(data.U_mag) : 0f0
    VelocityField(Array{Float32,4}(data.u), U_free)
end

"""
    compute_vorticity(data) -> vc

Full-domain vorticity magnitude. Used by the smoke panels.
"""
function compute_vorticity(data)
    u = data.u
    sx, sy, sz, _ = size(u)
    vc = zeros(Float32, sx, sy, sz)
    @inbounds for k in 2:sz-1, j in 2:sy-1, i in 2:sx-1
        wx = (u[i,j+1,k,3]-u[i,j-1,k,3])*0.5f0 - (u[i,j,k+1,2]-u[i,j,k-1,2])*0.5f0
        wy = (u[i,j,k+1,1]-u[i,j,k-1,1])*0.5f0 - (u[i+1,j,k,3]-u[i-1,j,k,3])*0.5f0
        wz = (u[i+1,j,k,2]-u[i-1,j,k,2])*0.5f0 - (u[i,j+1,k,1]-u[i,j-1,k,1])*0.5f0
        vc[i,j,k] = sqrt(wx*wx + wy*wy + wz*wz)
    end
    # sqrt + clamp to compress dynamic range (cores are 100x stronger than wake)
    return clamp.(sqrt.(vc), 0f0, 0.15f0)
end

"""
    extend_narrow_band_sdf(sdf, max_dist) -> ext

WaterLily stores the SDF as a narrow band clamped to ±1 voxel — outside that
band, every cell reads the same saturation value and carries no distance
information. For body-masking the volumetric smoke we need distances up to
`body_margin` voxels, so we extend the band by `max_dist` 6-neighbour dilation
passes. Each pass propagates d → d+1 into cells still at the saturation
ceiling, producing an approximate chebyshev distance field up to `max_dist+1`
voxels from the body.
"""
function extend_narrow_band_sdf(sdf::Array{Float32,3}, max_dist::Integer)
    nx, ny, nz = size(sdf)
    ceil_val = 0.99f0
    # Saturated cells start at +Inf so every propagation step is an improvement;
    # cells the dilation never reaches stay +Inf and the mask will clamp them
    # to "full smoke" naturally.
    ext = map(v -> v >= ceil_val ? Inf32 : v, sdf)
    scratch = similar(ext)
    for _ in 1:max_dist
        copyto!(scratch, ext)
        @inbounds for k in 1:nz, j in 1:ny, i in 1:nx
            best = scratch[i,j,k]
            if i > 1;  c = scratch[i-1,j,k] + 1f0; c < best && (best = c); end
            if i < nx; c = scratch[i+1,j,k] + 1f0; c < best && (best = c); end
            if j > 1;  c = scratch[i,j-1,k] + 1f0; c < best && (best = c); end
            if j < ny; c = scratch[i,j+1,k] + 1f0; c < best && (best = c); end
            if k > 1;  c = scratch[i,j,k-1] + 1f0; c < best && (best = c); end
            if k < nz; c = scratch[i,j,k+1] + 1f0; c < best && (best = c); end
            ext[i,j,k] = best
        end
    end
    return ext
end

"""
    compute_vorticity(data, body_margin; transition=2) -> vc

Full-domain vorticity magnitude with an SDF-derived body-exclusion mask.
Voxels within `body_margin` of the body surface (per `data.sdf`, extended
from the sim's narrow-band via `extend_narrow_band_sdf`) are forced to zero,
with a `transition`-voxel linear ramp outside that margin before full
vorticity is used. This prevents bright shear-layer smoke from occluding the
dolphin when rendered as a volumetric medium.
"""
function compute_vorticity(data, body_margin::Real; transition::Real=2)
    u::Array{Float32,4} = data.u
    sdf::Array{Float32,3} = data.sdf
    sx, sy, sz, _ = size(u)
    # WaterLily: `u` carries 1 ghost voxel on each face; `sdf` lives on the
    # inner grid only. u[i,j,k] at i∈[2..sx-1] maps to sdf[i-1, j-1, k-1].
    size(sdf) == (sx-2, sy-2, sz-2) ||
        error("sdf dims $(size(sdf)) don't match u inner dims ($(sx-2),$(sy-2),$(sz-2))")
    vc = zeros(Float32, sx, sy, sz)
    bm = Float32(body_margin)
    tr = Float32(max(transition, 1e-4))
    # Extend the clamped narrow band so sdf values up to ~bm+tr voxels from the
    # body are meaningful.
    sdf_ext = extend_narrow_band_sdf(sdf, Int(ceil(bm + tr)))
    @inbounds for k in 2:sz-1, j in 2:sy-1, i in 2:sx-1
        wx = (u[i,j+1,k,3]-u[i,j-1,k,3])*0.5f0 - (u[i,j,k+1,2]-u[i,j,k-1,2])*0.5f0
        wy = (u[i,j,k+1,1]-u[i,j,k-1,1])*0.5f0 - (u[i+1,j,k,3]-u[i-1,j,k,3])*0.5f0
        wz = (u[i+1,j,k,2]-u[i-1,j,k,2])*0.5f0 - (u[i,j+1,k,1]-u[i,j-1,k,1])*0.5f0
        mag = sqrt(wx*wx + wy*wy + wz*wz)
        mask = clamp((sdf_ext[i-1,j-1,k-1] - bm) / tr, 0f0, 1f0)
        vc[i,j,k] = mag * mask
    end
    return clamp.(sqrt.(vc), 0f0, 0.15f0)
end

function body_bounds(data)
    tris = data.body_mesh  # already in world frame
    bx = [tris[i][1,j] for i in eachindex(tris) for j in 1:3]
    by = [tris[i][2,j] for i in eachindex(tris) for j in 1:3]
    bz = [tris[i][3,j] for i in eachindex(tris) for j in 1:3]
    return extrema(bx), extrema(by), extrema(bz)
end

# ============================================================================
# Panel builders
# ============================================================================

function set_panel_cam!(ax::LScene, cam)
    update_cam!(ax.scene, cam.eye, cam.lookat, cam.up)
    cc = cameracontrols(ax.scene)
    cc.fov[] = Float64(cam.fov)
    update_cam!(ax.scene, cc)
    println("  panel cam set: eye=", cam.eye, " fov=", cc.fov[])
    return cc
end

function panel_streamlines!(ax::LScene, dmesh, vfunc, bounds, cam, domain_size)
    (bxr, byr, bzr) = bounds
    sx, sy, sz = domain_size
    xm = max(2f0, bxr[1]-15f0); xM = min(Float32(sx-1), bxr[2]+15f0)
    ym = max(2f0, byr[1]-20f0); yM = min(Float32(sy-1), byr[2]+100f0)
    zm = max(2f0, bzr[1]-15f0); zM = min(Float32(sz-1), bzr[2]+15f0)
    dolphin = mesh!(ax, dmesh; material=Hikari.CoatedDiffuse(
        reflectance=(0.30f0,0.35f0,0.45f0), roughness=0.02f0, eta=1.5f0))
    # High-contrast velocity coloring: sqrt-stretch + explicit colorrange + turbo
    stream = streamplot!(ax, vfunc, xm..xM, ym..yM, zm..zM;
        gridsize=(14, 24, 14), stepsize=0.35f0, maxsteps=1000, density=1.2,
        use_tubes=true, tube_radius=0.11f0, tube_n_sides=8,
        tube_spline=true, tube_spline_resolution=4,
        color=dx -> sqrt(norm(dx)),
        colorrange=(0.05f0, 0.55f0),
        colormap=:turbo, arrow_size=0)
    set_panel_cam!(ax, cam)
    return (dolphin=dolphin, stream=stream)
end

function panel_blue_volume!(ax::LScene, dmesh, vc, cam)
    nx, ny, nz = size(vc)
    dolphin = mesh!(ax, dmesh; material=Hikari.CoatedDiffuse(
        reflectance=(0.30f0,0.35f0,0.45f0), roughness=0.02f0, eta=1.5f0))
    vol = volume!(ax, 1f0..Float32(nx), 1f0..Float32(ny), 1f0..Float32(nz), vc;
        colormap=[RGBA(0,0,0,0), RGBA(0,0,0,0), RGBA(0.01,0.02,0.08,0.001), RGBA(0.03,0.08,0.25,0.002),
                  RGBA(0.08,0.2,0.55,0.005), RGBA(0.15,0.38,0.8,0.01), RGBA(0.25,0.55,0.92,0.02),
                  RGBA(0.4,0.72,0.98,0.035), RGBA(0.6,0.88,1.0,0.06)],
        colorrange=(0.015f0,0.13f0),
        material=(extinction_scale=1.5f0, asymmetry_g=0.6f0, single_scatter_albedo=0.5f0))
    set_panel_cam!(ax, cam)
    return (dolphin=dolphin, vol=vol)
end

function panel_warm_hero!(ax::LScene, dmesh, vc, cam)
    nx, ny, nz = size(vc)
    dolphin = mesh!(ax, dmesh; material=Hikari.CoatedDiffuse(
        reflectance=(0.25f0,0.30f0,0.40f0), roughness=0.02f0, eta=1.5f0))
    vol = volume!(ax, 1f0..Float32(nx), 1f0..Float32(ny), 1f0..Float32(nz), vc;
        colormap=[RGBA(0,0,0,0), RGBA(0,0,0,0), RGBA(0.03,0,0.08,0.001), RGBA(0.15,0.01,0.3,0.003),
                  RGBA(0.45,0.04,0.15,0.007), RGBA(0.75,0.12,0.03,0.012), RGBA(0.95,0.28,0.04,0.025),
                  RGBA(1.0,0.55,0.12,0.045), RGBA(1.0,0.78,0.32,0.07)],
        colorrange=(0.015f0,0.13f0),
        material=(extinction_scale=1.5f0, asymmetry_g=0.4f0, single_scatter_albedo=0.5f0))
    set_panel_cam!(ax, cam)
    return (dolphin=dolphin, vol=vol)
end

"""
Compute RGB dye volumes by advecting colored particles from 10 upstream emitters.
Each emitter has a distinct color; particles leave colored trails as they flow past the body.
"""
function compute_pathlines_dye(data, bounds, bcx, bcy, bcz; n_per=0, n_steps=350, dt=0.5f0)
    u = data.u
    sx, sy, sz = size(u)[1:3]
    (bxr, byr, bzr) = bounds

    # PERTURBATION velocity (freestream subtracted) -- same as streamplot uses.
    # Key insight: streamplot NORMALIZES the velocity vector before stepping.
    # This makes streamlines visible even with tiny perturbations (uniform arc-length).
    U_free = data.U_mag
    function vel_pert(pos)
        x, y, z = Float32(pos[1]), Float32(pos[2]), Float32(pos[3])
        (x<1||x>=sx||y<1||y>=sy||z<1||z>=sz) && return Vec3f(0, 0, 0)
        ix = clamp(floor(Int, x), 1, sx-1); iy = clamp(floor(Int, y), 1, sy-1); iz = clamp(floor(Int, z), 1, sz-1)
        fx = x - ix; fy = y - iy; fz = z - iz
        v = Vec3f(0, 0, 0)
        @inbounds for d in 1:3
            c000=u[ix,iy,iz,d]; c100=u[ix+1,iy,iz,d]; c010=u[ix,iy+1,iz,d]; c110=u[ix+1,iy+1,iz,d]
            c001=u[ix,iy,iz+1,d]; c101=u[ix+1,iy,iz+1,d]; c011=u[ix,iy+1,iz+1,d]; c111=u[ix+1,iy+1,iz+1,d]
            val = c000*(1-fx)*(1-fy)*(1-fz)+c100*fx*(1-fy)*(1-fz)+c010*(1-fx)*fy*(1-fz)+c110*fx*fy*(1-fz)+
                  c001*(1-fx)*(1-fy)*fz+c101*fx*(1-fy)*fz+c011*(1-fx)*fy*fz+c111*fx*fy*fz
            adj = d == 2 ? val - U_free : val
            v = setindex(v, adj, d)
        end
        return v
    end
    # Normalized advection: step is arc-length dt in direction of velocity
    function step_dir(p, dt)
        v = vel_pert(p)
        mag = sqrt(v[1]^2 + v[2]^2 + v[3]^2)
        mag < 1f-8 && return p  # stationary
        return p + dt * v / mag
    end

    colors = [
        RGB{Float32}(1.0, 0.15, 0.15), RGB{Float32}(1.0, 0.55, 0.0),
        RGB{Float32}(0.2, 1.0, 0.25),  RGB{Float32}(0.1, 0.95, 0.8),
        RGB{Float32}(1.0, 0.85, 0.1),  RGB{Float32}(0.15, 0.6, 1.0),
        RGB{Float32}(1.0, 0.3, 0.75),  RGB{Float32}(0.45, 0.25, 1.0),
        RGB{Float32}(1.0, 1.0, 1.0),   RGB{Float32}(0.0, 1.0, 0.5),
    ]
    n_clusters = length(colors)

    # Even sparser -- user wants fewer, more transparent trails
    seed_gx, seed_gy, seed_gz = 3, 6, 3
    seed_xs = range(bxr[1] - 8f0, bxr[2] + 8f0, length=seed_gx)
    seed_ys = range(byr[1] - 4f0, byr[2] + 20f0, length=seed_gy)
    seed_zs = range(bzr[1] - 8f0, bzr[2] + 8f0, length=seed_gz)

    Random.seed!(42)
    seeds = Vec3f[]
    seed_cid = Int[]
    for iz in 1:seed_gz, iy in 1:seed_gy, ix in 1:seed_gx
        hx = ix > seed_gx÷2 ? 1 : 0
        hy = iy > seed_gy÷2 ? 1 : 0
        hz = iz > seed_gz÷2 ? 1 : 0
        cluster = 1 + hx + 2*hy + 4*hz
        cluster = ((cluster - 1) % n_clusters) + 1
        push!(seeds, Vec3f(Float32(seed_xs[ix]), Float32(seed_ys[iy]), Float32(seed_zs[iz])))
        push!(seed_cid, cluster)
    end
    total_seeds = length(seeds)
    println("  dye seeds: $total_seeds, steps: $n_steps")

    # Lower-res dye grid + wider Gaussian splat -> soft trails (no straws)
    dye_res = (140, 280, 140)
    dye_lo = Vec3f(Float32(bxr[1]-15f0), Float32(byr[1]-5f0), Float32(bzr[1]-15f0))
    dye_hi = Vec3f(Float32(bxr[2]+15f0), Float32(byr[2]+90f0), Float32(bzr[2]+15f0))
    dye_span = dye_hi - dye_lo
    dr = zeros(Float32, dye_res...); dg = zeros(Float32, dye_res...); db = zeros(Float32, dye_res...)

    # Wider Gaussian splat -> very soft falloff (no hard edges)
    splat_radius = 4
    splat_sigma = 2.2f0
    inv2σ² = 1f0 / (2f0 * splat_sigma * splat_sigma)

    function deposit_gauss!(dr, dg, db, p, c, lo, span, res)
        gx = (p[1]-lo[1])/span[1]*(res[1]-1)+1
        gy = (p[2]-lo[2])/span[2]*(res[2]-1)+1
        gz = (p[3]-lo[3])/span[3]*(res[3]-1)+1
        (gx<1f0||gx>Float32(res[1])||gy<1f0||gy>Float32(res[2])||gz<1f0||gz>Float32(res[3])) && return
        cx = round(Int, gx); cy = round(Int, gy); cz = round(Int, gz)
        @inbounds for dk in -splat_radius:splat_radius, dj in -splat_radius:splat_radius, di in -splat_radius:splat_radius
            ix = cx+di; iy = cy+dj; iz = cz+dk
            (ix<1||ix>res[1]||iy<1||iy>res[2]||iz<1||iz>res[3]) && continue
            d2 = Float32((ix-gx)^2 + (iy-gy)^2 + (iz-gz)^2)
            w = exp(-d2 * inv2σ²)
            dr[ix,iy,iz] += w * c.r
            dg[ix,iy,iz] += w * c.g
            db[ix,iy,iz] += w * c.b
        end
    end

    # Forward + backward integration from each seed -- the full streamline trajectory
    @inbounds for d in (1f0, -1f0)
        particles_buf = copy(seeds)
        for step in 1:n_steps
            for i in 1:total_seeds
                p = particles_buf[i]
                if p[1]<1f0||p[1]>Float32(sx-2)||p[2]<1f0||p[2]>Float32(sy-2)||p[3]<1f0||p[3]>Float32(sz-2)
                    continue
                end
                new_p = step_dir(p, d*dt)
                particles_buf[i] = new_p
                deposit_gauss!(dr, dg, db, new_p, colors[seed_cid[i]], dye_lo, dye_span, dye_res)
            end
        end
    end

    # Log-compress and normalize -- soft tone curve
    dr_log = log.(1f0 .+ dr); dg_log = log.(1f0 .+ dg); db_log = log.(1f0 .+ db)
    max_v = max(maximum(dr_log), maximum(dg_log), maximum(db_log))
    return dr_log./max_v, dg_log./max_v, db_log./max_v, dye_lo, dye_hi
end

function panel_pathlines_dye!(ax::LScene, dmesh, dye_r, dye_g, dye_b, dye_lo, dye_hi, cam)
    dolphin = mesh!(ax, dmesh; material=Hikari.CoatedDiffuse(
        reflectance=(0.30f0,0.35f0,0.45f0), roughness=0.02f0, eta=1.5f0))

    # Per-voxel medium: subtle water everywhere + soft cubic dye on top.
    # Cubic falloff keeps bright cores dim and fades tails to near-transparent -> soft plumes.
    nx, ny, nz = size(dye_r)
    σ_s_grid = Array{Hikari.RGBSpectrum, 3}(undef, nx, ny, nz)
    σ_a_grid = Array{Hikari.RGBSpectrum, 3}(undef, nx, ny, nz)

    # Clear water: only a whisper of blue-tinted absorption, zero uniform scatter
    # so non-dye regions stay transparent. All visible scatter comes from dye.
    water_σa = Hikari.RGBSpectrum(0.004f0, 0.002f0, 0.0003f0)
    water_σs = Hikari.RGBSpectrum(0.001f0, 0.001f0, 0.001f0)

    α_peak = 0.4f0
    # Higher exponent -> fades faster, only bright cores show, tails near-transparent
    chan(d) = α_peak * d^2.2f0
    @inbounds for i in eachindex(dye_r)
        σ_s_grid[i] = Hikari.RGBSpectrum(
            water_σs.c[1] + chan(dye_r[i]),
            water_σs.c[2] + chan(dye_g[i]),
            water_σs.c[3] + chan(dye_b[i]))
        σ_a_grid[i] = water_σa
    end
    bounds = Raycore.Bounds3(Point3f(dye_lo...), Point3f(dye_hi...))
    medium = Hikari.RGBGridMedium(
        σ_a_grid=σ_a_grid, σ_s_grid=σ_s_grid,
        sigma_scale=1.5f0, g=0.3f0,
        bounds=bounds, majorant_res=Vec{3, Int64}(16, 16, 16))
    cube = GeometryBasics.normal_mesh(Rect3f(Vec3f(dye_lo...), Vec3f(dye_hi .- dye_lo)))
    # Water-IOR dielectric gives Fresnel reflections at grazing angles + subtle
    # refraction, so the cube reads as a tank of water. Mild roughness softens glints.
    glass = Hikari.Dielectric(index=1.33f0, roughness=0.001f0)
    tank = mesh!(ax, cube; material=Hikari.MediumInterface(glass; inside=medium))

    set_panel_cam!(ax, cam)
    return (dolphin=dolphin, tank=tank)
end

# ============================================================================
# Run
# ============================================================================

# DEFERRED_FREE_WARN_THRESHOLD removed: per-queue deferred lists are
# timeline-gated, no warning needed.

function construct_scene(; panel_size=(960, 540), data=load_step(1))
    dmesh = dolphin_body_mesh(data)
    vfunc = make_velocity_func(data; subtract_freestream=true)
    println("Computing vorticity...")
    vc = compute_vorticity(data)
    bounds = body_bounds(data)
    # Grid midpoint — used as the scene's "body center" for light placement
    # and pathline seeding. Matches how `generate_steps` picks its bcx/bcy/bcz.
    sx, sy, sz, _ = size(data.u)
    bcx = Float32(sx / 2); bcy = Float32(sy / 2); bcz = Float32(sz / 2)

    # Manually-tuned cameras (eye, lookat, up, fov) for each panel
    cam1 = (eye=Vec3f(312.32, -296.85, 243.94), lookat=Vec3f(35.83, 136.88, 71.08),
            up=Vec3f(-0.17, 0.27, 0.95), fov=10.0)
    cam2 = (eye=Vec3f(420.31, -299.46, 380.2),  lookat=Vec3f(46.32, 137.85, 73.87),
            up=Vec3f(-0.31, 0.36, 0.88), fov=10.0)
    cam3 = (eye=Vec3f(843.27, -153.6, 238.31),  lookat=Vec3f(41.73, 138.05, 79.23),
            up=Vec3f(-0.17, 0.06, 0.98), fov=10.0)
    cam4 = (eye=Vec3f(-567.33, -245.06, 195.65), lookat=Vec3f(58.07, 126.62, 73.63),
            up=Vec3f(0.14, 0.08, 0.99), fov=10.0)

    RayMakie.activate!()
    Lava.clear_spirv_disk_cache!()

    domain_size = size(data.u)[1:3]
    pw, ph = panel_size

    bg = RGBf(0f0, 0f0, 0f0)
    fig = Figure(size=(2pw, 2ph); backgroundcolor=bg, figure_padding=0)

    # Per-panel light sets (each panel renders its part of the figure with its own scene)
    lights_streamlines = [
        Makie.PointLight(RGBf(15000,12000,8000), Vec3f(bcx-50,bcy-90,bcz+60)),
        Makie.PointLight(RGBf(3000,4000,6000),   Vec3f(bcx+60,bcy+30,bcz+30)),
        Makie.PointLight(RGBf(2000,2000,1500),   Vec3f(bcx,bcy+80,bcz+50)),
        Makie.PointLight(RGBf(1500,2000,3000),   Vec3f(bcx,bcy-20,bcz-30)),
    ]
    lights_blue = [
        Makie.PointLight(RGBf(8000,12000,18000), Vec3f(bcx-40,bcy-80,bcz+50)),
        Makie.PointLight(RGBf(2000,3000,5000),   Vec3f(bcx+50,bcy+30,bcz+30)),
        Makie.PointLight(RGBf(1000,1500,2500),   Vec3f(bcx,bcy-20,bcz-30)),
    ]
    lights_warm = [
        Makie.PointLight(RGBf(16000,12000,7000), Vec3f(bcx-50,bcy-90,bcz+60)),
        Makie.PointLight(RGBf(1500,2500,4000),   Vec3f(bcx+60,bcy+40,bcz+25)),
        Makie.PointLight(RGBf(1000,900,500),     Vec3f(bcx,bcy+100,bcz+50)),
        Makie.PointLight(RGBf(1000,1200,2000),   Vec3f(bcx,bcy-20,bcz-30)),
    ]
    # Cam4 looks from (-567,-245,195) toward the cube. Put a strong backlight
    # on the far side (high +X, +Y) so light travels THROUGH the dye toward the
    # camera -- classic volumetric backlight.
    lights_dye = [
        Makie.PointLight(RGBf(80000,70000,55000), Vec3f(bcx+350, bcy+380, bcz+80)),
        Makie.PointLight(RGBf(18000,22000,32000), Vec3f(bcx+200, bcy+150, bcz-60)),
        Makie.PointLight(RGBf(2500,3500,6000),   Vec3f(bcx-60, bcy-40, bcz+40)),
    ]

    ax1 = LScene(fig[1, 1]; show_axis=false,
                 scenekw=(lights=lights_streamlines, backgroundcolor=RGBf(0.03,0.05,0.14)))
    ax2 = LScene(fig[1, 2]; show_axis=false,
                 scenekw=(lights=lights_blue, backgroundcolor=RGBf(0.02,0.04,0.12)))
    ax3 = LScene(fig[2, 1]; show_axis=false,
                 scenekw=(lights=lights_warm, backgroundcolor=RGBf(0.012,0.02,0.05)))
    ax4 = LScene(fig[2, 2]; show_axis=false,
                 scenekw=(lights=lights_dye, backgroundcolor=RGBf(0.005,0.01,0.03)))
    colgap!(fig.layout, 0); rowgap!(fig.layout, 0)
    p1 = panel_streamlines!(ax1, dmesh, vfunc, bounds, cam1, domain_size)
    p2 = panel_blue_volume!(ax2, dmesh, vc, cam2)
    p3 = panel_warm_hero!(ax3, dmesh, vc, cam3)
    # Use precomputed dye from step file if present (avoids 10s/frame CPU work).
    dye_r, dye_g, dye_b, dye_lo, dye_hi = if haskey(data, :dye_r)
        data.dye_r, data.dye_g, data.dye_b, data.dye_lo, data.dye_hi
    else
        compute_pathlines_dye(data, bounds, bcx, bcy, bcz)
    end
    p4 = panel_pathlines_dye!(ax4, dmesh, dye_r, dye_g, dye_b, dye_lo, dye_hi, cam4)
    plots = (streamlines=p1, blue=p2, warm=p3, dye=p4)
    scene_data = (; data, dmesh, vfunc, vc, dye_r, dye_g, dye_b, dye_lo, dye_hi,
                    bounds, bcx, bcy, bcz, cam1, cam2, cam3, cam4)
    return fig, plots, scene_data
end

# ============================================================================
# Per-step scene update + short-video recording
# ============================================================================

"""
Build just the dye medium (no surface material) from RGB grids.
"""
function build_dye_medium_only(dye_r, dye_g, dye_b, dye_lo, dye_hi)
    nx, ny, nz = size(dye_r)
    σ_s_grid = Array{Hikari.RGBSpectrum, 3}(undef, nx, ny, nz)
    σ_a_grid = Array{Hikari.RGBSpectrum, 3}(undef, nx, ny, nz)
    water_σa = Hikari.RGBSpectrum(0.004f0, 0.002f0, 0.0003f0)
    water_σs = Hikari.RGBSpectrum(0.001f0, 0.001f0, 0.001f0)
    α_peak = 0.4f0
    chan(d) = α_peak * d^2.2f0
    @inbounds for i in eachindex(dye_r)
        σ_s_grid[i] = Hikari.RGBSpectrum(
            water_σs.c[1] + chan(dye_r[i]),
            water_σs.c[2] + chan(dye_g[i]),
            water_σs.c[3] + chan(dye_b[i]))
        σ_a_grid[i] = water_σa
    end
    bnds3 = Raycore.Bounds3(Point3f(dye_lo...), Point3f(dye_hi...))
    return Hikari.RGBGridMedium(σ_a_grid=σ_a_grid, σ_s_grid=σ_s_grid,
        sigma_scale=1.5f0, g=0.3f0, bounds=bnds3, majorant_res=Vec{3, Int64}(16, 16, 16))
end

"""
Full MediumInterface (glass + medium) for the initial scene construction.
"""
function build_dye_medium(dye_r, dye_g, dye_b, dye_lo, dye_hi)
    medium = build_dye_medium_only(dye_r, dye_g, dye_b, dye_lo, dye_hi)
    glass = Hikari.Dielectric(index=1.33f0, roughness=0.001f0)
    return Hikari.MediumInterface(glass; inside=medium)
end

"""
Mutate the scene in place to reflect one simulation step's data. All four
primitives get updated: streamplot (via VelocityField mutation), both volumes
(new vc arrays), and the dye cube (new MediumInterface material).
"""
function update_scene!(plots, sd, step)
    # 1. Velocity sampler: mutate u in place, keep the same VelocityField instance
    sd.vfunc.u .= step.u
    Makie.update!(plots.streamlines.stream; arg1=sd.vfunc)

    # 2. Swap the deformed dolphin mesh into all four panel overlays. Each
    # sim step holds its own world-frame body_mesh (animated undulation).
    new_dmesh = dolphin_body_mesh(step)
    Makie.update!(plots.streamlines.dolphin; arg1=new_dmesh)
    Makie.update!(plots.blue.dolphin;        arg1=new_dmesh)
    Makie.update!(plots.warm.dolphin;        arg1=new_dmesh)
    Makie.update!(plots.dye.dolphin;         arg1=new_dmesh)

    # 3. Re-compute vorticity from new u + current body mesh; re-upload
    # to both volume plots.
    data_like = (u=step.u, U_mag=sd.vfunc.U_free,
                 body_mesh=step.body_mesh, body_scale=step.body_scale)
    new_vc = compute_vorticity(data_like)
    Makie.update!(plots.blue.vol; arg4=new_vc)
    Makie.update!(plots.warm.vol; arg4=new_vc)

    # 4. Dye material: in-place swap on the existing scene handle (the mesh
    # plot recipe listens for :material and routes through update_material!).
    mat = build_dye_medium(step.dye_r, step.dye_g, step.dye_b, step.dye_lo, step.dye_hi)
    Makie.update!(plots.dye.tank; material=mat)
    return nothing
end

"""
Record a short low-quality video that samples the simulation steps and exercises
every primitive's update path. Uses `Makie.record_longrunning` so partial runs
are resumable (frames land in `_frames/` next to the output).
"""
function short_video(; n_frames=15, spp=5, panel_size=(320, 180),
                       step_stride=max(1, 200 ÷ n_frames),
                       outpath=joinpath(@__DIR__, "dolphin_short.mp4"))
    integrator = Hikari.VolPath(; samples=spp, max_depth=4, hw_accel=true)
    RayMakie.activate!(; integrator, tonemap=:aces, gamma=2.2f0, exposure=2.0f0,
                       denoise=false)

    step1 = load_step(1)
    fig, plots, sd = construct_scene(; panel_size, data=step1)

    Makie.record_longrunning(fig, outpath, 1:n_frames; framerate=10) do i
        step_idx = 1 + (i - 1) * step_stride
        println("frame $i/$n_frames  step $step_idx")
        step = load_step(step_idx)
        update_scene!(plots, sd, step)
    end
    println("saved: ", outpath)
    return outpath
end

"""
HQ video: one sim step per video frame.  At default settings the four panels
combine to a 1920×1080 figure — same pixel/spp budget per frame as the single-
panel rainbow_glow HQ render, so the per-frame wallclock is comparable.
The smoke sim cache has 200 frames → 6.67 s @ 30 fps.
"""
function hq_video(; n_frames=200, spp=1000, panel_size=(960, 540),
                    outpath=joinpath(@__DIR__, "dolphin_hq.mp4"))
    integrator = Hikari.VolPath(; samples=spp, max_depth=8, hw_accel=true)
    RayMakie.activate!(; integrator, tonemap=:aces, gamma=2.2f0, exposure=2.0f0,
                       denoise=false)

    step1 = load_step(1)
    fig, plots, sd = construct_scene(; panel_size, data=step1)

    # Figure out the lowest missing frame so re-runs don't re-do update_scene!
    # work for frames whose PNG already exists.  Each sim step is independent
    # (load_step(i) reads its own data), so jumping straight to the first
    # missing frame is correct — construct_scene already bootstrapped us to
    # step 1, and the first un-cached frame's update_scene! transitions to
    # whatever state it needs.
    frame_dir = joinpath(@__DIR__, splitext(basename(outpath))[1] * "_frames")
    nd = max(4, length(string(n_frames)))
    first_missing = let m = n_frames + 1
        for i in 1:n_frames
            if !isfile(joinpath(frame_dir, "frame_$(lpad(i, nd, '0')).png"))
                m = i; break
            end
        end
        m
    end
    println("resume-aware hq_video: first_missing=$first_missing  (will skip update_scene! for i<$first_missing)")

    Makie.record_longrunning(fig, outpath, 1:n_frames; framerate=30, update=false) do i
        i < first_missing && return  # cached — record_longrunning skips colorbuffer
        step = load_step(i)
        update_scene!(plots, sd, step)
        # Telemetry: GPU-live bytes + queue lengths + slab counts (leak indicators).
        gpu_mib = Lava.GPU_LIVE_BYTES[] >> 20
        bq = Lava.vk_context().default_bq
        n_def = length(bq.deferred_frees)
        n_asdef = length(bq.deferred_as_frees)
        n_arg_slabs = length(bq.arg_slabs)
        n_ind_slabs = length(bq.indirect_slabs)
        n_inflight = length(bq.in_flight)
        n_blas = -1
        n_inst = -1
        try
            scr = Makie.getscreen(fig.scene)
            if scr !== nothing
                hs = scr.state.hikari_scene
                hwt = hs.accel.hwtlas
                n_blas = length(hwt.blas_list)
                n_inst = length(hwt.instance_blas_indices)
            end
        catch
        end
        println("frame $i/$n_frames step $i gpu=$(gpu_mib)MiB def=$(n_def) asdef=$(n_asdef) argslabs=$(n_arg_slabs) indslabs=$(n_ind_slabs) inflight=$(n_inflight) blas=$(n_blas) inst=$(n_inst)")
        flush(stdout)
    end
    println("saved: ", outpath)
    return outpath
end


# short_video(; outpath=joinpath(@__DIR__, "shorty.mp4"))

# begin
#     spp = 300
#     max_depth = 8
#     integrator = Hikari.VolPath(; samples=spp, max_depth=max_depth, hw_accel=true)
#     # New units after denoise.jl fix:
#     #   sigma_color = log2 luminance diff (stops) — HDR-correct
#     #   sigma_depth = relative depth tolerance (fraction of local depth per step)
#     # 1 iteration only -- a single 5x5 pass. With the asymmetric color weight
#     # this still suseries_01_step0001.pngppresses 1-px fireflies, but without the multi-iteration
#     # radius growth that causes à-trous's watercolour/posterization look.
#     denoise_config = Hikari.DenoiseConfig(
#         iterations=1,
#         sigma_color=0.35f0,   # tighter: ~1.27× luminance ratio at e^-1 weight
#         sigma_normal=96f0,    # tighter: respect fine surface variation
#         sigma_depth=0.03f0,   # 3% relative depth tolerance
#         use_variance=false,
#     )
#     RayMakie.activate!(; integrator, tonemap=:aces, gamma=2.2f0, exposure=2.0f0,
#                        denoise=true, denoise_config)
#     fig, plots, scene_data = construct_scene(panel_size=(960, 540))
#     # display(fig; update=false)
#     nothing
# end;

# begin
#     img = @time colorbuffer(fig; backend=RayMakie, update=false)
#     outpath = joinpath(@__DIR__, "dolphin_4panel_preview20-noise.png")
#     save(outpath, img)
# end
