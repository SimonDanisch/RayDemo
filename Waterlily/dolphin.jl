# Dolphin CFD + Ray Tracing 4-Panel Visualization
# Simulate a dolphin swimming with WaterLily (BiotSavart FMM) on GPU via Lava,
# then raytrace 4 visualization panels:
#   1. Streamline tubes (perturbation velocity)
#   2. Blue vorticity volume (classical CFD)
#   3. Dark underwater with warm vorticity glow (hero shot)
#   4. Combined streamlines + vorticity volume

using WaterLily, StaticArrays, BiotSavartBCs, WaterLilyMeshBodies, Random
using Lava, KernelAbstractions
using GLMakie, RayMakie, Hikari, Raycore
using GeometryBasics, FileIO, Colors, ImageCore
using LinearAlgebra: norm, cross, normalize
using JLD2

folder = joinpath(dirname(pathof(WaterLilyMeshBodies)), "..", "example")
const CACHE_FILE = joinpath(@__DIR__, "dolphin_sim_long.jld2")

# ============================================================================
# Simulation
# ============================================================================

function dolphin(; scale=3f0, Re=1e5, U=1, mem=Array)
    L = round(Int, 64 * scale)
    x₀ = SA[L÷4, 4+L÷2, 2L÷5]
    body = MeshBody(joinpath(folder, "LowPolyDolphin.stl"); scale, map=(x, t) -> x - x₀, mem)
    BiotSimulation((L ÷ 2, 3L ÷ 2, 3L ÷ 4), (0, U, 0), L; body, T=Float32, ν=U * L / Re, mem)
end

function save_sim(sim, path=CACHE_FILE)
    u_cpu = Array(sim.sim.flow.u)
    body_mesh_cpu = Array(sim.body.mesh)
    jldsave(path;
        u = u_cpu,
        U_mag = Float32(sim.sim.U),
        body_mesh = body_mesh_cpu,
        body_scale = Float32(sim.body.scale),
    )
    println("Saved simulation to $path ($(filesize(path) ÷ 1024^2) MiB)")
end

function load_sim(path=CACHE_FILE)
    f = jldopen(path, "r")
    data = (u = f["u"], U_mag = f["U_mag"], body_mesh = f["body_mesh"], body_scale = f["body_scale"])
    close(f)
    println("Loaded simulation from $path")
    return data
end

# ============================================================================
# Mesh & velocity helpers
# ============================================================================

function dolphin_body_mesh(data)
    tris = data.body_mesh
    sc = data.body_scale
    L = round(Int, 64 * sc)
    x₀ = SA{Float32}[L÷4, 4+L÷2, 2L÷5]
    n = length(tris)

    # Collect all triangle vertices
    all_verts = Vector{Point3f}(undef, n * 3)
    for (i, tri) in enumerate(tris)
        for j in 1:3
            all_verts[(i-1)*3 + j] = Point3f(tri[1,j] + x₀[1], tri[2,j] + x₀[2], tri[3,j] + x₀[3])
        end
    end

    # Merge coincident vertices for smooth normals
    # Use raw STL coordinates for the key (before offset, to avoid precision loss)
    unique_verts = Point3f[]
    vert_map = Dict{Tuple{Int,Int,Int}, Int}()
    indices = Vector{Int}(undef, n * 3)
    for (i, tri) in enumerate(tris)
        for j in 1:3
            key = (round(Int, tri[1,j] * 1000), round(Int, tri[2,j] * 1000), round(Int, tri[3,j] * 1000))
            idx = get!(vert_map, key) do
                push!(unique_verts, all_verts[(i-1)*3 + j])
                length(unique_verts)
            end
            indices[(i-1)*3 + j] = idx
        end
    end

    faces = [GLTriangleFace(indices[(i-1)*3+1], indices[(i-1)*3+2], indices[(i-1)*3+3]) for i in 1:n]

    # normal_mesh on a mesh with shared vertices will compute smooth (area-weighted) normals
    return GeometryBasics.normal_mesh(GeometryBasics.Mesh(unique_verts, faces))
end

function make_velocity_func(data; subtract_freestream=true)
    u = data.u
    sx, sy, sz, _ = size(u)
    U_free = subtract_freestream ? data.U_mag : 0f0
    function (pos)
        x, y, z = Float32(pos[1]), Float32(pos[2]), Float32(pos[3])
        (x < 1 || x >= sx || y < 1 || y >= sy || z < 1 || z >= sz) && return Vec3f(0, 0, 0)
        ix = clamp(floor(Int, x), 1, sx-1)
        iy = clamp(floor(Int, y), 1, sy-1)
        iz = clamp(floor(Int, z), 1, sz-1)
        fx, fy, fz = x - ix, y - iy, z - iz
        v = Vec3f(0, 0, 0)
        for d in 1:3
            c000 = u[ix,iy,iz,d]; c100 = u[ix+1,iy,iz,d]
            c010 = u[ix,iy+1,iz,d]; c110 = u[ix+1,iy+1,iz,d]
            c001 = u[ix,iy,iz+1,d]; c101 = u[ix+1,iy,iz+1,d]
            c011 = u[ix,iy+1,iz+1,d]; c111 = u[ix+1,iy+1,iz+1,d]
            val = c000*(1-fx)*(1-fy)*(1-fz) + c100*fx*(1-fy)*(1-fz) +
                  c010*(1-fx)*fy*(1-fz) + c110*fx*fy*(1-fz) +
                  c001*(1-fx)*(1-fy)*fz + c101*fx*(1-fy)*fz +
                  c011*(1-fx)*fy*fz + c111*fx*fy*fz
            adj = d == 2 ? val - U_free : val
            v = setindex(v, adj, d)
        end
        return v
    end
end

function compute_vorticity_crop(data, body_margin=12, wake_extent=90)
    u = data.u; tris = data.body_mesh
    sx, sy, sz, _ = size(u)
    sc = data.body_scale; L = round(Int, 64*sc)
    x₀ = SA{Float32}[L÷4, 4+L÷2, 2L÷5]
    bx = [tris[i][1,j]+x₀[1] for i in eachindex(tris) for j in 1:3]
    by = [tris[i][2,j]+x₀[2] for i in eachindex(tris) for j in 1:3]
    bz = [tris[i][3,j]+x₀[3] for i in eachindex(tris) for j in 1:3]

    i1 = max(2, round(Int, minimum(bx) - body_margin))
    i2 = min(sx-1, round(Int, maximum(bx) + body_margin))
    j1 = max(2, round(Int, minimum(by) - 8))
    j2 = min(sy-1, round(Int, maximum(by) + wake_extent))
    k1 = max(2, round(Int, minimum(bz) - body_margin))
    k2 = min(sz-1, round(Int, maximum(bz) + body_margin))

    vc = zeros(Float32, i2-i1+1, j2-j1+1, k2-k1+1)
    @inbounds for kk in 1:size(vc,3), jj in 1:size(vc,2), ii in 1:size(vc,1)
        i, j, k = ii+i1-1, jj+j1-1, kk+k1-1
        wx = (u[i,j+1,k,3]-u[i,j-1,k,3])*0.5f0 - (u[i,j,k+1,2]-u[i,j,k-1,2])*0.5f0
        wy = (u[i,j,k+1,1]-u[i,j,k-1,1])*0.5f0 - (u[i+1,j,k,3]-u[i-1,j,k,3])*0.5f0
        wz = (u[i+1,j,k,2]-u[i-1,j,k,2])*0.5f0 - (u[i,j+1,k,1]-u[i,j-1,k,1])*0.5f0
        vc[ii,jj,kk] = sqrt(wx^2 + wy^2 + wz^2)
    end
    # sqrt + clamp to compress dynamic range (cores are 100x stronger than wake)
    vc_proc = clamp.(sqrt.(vc), 0f0, 0.15f0)
    return vc_proc, (i1, i2, j1, j2, k1, k2)
end

function body_bounds(data)
    tris = data.body_mesh
    sc = data.body_scale; L = round(Int, 64*sc)
    x₀ = SA{Float32}[L÷4, 4+L÷2, 2L÷5]
    bx = [tris[i][1,j]+x₀[1] for i in eachindex(tris) for j in 1:3]
    by = [tris[i][2,j]+x₀[2] for i in eachindex(tris) for j in 1:3]
    bz = [tris[i][3,j]+x₀[3] for i in eachindex(tris) for j in 1:3]
    return extrema(bx), extrema(by), extrema(bz)
end

# ============================================================================
# Panel builders
# ============================================================================

function panel_streamlines!(ax::LScene, dmesh, vfunc, bounds, cam, domain_size; zoom=10.0)
    (bxr, byr, bzr) = bounds
    sx, sy, sz = domain_size
    xm = max(2f0, bxr[1]-15f0); xM = min(Float32(sx-1), bxr[2]+15f0)
    ym = max(2f0, byr[1]-20f0); yM = min(Float32(sy-1), byr[2]+100f0)
    zm = max(2f0, bzr[1]-15f0); zM = min(Float32(sz-1), bzr[2]+15f0)
    mesh!(ax, dmesh; material=Hikari.CoatedDiffuse(
        reflectance=(0.30f0,0.35f0,0.45f0), roughness=0.02f0, eta=1.5f0))
    streamplot!(ax, vfunc, xm..xM, ym..yM, zm..zM;
        gridsize=(14, 24, 14), stepsize=0.35f0, maxsteps=1000, density=1.2,
        use_tubes=true, tube_radius=0.11f0, tube_n_sides=8,
        tube_spline=true, tube_spline_resolution=4,
        color=dx -> log10(max(1f-4, norm(dx))) + 4f0,
        colormap=:inferno, arrow_size=0)
    update_cam!(ax.scene, cam..., Vec3f(0,0,1))
    cc = cameracontrols(ax.scene)
    println("  panel fov: before=", cc.fov[], " requested=", zoom)
    cc.fov[] = Float64(zoom)
    update_cam!(ax.scene, cc)
    println("  panel fov: after=", cc.fov[], " proj[1,1]=", ax.scene.camera.projection[][1,1])
    return ax
end

function panel_blue_volume!(ax::LScene, dmesh, vc, crop, cam; zoom=10.0)
    i1, i2, j1, j2, k1, k2 = crop
    mesh!(ax, dmesh; material=Hikari.CoatedDiffuse(
        reflectance=(0.30f0,0.35f0,0.45f0), roughness=0.02f0, eta=1.5f0))
    volume!(ax, Float32(i1)..Float32(i2), Float32(j1)..Float32(j2), Float32(k1)..Float32(k2), vc;
        colormap=[RGBA(0,0,0,0), RGBA(0,0,0,0), RGBA(0.01,0.02,0.08,0.001), RGBA(0.03,0.08,0.25,0.002),
                  RGBA(0.08,0.2,0.55,0.005), RGBA(0.15,0.38,0.8,0.01), RGBA(0.25,0.55,0.92,0.02),
                  RGBA(0.4,0.72,0.98,0.035), RGBA(0.6,0.88,1.0,0.06)],
        colorrange=(0.015f0,0.13f0),
        material=(extinction_scale=1.5f0, asymmetry_g=0.6f0, single_scatter_albedo=0.5f0))
    update_cam!(ax.scene, cam..., Vec3f(0,0,1))
    cc = cameracontrols(ax.scene)
    println("  panel fov: before=", cc.fov[], " requested=", zoom)
    cc.fov[] = Float64(zoom)
    update_cam!(ax.scene, cc)
    println("  panel fov: after=", cc.fov[], " proj[1,1]=", ax.scene.camera.projection[][1,1])
    return ax
end

function panel_warm_hero!(ax::LScene, dmesh, vc, crop, bcx, bcy, bcz; zoom=10.0)
    i1, i2, j1, j2, k1, k2 = crop
    mesh!(ax, dmesh; material=Hikari.CoatedDiffuse(
        reflectance=(0.25f0,0.30f0,0.40f0), roughness=0.02f0, eta=1.5f0))
    volume!(ax, Float32(i1)..Float32(i2), Float32(j1)..Float32(j2), Float32(k1)..Float32(k2), vc;
        colormap=[RGBA(0,0,0,0), RGBA(0,0,0,0), RGBA(0.03,0,0.08,0.001), RGBA(0.15,0.01,0.3,0.003),
                  RGBA(0.45,0.04,0.15,0.007), RGBA(0.75,0.12,0.03,0.012), RGBA(0.95,0.28,0.04,0.025),
                  RGBA(1.0,0.55,0.12,0.045), RGBA(1.0,0.78,0.32,0.07)],
        colorrange=(0.015f0,0.13f0),
        material=(extinction_scale=1.5f0, asymmetry_g=0.4f0, single_scatter_albedo=0.5f0))
    update_cam!(ax.scene, Vec3f(bcx+80, bcy-65, bcz+18), Vec3f(bcx-5, bcy+25, bcz), Vec3f(0,0,1))
    cc = cameracontrols(ax.scene)
    println("  panel fov: before=", cc.fov[], " requested=", zoom)
    cc.fov[] = Float64(zoom)
    update_cam!(ax.scene, cc)
    println("  panel fov: after=", cc.fov[], " proj[1,1]=", ax.scene.camera.projection[][1,1])
    return ax
end

"""
Compute RGB dye volumes by advecting colored particles from 10 upstream emitters.
Each emitter has a distinct color; particles leave colored trails as they flow past the body.
"""
function compute_pathlines_dye(data, bounds, bcx, bcy, bcz; n_per=0, n_steps=450, dt=0.5f0)
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

    # Grid-based seeding like streamplot! does.
    # Each seed is assigned a color by its octant position -> "10 clusters" of color.
    # Integrate forward AND backward with normalized perturbation velocity,
    # so each seed traces a full streamline trajectory (not just forward from upstream).
    colors = [
        RGB{Float32}(1.0, 0.15, 0.15), RGB{Float32}(1.0, 0.55, 0.0),
        RGB{Float32}(0.2, 1.0, 0.25),  RGB{Float32}(0.1, 0.95, 0.8),
        RGB{Float32}(1.0, 0.85, 0.1),  RGB{Float32}(0.15, 0.6, 1.0),
        RGB{Float32}(1.0, 0.3, 0.75),  RGB{Float32}(0.45, 0.25, 1.0),
        RGB{Float32}(1.0, 1.0, 1.0),   RGB{Float32}(0.0, 1.0, 0.5),
    ]
    n_clusters = length(colors)

    # Seed positions: sparse grid covering body + wake region (fewer seeds → less crowded)
    seed_gx, seed_gy, seed_gz = 6, 12, 6
    seed_xs = range(bxr[1] - 12f0, bxr[2] + 12f0, length=seed_gx)
    seed_ys = range(byr[1] - 5f0, byr[2] + 30f0, length=seed_gy)
    seed_zs = range(bzr[1] - 12f0, bzr[2] + 12f0, length=seed_gz)

    Random.seed!(42)
    seeds = Vec3f[]
    seed_cid = Int[]
    for iz in 1:seed_gz, iy in 1:seed_gy, ix in 1:seed_gx
        # Assign cluster by octant
        hx = ix > seed_gx÷2 ? 1 : 0
        hy = iy > seed_gy÷2 ? 1 : 0
        hz = iz > seed_gz÷2 ? 1 : 0
        cluster = 1 + hx + 2*hy + 4*hz
        cluster = ((cluster - 1) % n_clusters) + 1
        push!(seeds, Vec3f(Float32(seed_xs[ix]), Float32(seed_ys[iy]), Float32(seed_zs[iz])))
        push!(seed_cid, cluster)
    end
    total_seeds = length(seeds)

    # Dye volume: high-res for sharp trails
    dye_res = (200, 400, 200)
    dye_lo = Vec3f(Float32(bxr[1]-15f0), Float32(byr[1]-5f0), Float32(bzr[1]-15f0))
    dye_hi = Vec3f(Float32(bxr[2]+15f0), Float32(byr[2]+90f0), Float32(bzr[2]+15f0))
    dye_span = dye_hi - dye_lo
    dr = zeros(Float32, dye_res...); dg = zeros(Float32, dye_res...); db = zeros(Float32, dye_res...)

    function deposit!(dr, dg, db, p, c, lo, span, res)
        gx = (p[1]-lo[1])/span[1]*(res[1]-1)+1
        gy = (p[2]-lo[2])/span[2]*(res[2]-1)+1
        gz = (p[3]-lo[3])/span[3]*(res[3]-1)+1
        (gx<1||gx>=res[1]||gy<1||gy>=res[2]||gz<1||gz>=res[3]) && return
        ix=floor(Int,gx); iy=floor(Int,gy); iz=floor(Int,gz)
        fx=gx-ix; fy=gy-iy; fz=gz-iz
        @inbounds for (di,wx) in ((0,1-fx),(1,fx)), (dj,wy) in ((0,1-fy),(1,fy)), (dk,wz) in ((0,1-fz),(1,fz))
            w = wx*wy*wz
            dr[ix+di,iy+dj,iz+dk] += w * c.r
            dg[ix+di,iy+dj,iz+dk] += w * c.g
            db[ix+di,iy+dj,iz+dk] += w * c.b
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
                deposit!(dr, dg, db, new_p, colors[seed_cid[i]], dye_lo, dye_span, dye_res)
            end
        end
    end

    # Log-compress and normalize
    dr_log = log.(1f0 .+ dr); dg_log = log.(1f0 .+ dg); db_log = log.(1f0 .+ db)
    max_v = max(maximum(dr_log), maximum(dg_log), maximum(db_log))
    return dr_log./max_v, dg_log./max_v, db_log./max_v, dye_lo, dye_hi
end

function panel_pathlines_dye!(ax::LScene, dmesh, dye_r, dye_g, dye_b, dye_lo, dye_hi, bcx, bcy, bcz; zoom=10.0)
    mesh!(ax, dmesh; material=Hikari.CoatedDiffuse(
        reflectance=(0.30f0,0.35f0,0.45f0), roughness=0.02f0, eta=1.5f0))

    # Build a single per-voxel RGB dye medium from the three channel grids.
    nx, ny, nz = size(dye_r)
    σ_s_grid = Array{Hikari.RGBSpectrum, 3}(undef, nx, ny, nz)
    σ_a_grid = fill(Hikari.RGBSpectrum(0f0), nx, ny, nz)
    α_peak = 0.7f0
    @inbounds for i in eachindex(dye_r)
        function chan(d)
            t = clamp((d - 0.05f0) / (0.85f0 - 0.05f0), 0f0, 1f0)
            d * (t * α_peak)
        end
        σ_s_grid[i] = Hikari.RGBSpectrum(chan(dye_r[i]), chan(dye_g[i]), chan(dye_b[i]))
    end
    bounds = Raycore.Bounds3(Point3f(dye_lo...), Point3f(dye_hi...))
    medium = Hikari.RGBGridMedium(
        σ_a_grid=σ_a_grid, σ_s_grid=σ_s_grid,
        sigma_scale=6f0, g=0.5f0,
        bounds=bounds, majorant_res=Vec{3, Int64}(16, 16, 16))
    cube = GeometryBasics.normal_mesh(Rect3f(Vec3f(dye_lo...), Vec3f(dye_hi .- dye_lo)))
    mesh!(ax, cube; material=Hikari.MediumInterface(Hikari.NullMaterial(); inside=medium))

    update_cam!(ax.scene, Vec3f(bcx+60, bcy-80, bcz+25), Vec3f(bcx, bcy+20, bcz), Vec3f(0,0,1))
    cc = cameracontrols(ax.scene)
    println("  panel fov: before=", cc.fov[], " requested=", zoom)
    cc.fov[] = Float64(zoom)
    update_cam!(ax.scene, cc)
    println("  panel fov: after=", cc.fov[], " proj[1,1]=", ax.scene.camera.projection[][1,1])
    return ax
end

# ============================================================================
# Run
# ============================================================================

Lava.DEFERRED_FREE_WARN_THRESHOLD[] = 100000

if !isfile(CACHE_FILE)
    Lava.clear_spirv_disk_cache!()
    println("Running dolphin simulation (scale=3, Re=1e5, tU/L=5)...")
    sim = dolphin(; scale=3f0, Re=1e5, mem=Lava.LavaArray)
    sim_step!(sim, 5; verbose=true, remeasure=false)
    save_sim(sim)
end

function render_all(; spp=512, panel_size=(960, 540))
    data = load_sim()
    dmesh = dolphin_body_mesh(data)
    vfunc = make_velocity_func(data; subtract_freestream=true)
    println("Computing vorticity...")
    vc, crop = compute_vorticity_crop(data)
    bounds = body_bounds(data)
    sc = data.body_scale; L = round(Int, 64*sc)
    x₀ = SA{Float32}[L÷4, 4+L÷2, 2L÷5]
    bcx, bcy, bcz = Float32.(x₀ .+ 1)

    cam_blue = (Vec3f(bcx+90, bcy-55, bcz+35), Vec3f(bcx-5, bcy+20, bcz-3))

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
    lights_dye = [
        Makie.PointLight(RGBf(15000,12000,8000), Vec3f(bcx-50,bcy-90,bcz+60)),
        Makie.PointLight(RGBf(3000,5000,8000),   Vec3f(bcx+60,bcy+30,bcz+30)),
        Makie.PointLight(RGBf(2000,2000,1500),   Vec3f(bcx,bcy+80,bcz+50)),
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

    println("Populating panels...")
    println("  Panel 1: Dense streamlines")
    panel_streamlines!(ax1, dmesh, vfunc, bounds, cam_blue, domain_size)
    println("  Panel 2: Blue volume")
    panel_blue_volume!(ax2, dmesh, vc, crop, cam_blue)
    println("  Panel 3: Warm hero")
    panel_warm_hero!(ax3, dmesh, vc, crop, bcx, bcy, bcz)
    println("  Panel 4: Colored pathline dye")
    dye_r, dye_g, dye_b, dye_lo, dye_hi = compute_pathlines_dye(data, bounds, bcx, bcy, bcz)
    panel_pathlines_dye!(ax4, dmesh, dye_r, dye_g, dye_b, dye_lo, dye_hi, bcx, bcy, bcz)

    println("Rendering figure at $(spp) spp...")
    integrator = Hikari.VolPath(; samples=spp, max_depth=10, hw_accel=true)
    img = colorbuffer(fig; backend=RayMakie, integrator=integrator,
                     tonemap=:aces, gamma=2.2f0, exposure=2.0f0)
    outpath = joinpath(@__DIR__, "dolphin_4panel_final.png")
    save(outpath, img)
    println("Saved: $outpath ($(size(img)))")
    return img
end

render_all(; spp=256)
