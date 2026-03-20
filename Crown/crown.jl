# Crown Scene - pbrt-v4 VolPath Example
# Port of the crown scene from pbrt-v4-scenes with proper area lights and materials.
#
# This scene showcases:
# - MetalMaterial with gold (Au) properties for conductor materials
# - CoatedDiffuseMaterial for glossy painted surfaces
# - GlassMaterial for dielectric gems
# - MatteMaterial for diffuse surfaces
# - Area lights via Emissive material on meshes (5500K blackbody, scale=10)
# - DiffuseAreaLight created automatically per emissive triangle
#
# pbrt-v4 reference: crown.pbrt — 786 PLY meshes, 6 area light quads, ~3.5M triangles
#
# TODO: Load image textures from pbrt-data/textures/ for:
#   - Mix material masks (e.g. clamp_mask.png for gold/paint blending)
#   - Diffuse color maps (e.g. mitra_border_color.png)
#   - Bump maps (e.g. mitra_border_bump.png)
#   Hikari supports Texture(Matrix{Float32}) and Texture(Matrix{RGBSpectrum}).
#   The pbrt parser needs to extract "imagemap" texture definitions and load PNGs
#   via FileIO.load(), then pass them to material constructors.
#   Texture data is in pbrt-data/textures/ (14MB, already bundled).

include("../common/common.jl")
using GeometryBasics, StaticArrays, Colors

# Scene data lives alongside this script (geometry/*.ply, textures/, crown.pbrt)
const CROWN_DIR = @__DIR__
@assert isfile(joinpath(CROWN_DIR, "crown.pbrt")) "crown.pbrt not found in $CROWN_DIR"

# =============================================================================
# pbrt Parser — extracts materials, geometries, and area lights
# =============================================================================

"""Parse area light quads from crown.pbrt. Returns Vector of NamedTuples with vertices and light params."""
function parse_area_lights(content::String)
    lights = NamedTuple[]
    for m in eachmatch(r"AreaLightSource\s+\"diffuse\"\s+\"float scale\"\s*\[\s*([\d.e+-]+)\s*\]\s+\"blackbody L\"\s*\[\s*([\d.e+-]+)\s*\]\s+Shape\s+\"trianglemesh\"\s+\"integer indices\"\s*\[\s*([^\]]+)\]\s+\"normal N\"\s*\[\s*[^\]]+\]\s+\"point3 P\"\s*\[\s*([^\]]+)\]", content)
        scale = parse(Float32, m.captures[1])
        temperature = parse(Float32, m.captures[2])
        coords = parse.(Float32, split(strip(replace(m.captures[4], '\n' => ' '))))
        n_verts = length(coords) ÷ 3
        verts = [Point3f(coords[3i-2], coords[3i-1], coords[3i]) for i in 1:n_verts]
        idx_strs = split(strip(replace(m.captures[3], '\n' => ' ')))
        indices = parse.(Int, idx_strs) .+ 1  # 0-indexed → 1-indexed
        push!(lights, (vertices=verts, indices=indices, scale=scale, temperature=temperature))
    end
    return lights
end

function parse_crown_pbrt()
    content = read(joinpath(CROWN_DIR, "crown.pbrt"), String)
    materials = Dict{String, Dict{String,Any}}()
    media = Dict{String, Dict{String,Any}}()
    geometries = Vector{NamedTuple}()

    # Parse media definitions
    for m in eachmatch(r"MakeNamedMedium\s+\"([^\"]+)\"\s+([^\n]+(?:\n\s+[^\n]+)*)", content)
        name = m.captures[1]
        p = m.captures[2]
        params = Dict{String,Any}()
        fm = match(r"\"rgb sigma_a\"\s*\[\s*([\d.e+-]+)\s+([\d.e+-]+)\s+([\d.e+-]+)\s*\]", p)
        fm !== nothing && (params["sigma_a"] = (parse(Float32, fm.captures[1]), parse(Float32, fm.captures[2]), parse(Float32, fm.captures[3])))
        fm = match(r"\"rgb sigma_s\"\s*\[\s*([\d.e+-]+)\s+([\d.e+-]+)\s+([\d.e+-]+)\s*\]", p)
        fm !== nothing && (params["sigma_s"] = (parse(Float32, fm.captures[1]), parse(Float32, fm.captures[2]), parse(Float32, fm.captures[3])))
        media[name] = params
    end

    # Parse material definitions
    for m in eachmatch(r"MakeNamedMaterial\s+\"([^\"]+)\"\s+([^\n]+(?:\n\s+[^\n]+|\n#[^\n]+)*)", content)
        name = m.captures[1]
        params = Dict{String,Any}()
        p = m.captures[2]
        tm = match(r"\"string type\"\s*\[\s*\"([^\"]+)\"\s*\]", p)
        tm !== nothing && (params["type"] = tm.captures[1])
        for (k, pat) in [("roughness", r"\"float roughness\"\s*\[\s*([\d.e+-]+)\s*\]"),
                          ("uroughness", r"\"float uroughness\"\s*\[\s*([\d.e+-]+)\s*\]"),
                          ("vroughness", r"\"float vroughness\"\s*\[\s*([\d.e+-]+)\s*\]"),
                          ("eta", r"\"float eta\"\s*\[\s*([\d.e+-]+)\s*\]")]
            fm = match(pat, p)
            fm !== nothing && (params[k] = parse(Float32, fm.captures[1]))
        end
        fm = match(r"\"rgb reflectance\"\s*\[\s*([\d.e+-]+)\s+([\d.e+-]+)\s+([\d.e+-]+)\s*\]", p)
        fm !== nothing && (params["reflectance"] = (parse(Float32, fm.captures[1]), parse(Float32, fm.captures[2]), parse(Float32, fm.captures[3])))
        fm = match(r"\"spectrum eta\"\s*\[\s*([\d.e+-]+)\s+([\d.e+-]+)\s+([\d.e+-]+)\s+([\d.e+-]+)\s*\]", p)
        if fm !== nothing && !haskey(params, "eta")
            ior1 = parse(Float32, fm.captures[2])
            ior2 = parse(Float32, fm.captures[4])
            params["eta"] = (ior1 + ior2) / 2
        end
        contains(p, "metal-Au-eta") && (params["eta_spectrum"] = "gold")
        fm = match(r"\"string materials\"\s*\[\s*\"([^\"]+)\"\s+\"([^\"]+)\"\s*\]", p)
        fm !== nothing && (params["mix_materials"] = (fm.captures[1], fm.captures[2]))
        materials[name] = params
    end

    # Parse geometry
    current_material = nothing
    current_medium = nothing
    lines = split(content, '\n')
    for i in eachindex(lines)
        line = lines[i]
        m = match(r"NamedMaterial\s+\"([^\"]+)\"", line)
        m !== nothing && (current_material = m.captures[1])
        m = match(r"MediumInterface\s+\"([^\"]*)\"\s+\"([^\"]+)\"", line)
        m !== nothing && (current_medium = m.captures[2])
        if contains(line, "AttributeBegin")
            current_medium = nothing
        end
        if contains(line, "\"plymesh\"") && i + 1 <= length(lines)
            fm = match(r"\"string filename\"\s*\[\s*\"([^\"]+)\"\s*\]", lines[i+1])
            fm !== nothing && push!(geometries, (file=fm.captures[1], material=current_material, medium=current_medium))
        end
    end

    area_lights = parse_area_lights(content)
    return materials, media, geometries, area_lights
end

# =============================================================================
# Material Creation
# =============================================================================

function create_hikari_material(params::Dict, all_materials::Dict)
    t = get(params, "type", "")
    if t == "conductor"
        r = get(params, "roughness", 0.01f0)
        return Hikari.Gold(roughness=r)
    elseif t == "coateddiffuse"
        refl = get(params, "reflectance", (0.5f0, 0.5f0, 0.5f0))
        if haskey(params, "uroughness") || haskey(params, "vroughness")
            ur = get(params, "uroughness", 0.01f0)
            vr = get(params, "vroughness", 0.01f0)
            r = (ur, vr)
        else
            r = get(params, "roughness", 0.01f0)
        end
        return Hikari.CoatedDiffuseMaterial(reflectance=refl, roughness=r, eta=1.5f0)
    elseif t == "dielectric"
        return Hikari.GlassMaterial(index=get(params, "eta", 1.5f0))
    elseif t == "diffuse"
        kd = get(params, "reflectance", (0.5f0, 0.5f0, 0.5f0))
        return Hikari.MatteMaterial(Kd=Hikari.RGBSpectrum(kd[1], kd[2], kd[3]))
    elseif t == "mix"
        mix_names = get(params, "mix_materials", nothing)
        if mix_names !== nothing
            mat1_params = get(all_materials, mix_names[1], nothing)
            mat2_params = get(all_materials, mix_names[2], nothing)
            if mat1_params !== nothing && mat2_params !== nothing
                m1 = create_hikari_material(mat1_params, all_materials)
                m2 = create_hikari_material(mat2_params, all_materials)
                if m2 isa Hikari.ConductorMaterial
                    return m2
                elseif m1 isa Hikari.ConductorMaterial
                    return m1
                end
                return m1
            end
        end
        return Hikari.MatteMaterial(Kd=Hikari.RGBSpectrum(0.5f0))
    else
        return Hikari.MatteMaterial(Kd=Hikari.RGBSpectrum(0.5f0))
    end
end

function create_hikari_medium(params::Dict)
    σ_a = get(params, "sigma_a", (0f0, 0f0, 0f0))
    σ_s = get(params, "sigma_s", (0f0, 0f0, 0f0))
    return Hikari.HomogeneousMedium(
        σ_a=Hikari.RGBSpectrum(σ_a[1], σ_a[2], σ_a[3]),
        σ_s=Hikari.RGBSpectrum(σ_s[1], σ_s[2], σ_s[3])
    )
end

function apply_medium(material, medium_name, hikari_media::Dict)
    medium_name === nothing && return material
    medium = get(hikari_media, medium_name, nothing)
    medium === nothing && return material
    return Hikari.MediumInterface(material; inside=medium, outside=nothing)
end

function blackbody_rgb(T::Float32)
    x, y = Hikari.planckian_xy(T)
    X = x / y
    Y = 1f0
    Z = (1f0 - x - y) / y
    r =  3.2406f0 * X - 1.5372f0 * Y - 0.4986f0 * Z
    g = -0.9689f0 * X + 1.8758f0 * Y + 0.0415f0 * Z
    b =  0.0557f0 * X - 0.2040f0 * Y + 1.0570f0 * Z
    m = max(r, g, b, 1f-6)
    Hikari.RGBSpectrum(r / m, g / m, b / m)
end

function quad_mesh(verts::Vector{Point3f}, indices::Vector{Int})
    faces = [TriangleFace{Int}(indices[3i-2], indices[3i-1], indices[3i]) for i in 1:length(indices)÷3]
    GeometryBasics.normal_mesh(GeometryBasics.Mesh(verts, faces))
end

# =============================================================================
# Scene Setup
# =============================================================================

function create_scene(; resolution=(500, 700))
    println("Parsing crown.pbrt...")
    parsed_materials, parsed_media, geometries, area_lights = parse_crown_pbrt()
    println("  $(length(parsed_materials)) materials, $(length(parsed_media)) media, $(length(geometries)) meshes, $(length(area_lights)) area lights")

    mats = Dict(name => create_hikari_material(p, parsed_materials) for (name, p) in parsed_materials)
    hikari_media = Dict(name => create_hikari_medium(p) for (name, p) in parsed_media)
    default_mat = Hikari.Gold(roughness=0f0)

    lights = [PointLight(RGBf(0.001, 0.001, 0.001), Point3f(0, 0, 100))]
    scene = Scene(size=resolution; lights=lights)
    cam3d!(scene)

    for al in area_lights
        Le = blackbody_rgb(al.temperature)
        emissive_mat = Hikari.Emissive(Le=Le, scale=al.scale, two_sided=true)
        m = quad_mesh(al.vertices, al.indices)
        mesh!(scene, m; material=emissive_mat)
    end

    env_sphere_mesh = GeometryBasics.normal_mesh(Tesselation(Sphere(Point3f(0), 100f0), 64))
    env_mat = Hikari.MatteMaterial(Kd=Hikari.RGBSpectrum(0.2f0, 0.2f0, 0.2f0))
    mesh!(scene, env_sphere_mesh; material=env_mat)

    loaded = 0
    failed = 0
    for (i, geom) in enumerate(geometries)
        path = joinpath(CROWN_DIR, geom.file)
        isfile(path) || (failed += 1; continue)
        try
            mesh_data = load(path)
            mat = get(mats, geom.material, default_mat)
            mat = apply_medium(mat, geom.medium, hikari_media)
            mesh!(scene, mesh_data; material=mat)
            loaded += 1
        catch
            failed += 1
        end
        if i % 200 == 0
            println("  Loaded $loaded / $(length(geometries)) meshes...")
        end
    end
    println("  Loaded $loaded meshes ($failed failed)")

    update_cam!(scene, Vec3f(0, 10, 30), Vec3f(0, 8, -25), Vec3f(0, 1, 0))
    return scene
end

# =============================================================================
# Standardized RayDemo render_scene API
# =============================================================================

function render_scene(;
    device=DEVICE,
    resolution=(500, 700),
    samples=16,
    max_depth=12,
    output_path=joinpath(@__DIR__, "output", "crown.png"),
    hw_accel=false,
)
    scene = create_scene(; resolution=resolution)
    GC.gc(true)
    sensor = Hikari.FilmSensor(; iso=10, exposure_time=1.0, white_balance=4000)
    RayMakie.activate!(; device=device, sensor=sensor, exposure=1.0f0, tonemap=:aces, gamma=2.2f0)
    integrator = Hikari.VolPath(; samples=samples, max_depth=max_depth, hw_accel=hw_accel)
    @time img = colorbuffer(scene; backend=RayMakie, integrator=integrator)
    mkpath(dirname(output_path))
    save(output_path, img)
    @info "Saved → $output_path"
    return img
end

# render_scene()
