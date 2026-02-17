# Rayshader-style Terrain with Clouds
# Real elevation data (via Tyler) + BOMEX LES cloud data
# Inspired by https://www.rayshader.com/
#
# Requires: bomex_1024.nanovdb (generate with convert_bomex.jl)

include(joinpath(@__DIR__, "..", "common", "common.jl"))
using Colors, GeometryBasics
using LinearAlgebra: normalize
using Hikari: RGBSpectrum, compute_perez_coefficients, compute_zenith_values, _compute_sky_radiance

using Tyler
using Tyler: ElevationProvider, ElevationData, PathDownloader, fetch_tile, get_downloader
using MapTiles: Tile, TileGrid, web_mercator, wgs84
using MapTiles: extent as tile_extent
using Extents: Extent

# =============================================================================
# Tile Fetching & Stitching
# =============================================================================

const TILE_CACHE = Dict{String, Any}()

function fetch_tiles_for_extent(lat, lon, delta; zoom=12)
    cache_key = "tiles_$(lat)_$(lon)_$(delta)_$(zoom)"
    if haskey(TILE_CACHE, cache_key)
        return TILE_CACHE[cache_key]
    end

    provider = ElevationProvider()
    downloader = get_downloader(provider)
    ext = Extent(X=(lon - delta/2, lon + delta/2), Y=(lat - delta/2, lat + delta/2))
    tiles = collect(TileGrid(ext, zoom, wgs84))

    tile_data = Dict{Tile, ElevationData}()
    for tile in tiles
        tile_data[tile] = fetch_tile(provider, downloader, tile)
    end

    result = (tile_data, tiles, ext)
    TILE_CACHE[cache_key] = result
    return result
end

function get_elevation_stats(tile_data)
    elev_min = Inf32
    elev_max = -Inf32
    for data in values(tile_data)
        mini, maxi = extrema(data.elevation)
        elev_min = min(elev_min, mini)
        elev_max = max(elev_max, maxi)
    end
    return elev_min, elev_max
end

function compute_overall_extent(tiles)
    x_min, x_max = Inf, -Inf
    y_min, y_max = Inf, -Inf
    for tile in tiles
        ext = tile_extent(tile, wgs84)
        x_min = min(x_min, ext.X[1]); x_max = max(x_max, ext.X[2])
        y_min = min(y_min, ext.Y[1]); y_max = max(y_max, ext.Y[2])
    end
    return (X=(x_min, x_max), Y=(y_min, y_max))
end

# Stitch tiles into combined elevation and color arrays.
# Elevation data needs TRANSPOSE; color data needs VERTICAL FLIP.
# After processing: row 1 = south, row end = north, col 1 = west, col end = east.
function stitch_tiles(tile_data, tiles, zoom)
    xs_tiles = sort(unique(t.x for t in tiles))
    ys_tiles = sort(unique(t.y for t in tiles))

    first_data = first(values(tile_data))
    first_elev_t = permutedims(first_data.elevation, (2, 1))
    tile_h, tile_w = size(first_elev_t)

    has_color = !isempty(first_data.color)
    if has_color
        color_tile_h, color_tile_w = size(first_data.color)
    end

    n_tiles_x = length(xs_tiles)
    n_tiles_y = length(ys_tiles)

    # Elevation tiles have shared boundary rows/cols (e.g. 257x257)
    total_h = n_tiles_y * tile_h - (n_tiles_y - 1)
    total_w = n_tiles_x * tile_w - (n_tiles_x - 1)
    elevation = zeros(Float32, total_h, total_w)

    if has_color
        color_total_h = n_tiles_y * color_tile_h
        color_total_w = n_tiles_x * color_tile_w
        color_raw = zeros(RGBf, color_total_h, color_total_w)
    end

    for (row_idx, ty) in enumerate(reverse(ys_tiles))  # south to north
        for (col_idx, tx) in enumerate(xs_tiles)        # west to east
            tile = Tile(tx, ty, zoom)
            haskey(tile_data, tile) || continue
            data = tile_data[tile]

            elev_t = permutedims(Float32.(data.elevation), (2, 1))
            row_start = (row_idx - 1) * (tile_h - 1) + 1
            col_start = (col_idx - 1) * (tile_w - 1) + 1
            elevation[row_start:row_start+tile_h-1, col_start:col_start+tile_w-1] .= elev_t

            if has_color && !isempty(data.color)
                color_data = reverse(RGBf.(data.color), dims=1)
                c_row_start = (row_idx - 1) * color_tile_h + 1
                c_col_start = (col_idx - 1) * color_tile_w + 1
                color_raw[c_row_start:c_row_start+color_tile_h-1, c_col_start:c_col_start+color_tile_w-1] .= color_data
            end
        end
    end

    # Resample color to match elevation grid size
    if has_color
        if size(color_raw) == size(elevation)
            color = color_raw
        else
            eh, ew = size(elevation)
            ch, cw = size(color_raw)
            color = Matrix{RGBf}(undef, eh, ew)
            for i in 1:eh, j in 1:ew
                ci = clamp(round(Int, (i - 0.5) * ch / eh + 0.5), 1, ch)
                cj = clamp(round(Int, (j - 0.5) * cw / ew + 0.5), 1, cw)
                color[i, j] = color_raw[ci, cj]
            end
        end
    else
        color = Matrix{RGBf}(undef, 0, 0)
    end

    return elevation, color
end



# =============================================================================
# Filled Sides Mesh (solid block appearance)
# =============================================================================

function create_sides_mesh(xs, ys, elevation, base_z)
    nx, ny = size(elevation)
    xs_range = xs isa Tuple ? LinRange(xs[1], xs[2], nx) : xs
    ys_range = ys isa Tuple ? LinRange(ys[1], ys[2], ny) : ys

    boundary_top = Point3f[]
    boundary_bottom = Point3f[]

    # Bottom edge (row 1)
    for j in 1:ny
        z = elevation[1, j]
        push!(boundary_top, Point3f(xs_range[1], ys_range[j], z))
        push!(boundary_bottom, Point3f(xs_range[1], ys_range[j], base_z))
    end
    # Right edge (col end)
    for i in 2:nx
        z = elevation[i, end]
        push!(boundary_top, Point3f(xs_range[i], ys_range[end], z))
        push!(boundary_bottom, Point3f(xs_range[i], ys_range[end], base_z))
    end
    # Top edge (row end, reversed)
    for j in (ny-1):-1:1
        z = elevation[end, j]
        push!(boundary_top, Point3f(xs_range[end], ys_range[j], z))
        push!(boundary_bottom, Point3f(xs_range[end], ys_range[j], base_z))
    end
    # Left edge (col 1, reversed)
    for i in (nx-1):-1:2
        z = elevation[i, 1]
        push!(boundary_top, Point3f(xs_range[i], ys_range[1], z))
        push!(boundary_bottom, Point3f(xs_range[i], ys_range[1], base_z))
    end

    n_boundary = length(boundary_top)
    vertices = vcat(boundary_top, boundary_bottom)

    faces = QuadFace{Int}[]
    for i in 1:n_boundary
        i_next = mod1(i + 1, n_boundary)
        push!(faces, QuadFace(i, i_next, i_next + n_boundary, i + n_boundary))
    end

    # Bottom face
    n_side = length(vertices)
    push!(vertices, Point3f(xs_range[1], ys_range[1], base_z))
    push!(vertices, Point3f(xs_range[end], ys_range[1], base_z))
    push!(vertices, Point3f(xs_range[end], ys_range[end], base_z))
    push!(vertices, Point3f(xs_range[1], ys_range[end], base_z))
    push!(faces, QuadFace(n_side + 1, n_side + 4, n_side + 3, n_side + 2))

    return GeometryBasics.Mesh(vertices, faces)
end

function add_filled_sides!(scene, xs, ys, elevation, base_z; color=RGBf(0.25, 0.22, 0.2))
    sides_mesh = create_sides_mesh(xs, ys, elevation, base_z)
    mesh!(scene, sides_mesh; color=color)
end

# =============================================================================
# Scene Construction
# =============================================================================

function rayshader_scene(;
        lat=47.087441, lon=13.377214, delta=0.1, zoom=12,
        cloud_altitude_factor=1.5,
        sun_altitude=30.0, sun_azimuth=135.0,
        separate_sun_sky=true, sun_intensity=1.0f0, turbidity=2.0f0,
        figsize=(1024, 768))

    tile_data, tiles, ext = fetch_tiles_for_extent(lat, lon, delta; zoom=zoom)

    elev_min, elev_max = get_elevation_stats(tile_data)
    terrain_height = elev_max - elev_min
    overall_ext = compute_overall_extent(tiles)

    # Sun direction from altitude/azimuth
    alt_rad = deg2rad(sun_altitude)
    azi_rad = deg2rad(sun_azimuth)
    sun_dir = Vec3f(cos(alt_rad) * sin(azi_rad), cos(alt_rad) * cos(azi_rad), sin(alt_rad))

    lights = [
        Makie.SunSkyLight(sun_dir; intensity=sun_intensity, turbidity=turbidity, ground_enabled=false),
    ]

    scene = Makie.Scene(; size=figsize, lights=lights)
    cam3d!(scene)

    # Normalize geographic coords to 0-2 scene range
    lat_to_m = 111000.0
    lon_to_m = 111000.0 * cos(deg2rad(lat))
    x_extent_m = (overall_ext.X[2] - overall_ext.X[1]) * lon_to_m
    y_extent_m = (overall_ext.Y[2] - overall_ext.Y[1]) * lat_to_m
    scale_xy = 2.0 / max(x_extent_m, y_extent_m)
    base_z_norm = -0.05f0

    geo_to_norm(lon_val, lat_val) = (
        Float32((lon_val - overall_ext.X[1]) * lon_to_m * scale_xy),
        Float32((lat_val - overall_ext.Y[1]) * lat_to_m * scale_xy),
    )
    x_min_norm, y_min_norm = geo_to_norm(overall_ext.X[1], overall_ext.Y[1])
    x_max_norm, y_max_norm = geo_to_norm(overall_ext.X[2], overall_ext.Y[2])

    # Stitch tiles into single surface
    stitched_elev, stitched_color = stitch_tiles(tile_data, tiles, zoom)
    stitched_elev_norm = (stitched_elev .- elev_min) .* Float32(scale_xy)
    max_elev_norm = maximum(stitched_elev_norm)

    if !isempty(stitched_color)
        surface!(scene, (x_min_norm, x_max_norm), (y_min_norm, y_max_norm), stitched_elev_norm;
            color=stitched_color, shading=NoShading)
    else
        surface!(scene, (x_min_norm, x_max_norm), (y_min_norm, y_max_norm), stitched_elev_norm;
            colormap=:terrain)
    end

    add_filled_sides!(scene, (x_min_norm, x_max_norm), (y_min_norm, y_max_norm),
                      stitched_elev_norm, base_z_norm)

    # Cloud volume above terrain (NanoVDB sparse tree from pre-converted file)
    cloud_base_norm = max_elev_norm + 0.1f0
    cloud_thickness_norm = terrain_height * Float32(scale_xy) * cloud_altitude_factor
    cloud_xy_extent = max(x_max_norm - x_min_norm, y_max_norm - y_min_norm)

    bomex = joinpath(@__DIR__, "bomex_1024.nanovdb")
    cloud_medium = Hikari.NanoVDBMedium(bomex;
        σ_a=Hikari.RGBSpectrum(0.0f0),
        σ_s=Hikari.RGBSpectrum(150f0),
        majorant_res=Vec3i(64, 64, 64),
        g=0.8f0,
    )

    # Position cloud box above terrain, matching the scene's normalized coordinates
    cloud_origin = Point3f(x_min_norm, y_min_norm, cloud_base_norm)
    cloud_size = Vec3f(cloud_xy_extent, cloud_xy_extent, cloud_thickness_norm)
    transparent = Hikari.GlassMaterial(
        Kr=Hikari.RGBSpectrum(0f0), Kt=Hikari.RGBSpectrum(1f0), index=1.0f0
    )
    volume_material = Hikari.MediumInterface(transparent; inside=cloud_medium, outside=nothing)
    mesh!(scene, Rect3f(cloud_origin, cloud_size); material=volume_material)

    # Camera
    center_x = (x_min_norm + x_max_norm) / 2
    center_y = (y_min_norm + y_max_norm) / 2
    center_z = max_elev_norm / 2
    cam_pos = Vec3f(center_x, center_y, max_elev_norm) .+ Vec3f(1.9, -1.9, 1.9)
    look_at = Vec3f(center_x, center_y, center_z) .- Vec3f(0, 0, 0.35)
    update_cam!(scene, cam_pos, look_at, Vec3f(0, 0, 1))
    scene.camera_controls.fov[] = 40.0

    return scene, max_elev_norm
end

# =============================================================================
# Run
# =============================================================================

function render_scene(;
    device=DEVICE,
    resolution=(1024, 768),
    samples=1000,
    max_depth=50,
    output_path=joinpath(@__DIR__, "terrain.png"),
)
    scene, _ = rayshader_scene(;
        lat=47.087441, lon=13.377214,
        delta=0.08, zoom=13,
        cloud_altitude_factor=1.2,
        sun_altitude=20.0, sun_azimuth=135.0,
        figsize=resolution,
    )
    integrator = Hikari.VolPath(samples=samples, max_depth=max_depth)
    sensor = Hikari.FilmSensor(iso=50, white_balance=6500)
    @time result = Makie.colorbuffer(scene;
        device=device, integrator=integrator, sensor=sensor,
        exposure=1f0, tonemap=:aces, gamma=2.0f0, update=false,
    )
    save(output_path, result)
    @info "Saved → $output_path"
    return result
end

# render_scene()
