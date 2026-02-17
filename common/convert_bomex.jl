# One-time conversion: BOMEX JLD2 → NanoVDB
# Requires Oceananigans. After running, bomex_1024.nanovdb can be used
# without Oceananigans in breeze.jl and terrain.jl.

using Hikari
using GeometryBasics: Point3f, Vec3f
using Oceananigans: FieldTimeSeries, interior
using Oceananigans.OutputReaders: OnDisk

function convert_bomex(jld2_path, nvdb_path; frame=5)
    qlt = FieldTimeSeries(jld2_path, "qˡ"; backend=OnDisk())
    cloud_data = Float32.(interior(qlt[frame]))
    grid = qlt.grid
    grid_extent = Vec3f(grid.Lx, grid.Ly, grid.Lz)

    # Rescale to avoid Float32 precision issues with large coordinates (12800m domain)
    # This divides spatial extent by 1000 and multiplies density by 100
    scaled_extent = grid_extent ./ 1000f0
    scaled_data = cloud_data .* 100f0

    origin = Point3f(-scaled_extent[1]/2, -scaled_extent[2]/2, 0)
    Hikari.save_nanovdb(nvdb_path, scaled_data, origin, scaled_extent)

    println("Converted $(size(cloud_data)) grid → $nvdb_path")
    println("  File size: $(filesize(nvdb_path) ÷ 1024) KB")
    println("  Original extent: $grid_extent")
    println("  Scaled extent: $scaled_extent")
    println("  Density range: [$(minimum(cloud_data)), $(maximum(cloud_data))] → [$(minimum(scaled_data)), $(maximum(scaled_data))]")
end

convert_bomex(
    joinpath(@__DIR__, "bomex_1024.jld2"),
    joinpath(@__DIR__, "bomex_1024.nanovdb")
)
