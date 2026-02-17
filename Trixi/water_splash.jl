include(joinpath(@__DIR__, "..", "common", "common.jl"))
using GeometryBasics
using GeometryBasics: Point3f, GLTriangleFace, Rect3f, normal_mesh, Mesh
using FFMPEG_jll

# ReadVTK only supports XML format, so we need a custom parser for legacy binary VTK
function read_legacy_vtk_mesh(filename)
    open(filename, "r") do f
        readline(f)  # # vtk DataFile Version X.X
        readline(f)  # title
        readline(f)  # BINARY
        readline(f)  # empty line
        readline(f)  # DATASET UNSTRUCTURED_GRID

        # POINTS line
        points_line = readline(f)
        parts = split(points_line)
        n_points = parse(Int, parts[2])

        # Read binary points (big-endian float32)
        points_raw = read(f, 3 * n_points * 4)
        points = ntoh.(reinterpret(Float32, points_raw))
        points = reshape(points, 3, n_points)

        # Find CELLS keyword after binary data
        while !eof(f)
            line = readline(f)
            if startswith(line, "CELLS")
                parts = split(line)
                n_cells = parse(Int, parts[2])
                total_size = parse(Int, parts[3])

                cells_raw = read(f, total_size * 4)
                cells = ntoh.(reinterpret(Int32, cells_raw))

                # Extract triangle faces
                faces = Matrix{Int32}(undef, 3, n_cells)
                idx = 1
                for i in 1:n_cells
                    faces[1, i] = cells[idx + 1] + 1  # +1 for Julia 1-based indexing
                    faces[2, i] = cells[idx + 2] + 1
                    faces[3, i] = cells[idx + 3] + 1
                    idx += 4
                end

                return points, faces
            end
        end
        error("CELLS not found in VTK file")
    end
end

# Helper to convert points/faces to mesh arrays
function vtk_to_mesh_arrays(pts, faces)
    vertices = [Point3f(pts[1,i], pts[2,i], pts[3,i]) for i in 1:size(pts, 2)]
    triangles = [GLTriangleFace(faces[1,i], faces[2,i], faces[3,i]) for i in 1:size(faces, 2)]
    return vertices, triangles
end

# =============================================================================
# Materials
# =============================================================================

# Water material - physically correct dielectric surface (Fresnel handles reflection split)
# Kr/Kt = 1: no surface absorption. Water's blue tint is volumetric, not surface.
water_material = Hikari.GlassMaterial(
    Kr=Hikari.RGBSpectrum(1f0, 1f0, 1f0),
    Kt=Hikari.RGBSpectrum(1f0, 1f0, 1f0),
    roughness=0.01f0,
    index=1.33f0
)

# Ball material - red matte
ball_material = Hikari.Gold(; roughness=0.01f0)

# Ground plane material - slightly reflective floor
ground_material = Hikari.PlasticMaterial(
    Kd=Hikari.RGBSpectrum(0.4f0, 0.4f0, 0.45f0),
    Ks=Hikari.RGBSpectrum(0.2f0, 0.2f0, 0.2f0),
    roughness=0.3f0
)

# =============================================================================
# Scene setup
# =============================================================================

# Get frame files
function get_frame_files(frame_idx)
    fluid_file = joinpath(@__DIR__, "fluid", "fluid_1_surface_$frame_idx.vtk")
    solid_file = joinpath(@__DIR__, "fluid", "solid_1_surface_$frame_idx.vtk")
    return fluid_file, solid_file
end

# Load a VTK file and convert to GeometryBasics mesh
function load_vtk_mesh(filename)
    pts, faces = read_legacy_vtk_mesh(filename)
    verts, tris = vtk_to_mesh_arrays(pts, faces)
    return normal_mesh(GeometryBasics.Mesh(verts, tris))
end

function build_water_scene(frame_idx; size=(1200, 1200))
    fluid_file, solid_file = get_frame_files(frame_idx)

    # Load meshes
    fluid_mesh = load_vtk_mesh(fluid_file)
    solid_mesh = load_vtk_mesh(solid_file)

    fig = Figure(size=size)

    # Environment light with rotation (rotate to get better lighting angle)
    ax = LScene(fig[1, 1]; show_axis=false, scenekw=(;
        lights=[
            Makie.SunSkyLight(
                Vec3f(0.4, -0.3, 0.7);           # sun direction (late afternoon)
                intensity=1.0f0,
                turbidity=3.0f0,
                ground_enabled=false,             # pure sky dome, no ground plane
            ),
        ]
    ))

    # Plot the water surface
    water_plot = mesh!(ax, fluid_mesh,
        color=RGBf(0.6, 0.8, 0.95),
        shading=true
    )
    water_plot.material[] = water_material

    # Plot the ball
    ball_plot = mesh!(ax, solid_mesh,
        color=RGBf(0.8, 0.2, 0.2),
        shading=true
    )
    ball_plot.material[] = ball_material

    # Compute scene bounds from initial frame for fixed camera
    pts, _ = read_legacy_vtk_mesh(fluid_file)
    x_min, x_max = extrema(pts[1,:])
    y_min, y_max = extrema(pts[2,:])
    z_min, z_max = extrema(pts[3,:])

    # Ground plane below the fluid — catches caustics and shadows, provides contrast
    extent = max(x_max - x_min, z_max - z_min) * 2f0
    cx, cz = (x_min + x_max) / 2, (z_min + z_max) / 2
    ground_y = y_min - 0.05f0
    ground_plot = mesh!(ax,
        Rect3f(Vec3f(cx - extent, ground_y - 0.01, cz - extent), Vec3f(2extent, 0.02, 2extent)),
        color=RGBf(0.5, 0.5, 0.5), shading=true)
    ground_plot.material[] = ground_material

    # Set fixed camera (won't move during animation)
    center_pt = Point3f((x_min + x_max)/2, (y_min + y_max)/2, (z_min + z_max)/2)
    cam = ax.scene.camera_controls
    cam.eyeposition[] = Vec3f(center_pt[1] + 2.5, center_pt[2] + 1.5, center_pt[3] + 2.0)
    cam.lookat[] = Vec3f(center_pt...) .+ Vec3f(0, 0.3, 0)
    cam.upvector[] = Vec3f(0, 1, 0)  # Y-up
    cam.fov[] = 45f0
    update_cam!(ax.scene, cam)

    return fig, ax, water_plot, ball_plot
end

# Update mesh plots with new frame data (in-place via compute graph)
function update_frame!(water_plot, ball_plot, frame_idx)
    fluid_file, solid_file = get_frame_files(frame_idx)
    water_plot.mesh[] = load_vtk_mesh(fluid_file)
    ball_plot.mesh[] = load_vtk_mesh(solid_file)
    return nothing
end

# =============================================================================
# Animation with Makie.record (in-place mesh updates, no scene re-creation)
# =============================================================================

function render_video(;
    device=DEVICE,
    resolution=(1200, 1200),
    samples=100,
    max_depth=12,
    nframes=480,
    output_path=joinpath(@__DIR__, "water_splash.mp4"),
)
    frame_range = 1:max(1, 480 ÷ nframes):480
    integrator = Hikari.VolPath(
        samples=samples, max_depth=max_depth, max_component_value=10,
        regularize=true
    )
    sensor = Hikari.FilmSensor(iso=150, white_balance=10000)

    # Build scene once with first frame
    fig, ax, water_plot, ball_plot = build_water_scene(1; size=resolution)

    n_total = length(frame_range)
    colorbuffer(fig; device=device, integrator=integrator, tonemap=:aces, gamma=2.0, sensor=sensor, update=false)
    Makie.record_longrunning(ax.scene, output_path, frame_range;
        framerate=30, device=device, integrator=integrator, tonemap=:aces, gamma=2.0, sensor=sensor, update=false
    ) do frame_idx
        update_frame!(water_plot, ball_plot, frame_idx)
    end

    @info "Saved → $output_path"
end

# render_video()
