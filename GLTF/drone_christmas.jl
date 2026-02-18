include("../common/common.jl")
using Colors, MeshIO

# Load both models
# From https://sketchfab.com/3d-models/25k-followers-christmas-special-1fd1cdc4c2d94ecfa8d80a25f1818e8c
christmas = load(joinpath(@__DIR__, "christmas.glb"))
# From JuliaHub
drone = load(joinpath(@__DIR__, "Drone.glb"))

function create_scene(; resolution=(2400, 1800))
    # --- Lights ---
    lights = [
        # Warm key light from above-right
        Makie.PointLight(RGBf(1.0, 0.9, 0.7) * 5, Vec3f(3, -2, 6), 15.0),
        # Cool fill light from left
        Makie.PointLight(RGBf(0.4, 0.5, 0.7) * 3, Vec3f(-3, -1, 6), 10.0),
        Makie.PointLight(RGBf(0.6, 0.6, 0.8) * 2, Vec3f(2.0, -2.5, 2.0), 10.0),
    ]

    scene = Scene(backgroundcolor=RGBf(0.02, 0.02, 0.04), size=resolution, lights=lights)
    cam3d!(scene)

    # --- Christmas tree ---
    # The "decorations" material has both diffuse and emissive maps in the GLB.
    # glb_material_to_hikari automatically creates a MediumInterface with arealight,
    # so the decorations both reflect light AND glow.
    tree_plot = mesh!(scene, christmas)

    # --- Drone ---
    # Position the drone hovering above and to the side of the tree
    # The drone is small (~0.6 units) so scale it up a bit
    drone_plot = mesh!(scene, drone)
    Makie.translate!(drone_plot, Vec3f(-1.5, -0.5, 2.0))
    Makie.scale!(drone_plot, Vec3f(2, 2, 2))
    Makie.rotate!(drone_plot, Vec3f(0, 1, 1), Float32(π/6))

    # --- Room ---
    # Floor
    floor_mat = Hikari.MatteMaterial(
        Hikari.ConstTexture(Hikari.RGBSpectrum(0.15f0, 0.12f0, 0.1f0)),
        Hikari.ConstTexture(20f0)
    )
    mesh!(scene, Rect3f(Vec3f(-5, -5, -0.01), Vec3f(10, 10, 0.01));
          color=RGBf(0.15, 0.12, 0.1), material=floor_mat)

    # Back wall
    wall_mat = Hikari.MatteMaterial(
        Hikari.ConstTexture(Hikari.RGBSpectrum(0.2f0, 0.18f0, 0.17f0)),
        Hikari.ConstTexture(10f0)
    )
    mesh!(scene, Rect3f(Vec3f(-5, 4.99, 0), Vec3f(10, 0.01, 5));
          color=RGBf(0.2, 0.18, 0.17), material=wall_mat)
    mesh!(scene, Rect3f(Vec3f(4.99, -5, 0), Vec3f(0.01, 10, 5));
        color=RGBf(0.2, 0.18, 0.17), material=wall_mat)
    # Left wall
    mesh!(scene, Rect3f(Vec3f(-5, -5, 0), Vec3f(0.01, 10, 5));
        color=RGBf(0.2, 0.18, 0.17), material=wall_mat)

    # --- Camera ---
    update_cam!(scene, Vec3f(2.0, -2.5, 2.0), Vec3f(-0.8, 0, 1.0))

    return scene
end

function render_scene(;
    device=DEVICE,
    resolution=(2400, 1800),
    samples=10,
    max_depth=10,
    output_path=joinpath(@__DIR__, "drone_christmas.png"),
)
    scene = create_scene(; resolution=resolution)
    integrator = Hikari.VolPath(samples=samples, max_depth=max_depth, max_component_value=1f0)
    sensor = Hikari.FilmSensor(iso=300, white_balance=6500)
    img = colorbuffer(scene; device=device, integrator=integrator, sensor=sensor)
    save(output_path, img)
    @info "Saved → $output_path"
    return img
end

function render_interactive(;
    device=DEVICE,
    resolution=(1200, 900),
)
    scene = create_scene(; resolution=resolution)
    TraceMakie.interactive_window(scene; device=device)
    display(scene; backend=GLMakie, update=false)
end

# render_scene()
