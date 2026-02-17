# CMS Detector — Ray-Traced Visualization
#
# Loads the CMS particle detector from GDML via Geant4.jl, applies a quadrant cut
# to reveal internal structure, and renders with physically-based materials.

include(joinpath(@__DIR__, "..", "common", "common.jl"))
using Geant4
using Geant4.SystemOfUnits
using GeometryBasics
using Colors

# Load quadrant cut and scene construction utilities
include(joinpath(@__DIR__, "..", "common", "cms_utils.jl"))

# Increase tessellation quality for curved surfaces
HepPolyhedron!SetNumberOfRotationSteps(128)

# ============================================================================
# Load the CMS detector from GDML
# ============================================================================

detname = "cms2018"

gdml_path = joinpath(dirname(pathof(Geant4)), "..", "docs", "src", "examples", "$(detname).gdml")

detector = G4JLDetectorGDML(gdml_path; validate_schema=false)
app = G4JLApplication(; detector=detector, physics_type=FTFP_BERT)
configure(app)
initialize(app)

world = GetWorldVolume()

# ============================================================================
# Collect meshes with quadrant cut
# ============================================================================

println("Collecting detector meshes with quadrant cut...")
@time lv_meshes, glass_meshes = collect_detector_meshes(world; maxlevel=5, cut_quadrant=true)

# ============================================================================
# Render scene
# ============================================================================

function render_scene(;
    device=DEVICE,
    resolution=(800, 1080),
    samples=1000,
    max_depth=6,
    output_path=joinpath(@__DIR__, "cms_detector.png"),
)
    fig_trace, ax_trace = create_trace_scene(lv_meshes; glass_meshes=nothing, resolution=resolution)

    set_camera!(ax_trace;
        eyeposition = [22278.71, 22075.75, 993.36],
        lookat = [-2694.81, -2659.24, -72.56],
        upvector = [-0.02, -0.02, 1.0],
        fov = 45.0
    )

    integrator = Hikari.VolPath(samples=samples, max_depth=max_depth)
    sensor = Hikari.FilmSensor(iso=30, white_balance=6500)

    println("Rendering with GPU, $samples samples...")
    @time result = Makie.colorbuffer(ax_trace.scene;
        device=device,
        integrator=integrator,
        tonemap=:aces,
        sensor=sensor,
        update=false
    )

    save(output_path, result)
    @info "Saved → $output_path"
    return result
end

# render_scene()
