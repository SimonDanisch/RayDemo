# Bunny Cloud Scene - NanoVDB Volumetric Path Tracing Example
# Uses actual NanoVDB volumetric data from pbrt-v4-scenes for spatially-varying density
# This parses the NanoVDB file format directly in Julia and renders with GridMedium + VolPath
include("../common/common.jl")
using GeometryBasics

# Rotation matrix helpers for pbrt-style transforms
_RotX(θ) = Mat3f(1, 0, 0, 0, cos(θ), -sin(θ), 0, sin(θ), cos(θ))
_RotZ(θ) = Mat3f(cos(θ), -sin(θ), 0, sin(θ), cos(θ), 0, 0, 0, 1)

function create_scene(;
    resolution=(800, 600),
    nvdb_path::String=joinpath(@__DIR__, "bunny_cloud.nvdb"),
    sigma_s=10.0f0,
    sigma_a=0.5f0,
    g=0.0f0,
    majorant_res=Vec3i(64, 64, 64)
)
    # pbrt bunny-cloud applies: Rotate 180 0 0 1, then Rotate 90 1 0 0
    # Matrix form: R_x(90°) * R_z(180°) (right-to-left application)
    bunny_transform = _RotX(Float32(π/2)) * _RotZ(Float32(π))

    # Create NanoVDBMedium directly from file with rotation transform
    nanovdb_medium = Hikari.NanoVDBMedium(
        nvdb_path;
        σ_a = Hikari.RGBSpectrum(sigma_a),
        σ_s = Hikari.RGBSpectrum(sigma_s),
        g = g,
        transform = bunny_transform,
        majorant_res = majorant_res
    )
    # Create scene
    s = Scene(size=resolution; lights=Makie.AbstractLight[])
    cam3d!(s)

    # Camera setup matching pbrt
    cam_pos = Vec3f(0, 120, 50)
    look_at = Vec3f(7, 0, 17)
    update_cam!(s, cam_pos, look_at, Vec3f(0, 0, 1))
    s.camera_controls.fov[] = 25.0

    # Medium boundary: pbrt's `Material "interface"`, which is a NULL material —
    # `materials.cpp`: `else if (name == "interface") return nullptr;`. It has no
    # BSDF at all; the only thing the surface does is swap the ray's medium.
    #
    # NOT a Dielectric. This used to be `Dielectric(Kr=0, Kt=1, index=1)`, which
    # looks transparent but is not the same thing: it is a real BSDF, so it
    # consumes a path vertex, goes through sampling/MIS, and makes shadow rays
    # treat the boundary as an occluder — the cloud came out under-lit.
    boundary = Hikari.NullMaterial()

    # Sphere geometry for the medium boundary. pbrt uses an ANALYTIC
    # `Shape "sphere" "float radius" 45`; we tessellate, so the count is set high
    # enough that the silhouette (where the medium begins) is not visibly
    # faceted. The default `normal_mesh(Sphere(...))` tessellation is far too
    # coarse for a 45-unit sphere filling much of the frame.
    sphere_mesh = GeometryBasics.normal_mesh(
        GeometryBasics.Tesselation(GeometryBasics.Sphere(Point3f(0, 0, 0), 45f0), 256))

    # Volume sphere with NanoVDBMedium
    volume_material = Hikari.MediumInterface(boundary; inside=nanovdb_medium, outside=nothing)
    mesh!(s, sphere_mesh; material=volume_material)

    # Ground: disk radius 1000, matching bunny-cloud.pbrt's
    #   AttributeBegin / Translate 0 -50 0 / Shape "disk" "float radius" 1000
    #
    # The disk lies in the z = HEIGHT plane, and the .pbrt sets no `height`, so
    # height = 0 (pbrt-v4 shapes.h: `tShapeHit = (height - oi.z) / di.z`, and
    # `Bounds3f((-r,-r,height), (r,r,height))`). `Translate 0 -50 0` therefore
    # moves the disk's CENTRE along Y — it does not lower the ground.
    #
    # This scene is authored Z-up in pbrt already (`LookAt 0 120 50  7 0 17
    # 0 0 1`), so there is no Y-up→Z-up conversion to do. Reading the translate
    # as a height put the ground at z = -50, twice as far below the camera at
    # z = 50 as it should be: rays near the top of the frame then needed ~1850
    # units to reach it, overshot the radius-1000 rim, and escaped into the
    # sky map's black lower hemisphere. That was the black band across the top
    # of the render and the curved horizon below it — the disk's rim. pbrt has
    # no horizon in frame at all; the ground fills it.
    ground_material = Hikari.CoatedDiffuse(
        reflectance = (0.4f0, 0.45f0, 0.35f0),
        roughness = 0f0,
        eta = 1.5f0,
        thickness = 0.01f0
    )
    ground_r = 1000f0
    ground_z = 0f0
    ground_cy = -50f0          # the disk's centre, from `Translate 0 -50 0`
    ground_segments = 96
    ground_verts = Vector{Point3f}(undef, ground_segments + 1)
    ground_verts[1] = Point3f(0, ground_cy, ground_z)
    for i in 0:ground_segments-1
        ground_verts[i+2] = Point3f(ground_r*cos(2π*i/ground_segments),
                                    ground_cy + ground_r*sin(2π*i/ground_segments), ground_z)
    end
    ground_faces = [TriangleFace{Int}(1, 2 + i, 2 + mod(i+1, ground_segments))
                    for i in 0:ground_segments-1]
    ground_geo = GeometryBasics.Mesh(ground_verts, ground_faces)
    mesh!(s, ground_geo; color=RGBf(0.4f0, 0.45f0, 0.35f0), material=ground_material)

    # Environment light, matching
    #   AttributeBegin / Rotate 10 1 0 0 / LightSource "infinite" ... "float scale" 4
    #
    # The `Rotate 10 1 0 0` was missing. pbrt's `ImageInfiniteLight::Le` does
    # `renderFromLight.ApplyInverse(ray.d)` before the equal-area lookup, and
    # Hikari's `direction_to_uv_equal_area` does `transpose(rotation) * dir` —
    # the same thing for a rotation matrix — so passing the CTM rotation here
    # matches. Without it the whole sky (and therefore every shadow direction)
    # sat 10° off.
    sky_path = joinpath(@__DIR__, "..", "assets", "sky.exr")
    sky_image = FileIO.load(sky_path)
    env_light = Makie.EnvironmentLight(4.0f0, sky_image;
                                       rotation_angle = 10f0,
                                       rotation_axis = Vec3f(1, 0, 0))
    push_light!(s, env_light)
    return s
end

function render_scene(;
    device=DEVICE,
    resolution=(1920, 1080),
    samples=5,
    max_depth=50,
    output_path=joinpath(@__DIR__, "output", "bunny_cloud.png"),
)
    scene = create_scene(; resolution=resolution)
    integrator = Hikari.VolPath(samples=samples, max_depth=max_depth)
    @time img = colorbuffer(scene;
        device=device, integrator=integrator,
        exposure=0.5, tonemap=nothing, gamma=2.2f0,
    )
    mkpath(dirname(output_path))
    save(output_path, img)
    @info "Saved → $output_path"
    return img
end

# render_scene()

if abspath(PROGRAM_FILE) == @__FILE__
    scene = create_scene(; resolution=(1920, 1080))
    sensor = Hikari.PixelSensor(sensor="nikon_d850", iso=90f0, whitebalance=5000f0)
    RayMakie.vulkan_viewer(scene; integrator=Hikari.VolPath(; hw_accel=true, sensor))
end
