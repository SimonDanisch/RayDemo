# WaterLily Sphere Wake Simulation
# Simulation setup, loading, and vorticity extraction

using WaterLily
using Printf, JLD2
using BiotSavartBCs
using WaterLily: dot, sgs!, inside, ω_mag
using AMDGPU

#==============================================================================#
#                           SIMULATION PARAMETERS                               #
#==============================================================================#

const PARAMS = (
    # Grid parameters
    p = 4,                      # Grid resolution exponent (m = 3 * 2^p)

    # Physics
    Re = 3700,                  # Reynolds number
    U = 1,                      # Freestream velocity

    # SGS model
    explicit_sgs = true,        # Use explicit Smagorinsky SGS model
    Cs = 0.17f0,               # Smagorinsky constant

    # Timing (in CTU - convective time units)
    time_max = 200,            # Total simulation time
    stats_init = 100,          # Time to start collecting statistics
    stats_interval = 0.1,      # Interval for statistics collection
    save_interval = 0.5,       # Interval for saving checkpoints

    # Backend
    mem = ROCArray,            # Array type (ROCArray, CuArray, Array)
    T = Float32,               # Floating point precision

    # Data directory
    datadir = joinpath(@__DIR__, "sim"),
)

#==============================================================================#
#                              HELPER FUNCTIONS                                 #
#==============================================================================#

derived_params(p) = (
    m = 3 * 2^p.p,
    R = (3 * 2^p.p) ÷ 3,
    Δ = sqrt(3.0f0),           # Filter width √(1² + 1² + 1²)
    udf = p.explicit_sgs ? sgs! : nothing,
    λ = p.explicit_sgs ? cds : quick,
    fname_base = joinpath(p.datadir, "p$(p.p)"),
)

# SGS model
smagorinsky(I::CartesianIndex{m} where m; S, Cs, Δ) =
    @views (Cs * Δ)^2 * sqrt(2dot(S[I, :, :], S[I, :, :]))

# Sphere geometry
function make_sphere(m; n=5m ÷ 2, R=m ÷ 3, U=1, Re=3700, T=Float32, mem=Array)
    body = AutoBody((x, t) -> √sum(abs2, x .- m ÷ 2) - R)
    BiotSimulation((n, m, m), (U, 0, 0), 2R; ν=U * 2R / Re, body, T, mem)
end

#==============================================================================#
#                           LOAD SIMULATION                                    #
#==============================================================================#

function load_simulation(load_time, params=PARAMS)
    (; p, Re, U, mem, T) = params
    (; m, R, fname_base) = derived_params(params)

    t_str = @sprintf("%i", load_time)
    println("Loading simulation from t=$load_time...")

    sim = make_sphere(m; R, U, Re, T, mem)
    load!(sim; fname="$(fname_base)_t$(t_str).jld2")
    println("  Loaded: $(fname_base)_t$(t_str).jld2")

    return sim
end

#==============================================================================#
#                         VORTICITY EXTRACTION                                 #
#==============================================================================#

"""
    extract_vorticity(sim) -> Array{Float32, 3}

Extract vorticity magnitude from simulation as a 3D array suitable for volume rendering.
"""
function extract_vorticity(sim)
    a = sim.flow.σ
    WaterLily.@inside a[I] = ω_mag(I, sim.flow.u)
    return Float32.(Array(a[inside(a)]))
end

#==============================================================================#
#                              SGS HELPERS                                     #
#==============================================================================#

"""
    prepare_sgs(sim, params) -> S

Allocate the SGS strain-rate tensor array on the simulation backend.
"""
function prepare_sgs(sim, params)
    (; mem, T) = params
    return zeros(T, size(sim.flow.p)..., ndims(sim.flow.p), ndims(sim.flow.p)) |> mem
end
