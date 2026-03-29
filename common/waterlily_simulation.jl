using WaterLily
using Printf, JLD2
using BiotSavartBCs
using WaterLily: dot, sgs!, inside, ω_mag
using Lava
const PARAMS = (
    # Grid parameters
    p=4,                      # Grid resolution exponent (m = 3 * 2^p)
    # Physics
    Re = 3700,                  # Reynolds number
    U = 1,                      # Freestream velocity

    # Timing (in CTU - convective time units)
    time_max=200,            # Total simulation time
    stats_init=100,          # Time to start collecting statistics
    stats_interval=0.1,      # Interval for statistics collection
    save_interval=0.5,       # Interval for saving checkpoints
    # Backend
    mem=LavaArray,            # Array type (ROCArray, CuArray, Array)
    T=Float32,               # Floating point precision
    # Data directory
    datadir=joinpath(@__DIR__, "sim"),
)

#==============================================================================#
#                              HELPER FUNCTIONS                                 #
#==============================================================================#

derived_params(p) = (
    m = 3 * 2^p.p,
    R = (3 * 2^p.p) ÷ 3,
    fname_base = joinpath(p.datadir, "p$(p.p)"),
)

# Sphere geometry
function make_sphere(m; n=5m ÷ 2, R=m ÷ 3, U=1, Re=3700, T=Float32, mem=Array)
    body = AutoBody((x, t) -> √sum(abs2, x .- m ÷ 2) - R)
    BiotSimulation((n, m, m), (U, 0, 0), 2R; ν=U * 2R / Re, body, T, mem)
end

function load_simulation(load_time, params=PARAMS)
    (; p, Re, U, mem, T, datadir) = params
    (; m, R, fname_base) = derived_params(params)

    t_str = @sprintf("%i", load_time)
    println("Loading simulation from t=$load_time...")
    sim = make_sphere(m; R, U, Re, T, mem)
    load!(sim; fname="$(fname_base)_t$(t_str).jld2")
    println("  Loaded: $(fname_base)_t$(t_str).jld2")
    return sim
end
function save_sim(sim, datadir, meanflow, force, t)
    fname_save = joinpath(datadir, "p$(p)")
    t_str = @sprintf("%i", sim_time(sim))
    save!(fname_save * "_t$(t_str).jld2", sim)
    save!(fname_save * "_t$(t_str)_meanflow.jld2", meanflow)
    jldsave(fname_save * "_t$(t_str)_force.jld2"; force, t)
end

function run_simulation(params=PARAMS)
    (; p, Re, U, mem, T, datadir) = params
    (; m, R, fname_base) = derived_params(params)
    sim = make_sphere(m; R, U, Re, T, mem)
    S = zeros(T, size(sim.flow.p)..., ndims(sim.flow.p), ndims(sim.flow.p)) |> mem # working array holding a tensor for each cell
    force, t = Vector{T}[], T[] # force coefficients, time
    stats_init = 100 # in CTU
    sim_step!(sim, stats_init; remeasure=false, verbose=true, S)
    stats_interval = 0.1 # in CTU
    save_interval = 50 # in CTU
    time_max = 300
    meanflow = MeanFlow(sim.flow; uu_stats=true)
    next_save = sim_time(sim) + save_interval
    while sim_time(sim) < time_max
        sim_step!(sim, sim_time(sim) + stats_interval; remeasure=false, verbose=false, S)
        sim_info(sim)

        WaterLily.update!(meanflow, sim.flow)
        push!(force, WaterLily.total_force(sim) / (0.5 * sim.U^2 * sim.L^2))
        push!(t, sim_time(sim))

        if WaterLily.sim_time(sim) > next_save || sim_time(sim) > time_max
            save_sim(sim, datadir, meanflow, force, t)
            next_save = sim_time(sim) + save_interval
            println("Saved simulation and mean flow statistics.")
        end

    end
    return sim, meanflow, force, t
end

function extract_vorticity(sim)
    a = sim.flow.σ
    WaterLily.@inside a[I] = ω_mag(I, sim.flow.u)
    return Float32.(Array(a[inside(a)]))
end

function prepare_sgs(sim, params)
    (; mem, T) = params
    return zeros(T, size(sim.flow.p)..., ndims(sim.flow.p), ndims(sim.flow.p)) |> mem
end
