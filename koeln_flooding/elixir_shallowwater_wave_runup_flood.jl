
# Include packages
using TrixiBottomTopography
using JuMP
using HiGHS
using OrdinaryDiffEqSSPRK
using Trixi
using TrixiShallowWater


# Point to the two dimensional bottom data
bathymetry_data = joinpath(@__DIR__, "dgm_merged.txt")

# Shape preserving L1 spline interpolation of the data
spline_struct = LaverySpline2D(bathymetry_data)
spline_func(x::Float64, y::Float64) = spline_interpolation(spline_struct, x, y)

# H0 has an elevation 41 m
equations = ShallowWaterEquations2D(gravity = 9.81, H0 = 41.0,
                                    threshold_desingularization = 1e-8)

function initial_condition_wave(x, t, equations::ShallowWaterEquations2D)
    # # Calculate primitive variables
    v1 = 0
    v2 = 0
    b = spline_func(x[1], x[2])
    H = max(equations.H0, b + equations.threshold_limiter)

    return prim2cons(SVector(H, v1, v2, b), equations)
end

function boundary_condition_wave_maker(u_inner, normal_direction::AbstractVector, x, t,
                                       surface_flux_functions, equations::ShallowWaterEquations2D)
    surface_flux_function, nonconservative_flux_function = surface_flux_functions

    # Use an exponential window function to put the tsunami in the river mouth
    # Parameters for a soliton type wave
    A = 100
    scal = 10 * 0.25 / (1 * sqrt(log(2)))
    kx = 1 / sqrt(30)
    t0 = 26.0
    arg = ( kx * (t - t0) ) / scal
    L = 357250
    w = 225
    window = exp(-((x[1] - L) / w)^2)
    eta = window * A * sech(arg)^2
    H_ext = equations.H0 + eta

    # Create the external water height and internal reference height
    h_ext = max(equations.threshold_limiter, H_ext - u_inner[4])
    h0 = max(equations.threshold_limiter, equations.H0 - u_inner[4])

    # Compute the incoming velocity as in Eq. (10) of the paper
    # "A limiter-based well-balanced discontinuous Galerkin method for shallow-water flows
    #  with wetting and drying: Triangular grids" by Vater, Beisiegel, and Behrens
    v2_ext = 2 * (sqrt(equations.gravity * h_ext) - sqrt(equations.gravity * h0))

    # Create the external solution state in the conservative variables
    u_outer = SVector(h_ext, zero(eltype(x)), -h_ext * v2_ext, u_inner[4])

    # Calculate the boundary flux
    flux = surface_flux_function(u_inner, u_outer, normal_direction, equations)

    noncons_flux = nonconservative_flux_function(u_inner, u_outer, normal_direction,
                                                 equations)

    return flux, noncons_flux
end

# Setting initial condition
initial_condition = initial_condition_wave

# Setting the boundary conditions
boundary_condition = (; Bottom = BoundaryConditionDirichlet(initial_condition),
                        Left = BoundaryConditionDirichlet(initial_condition),
                        Right = BoundaryConditionDirichlet(initial_condition),
                        Top = boundary_condition_wave_maker)

###############################################################################
# Get the DG approximation space

volume_flux = (flux_wintermeyer_etal, flux_nonconservative_wintermeyer_etal)
surface_flux = (FluxHydrostaticReconstruction(flux_hll_chen_noelle,
                                              hydrostatic_reconstruction_chen_noelle),
                flux_nonconservative_chen_noelle)

basis = LobattoLegendreBasis(7)

indicator_sc = IndicatorHennemannGassnerShallowWater(equations, basis,
                                                     alpha_max = 0.5,
                                                     alpha_min = 0.001,
                                                     alpha_smooth = true,
                                                     variable = waterheight)
volume_integral = VolumeIntegralShockCapturingHG(indicator_sc;
                                                 volume_flux_dg = volume_flux,
                                                 volume_flux_fv = surface_flux)

solver = DGSEM(basis, surface_flux, volume_integral)

###############################################################################
# Get the mesh

mesh_file = Trixi.download("https://gist.githubusercontent.com/andrewwinters5000/18cad369281c6a91ab3e2a8303a4e500/raw/c3a55010a13147fea822220c1c890e3932918203/CartesianBox.inp",
                           joinpath(@__DIR__, "CartesianBox.inp"))
mesh = P4estMesh{2}(mesh_file; initial_refinement_level = 1)

# create the semi discretization object
semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver,
                                    boundary_conditions = boundary_condition)

###############################################################################
# ODE solvers, callbacks etc.

tspan = (0.0, 60.0)
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()

analysis_interval = 1000
analysis_callback = AnalysisCallback(semi, interval = analysis_interval,
                                     save_analysis = false)

alive_callback = AliveCallback(analysis_interval = analysis_interval)

save_solution = SaveSolutionCallback(dt = 2.0,
                                     save_initial_solution = true,
                                     save_final_solution = true)

stepsize_callback = StepsizeCallback(cfl = 1.0)

callbacks = CallbackSet(summary_callback,
                        analysis_callback,
                        alive_callback,
                        save_solution,
                        stepsize_callback)

stage_limiter! = PositivityPreservingLimiterShallowWater(variables = (waterheight,))

###############################################################################
# run the simulation

# use a Runge-Kutta method with error based time step size control
sol = solve(ode, SSPRK43(stage_limiter!);
            ode_default_options()...,
            callback = callbacks,
            adaptive = false, dt = 1.0,
            saveat = 2.0);

###############################################################################
# Save plotting data (plain arrays that serialize cleanly via JLD2)

using JLD2, StructArrays

# Extract triangulated plot data from Trixi
pd1 = Trixi.PlotData2DTriangulated(sol.u[1], semi; nvisnodes=30)
x_nodes = collect(vec(pd1.x))
y_nodes = collect(vec(pd1.y))
b_nodes = collect(vec(StructArrays.components(pd1.data)[4]))  # bathymetry
ref_tri = collect(pd1.t)  # per-element triangulation

# Build global triangulation (replicate per-element connectivity for all elements)
n_nodes_per_elem = size(pd1.x, 1)
n_elements = size(pd1.x, 2)
n_tri_per_elem = size(ref_tri, 1)
global_tri = Matrix{Int32}(undef, n_tri_per_elem * n_elements, 3)
for e in 1:n_elements
    offset = Int32((e - 1) * n_nodes_per_elem)
    rows = ((e-1)*n_tri_per_elem+1):(e*n_tri_per_elem)
    global_tri[rows, :] .= ref_tri .+ offset
end

# Extract water surface height H for each saved frame
times = collect(sol.t)
H_frames = Matrix{Float64}(undef, length(x_nodes), length(times))
for i in eachindex(times)
    pd_i = Trixi.PlotData2DTriangulated(sol.u[i], semi; nvisnodes=30)
    H_frames[:, i] .= vec(StructArrays.components(pd_i.data)[1])
end

jldsave(joinpath(@__DIR__, "simulation_data.jld2"); x_nodes, y_nodes, b_nodes, global_tri, H_frames, times)
@info "Saved simulation_data.jld2 ($(length(times)) frames)"
