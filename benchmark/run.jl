# Unified Benchmark Runner
#
# Single entry point that runs ALL benchmarks (render + AK) for all detected backends.
# Results are saved with versioned naming: {platform}_{device}_{backend}_{version}.json
# GPU backends use the GPU name, CPU backends use the CPU name as {device}.
#
# Usage — load your backend(s) first, then include and run:
#
#   using Lava, AMDGPU        # load whichever backends you have
#   include("RayDemo/benchmark/run.jl")
#
# Results end up in RayDemo/benchmark/results/:
#   linux_7900xtx_lava_sw_v1.json      # GPU render benchmarks
#   linux_7900xtx_lava_hw_v1.json
#   linux_7900xtx_amdgpu_v1.json
#   linux_7900x_pbrt_cpu_v1.json       # CPU backends use CPU name
#   linux_7900xtx_ak_10m_lava_v1.json  # AK benchmarks (GPU)
#   linux_7900xtx_ak_10m_amdgpu_v1.json
#   linux_7900x_ak_10m_cpu_v1.json     # AK CPU uses CPU name
#
# Existing results are NEVER overwritten (skipped if file exists).
# To re-run benchmarks, bump the version: run_all_benchmarks(version="v2")
# To force overwrite: run_all_benchmarks(force=true)
#
# Each platform pushes their results/ to the repo. Then visualize:
#   include("RayDemo/benchmark/plot_results.jl")
#   plot_all_results()

include(joinpath(@__DIR__, "run_benchmarks.jl"))

run_all_benchmarks()
