# Unified Benchmark Runner
#
# Single entry point that runs ALL benchmarks (render + AK) for all detected backends.
# Results are saved with platform-specific naming: {platform}_{device}_{backend}.json
# GPU backends use the GPU name, CPU backends use the CPU name as {device}.
#
# Usage — load your backend(s) first, then include and run:
#
#   using Lava, AMDGPU        # load whichever backends you have
#   include("RayDemo/benchmark/run.jl")
#
# Results end up in RayDemo/benchmark/results/:
#   linux_7900xtx_lava_sw.json         # GPU render benchmarks
#   linux_7900xtx_lava_hw.json
#   linux_7900xtx_amdgpu.json
#   linux_7900x_pbrt_cpu.json          # CPU backends use CPU name
#   linux_7900xtx_ak_lava.json         # AK benchmarks (GPU)
#   linux_7900xtx_ak_amdgpu.json
#   linux_7900x_ak_cpu.json            # AK CPU uses CPU name
#
# Each platform pushes their results/ to the repo. Then visualize:
#   include("RayDemo/benchmark/plot_results.jl")
#   plot_all_results()

include(joinpath(@__DIR__, "run_benchmarks.jl"))

run_all_benchmarks()
