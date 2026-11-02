using GPUSelect, Mantle
using GLMakie, RayMakie, Hikari
using FileIO, ImageShow

# Which hardware every scene script targets. They all read `DEVICE` and none of
# them names a backend, so this line is the only thing to change.
const TARGET = :Lava

# `GPUSelect.Backend(:Lava)` looks up `Lava.LavaBackend`, and there has been no
# such name since the Lava→Mantle split: Lava is the Julia→SPIR-V compiler and
# the backend it used to carry lives in Mantle's Vulkan extension. GPUSelect is
# an upstream package and its table still says otherwise, so asking it for
# `:Lava` throws `UndefVarError` — which the benchmark driver caught per scene
# and reported as "Failed to load crown", five times, with a summary of "no
# results" and an exit code of zero.
#
# `Mantle.defaultbackend()` is the same question asked of the package that now
# answers it: it probes every registered backend and returns the one that came
# up. Every other target still goes through GPUSelect, which is right for them —
# CUDA and AMDGPU are not Mantle's.
global DEVICE = TARGET === :Lava ? Mantle.defaultbackend() : GPUSelect.Backend(TARGET)
set_theme!()
