using GPUSelect
using GLMakie, RayMakie, Hikari, Mantle
using FileIO, ImageShow

# The GPU backend every scene script reads.
#
# `Mantle.defaultbackend()` and not `GPUSelect.Backend(:Lava)`, which is what
# this was. GPUSelect resolves a target by fetching a named type out of a named
# package — `Lava.LavaBackend` for `:Lava` — and since the 2026-08-27 split that
# type is neither Lava's nor Mantle's: the Vulkan runtime lives in
# `MantleVulkanExt`, so the lookup is `UndefVarError: LavaBackend not defined in
# Lava` and every scene here failed to load.
#
# Asking Mantle is the replacement the split introduced, and it answers the same
# question better: each backend registers a probe when its extension loads, and
# the highest-priority one that reports itself usable wins. `using Lava` on a
# machine with a driver gets Vulkan; `using Metal` on an Apple GPU gets Metal;
# neither is named here.
#
# To target something GPUSelect still owns — `:CUDA`, `:AMDGPU`, `:oneAPI`,
# `:CPU` — write `GPUSelect.Backend(:CUDA)` here instead. Those paths do not go
# through Mantle at all.
global DEVICE = Mantle.defaultbackend()
set_theme!()
