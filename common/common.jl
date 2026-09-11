using Mantle
using GLMakie, RayMakie, Hikari
using FileIO, ImageShow

# The GPU every scene script reads.
#
# `Mantle.defaultbackend()` and not `GPUSelect.Backend(:Lava)`: GPUSelect maps
# `:Lava` to `Mantle.LavaBackend`, and the Vulkan RUNTIME moved out of Lava into
# `MantleVulkanExt` on 2026-08-27 — Lava is the Julia→SPIR-V compiler now and
# has no such name, so every scene died with `UndefVarError: LavaBackend not
# defined in Lava`. Asking Mantle is the portable spelling: `using Lava`
# registers the Vulkan backend, `using Metal` would register the other one, and
# `defaultbackend()` answers with whichever is there.
global DEVICE = Mantle.defaultbackend()
set_theme!()
