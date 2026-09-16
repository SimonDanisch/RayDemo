using GPUSelect
using GLMakie, RayMakie, Hikari
import Mantle
using FileIO, ImageShow

# `Mantle.LavaBackend()` directly, and not `GPUSelect.Backend(:Lava)`.
#
# GPUSelect resolves a target to a constructor by loading the module it is named
# after and reading a symbol out of it — `:Lava` means `getfield(Lava,
# :LavaBackend)`. That name moved to Mantle on 2026-08-27 with the rest of the
# runtime (115 names; see `Lava.jl`'s header), so every scene script here has
# been dying at load with "UndefVarError: `LavaBackend` not defined in `Lava`".
# GPUSelect is a registered package and its table cannot be fixed from here;
# naming the owner is one line and says where the backend comes from.
#
# The genericity GPUSelect buys is intact for the other targets: swap this for
# `GPUSelect.Backend(:CUDA)` (or :AMDGPU, :Metal, :oneAPI, :GPU, :CPU) and every
# scene follows, because they all read `DEVICE`. The device CHOICE within Vulkan
# — discrete GPU, falling back to lavapipe — was never GPUSelect's; it is
# Mantle's, inside the backend.
global DEVICE = Mantle.LavaBackend()
set_theme!()
