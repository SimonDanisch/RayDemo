using GPUSelect
using GLMakie, RayMakie, Hikari
using FileIO, ImageShow

# Backend-independent GPU selection via GPUSelect. Every scene script reads
# `DEVICE`, so this is the one place the target hardware is chosen.
#
# `RAYDEMO_BACKEND` overrides it: `:Lava` (Vulkan), `:GPU` (the native backend
# for this machine — Metal on Apple silicon, CUDA on NVIDIA, ...), `:CPU`.
# Without it, Vulkan when a loader is present and the machine's native GPU
# backend otherwise, because `Backend(:Lava)` throws on a Mac and hardcoding it
# meant no scene in this repo could run on one.
function _raydemo_default_backend()
    haskey(ENV, "RAYDEMO_BACKEND") && return Symbol(ENV["RAYDEMO_BACKEND"])
    avail = GPUSelect.available_backends()
    # Apple silicon has no Vulkan loader, so `:Lava` cannot produce a device
    # there however the manifest looks; `:GPU` resolves to Metal. Everywhere
    # else prefer `:Lava`, because `:GPU` would pick the vendor backend (CUDA,
    # ROCm) and that is a different renderer, not the same one on other
    # hardware. This is a launcher choosing an API for the machine it is on —
    # nothing below it branches on the platform.
    apple_silicon = Sys.isapple() && Sys.ARCH === :aarch64
    !apple_silicon && :Lava in avail && return :Lava
    :GPU in avail && return :GPU
    :Lava in avail && return :Lava
    return :CPU
end

global DEVICE = GPUSelect.Backend(_raydemo_default_backend())
set_theme!()
