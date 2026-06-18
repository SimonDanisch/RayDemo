using GPUSelect
using GLMakie, RayMakie, Hikari
using FileIO, ImageShow

# Backend-independent GPU selection via GPUSelect. `Backend(:Lava)` returns a
# LavaBackend() (Vulkan; picks the discrete GPU, falls back to lavapipe when
# none). Swap the symbol (:CUDA, :AMDGPU, :Metal, :oneAPI, :GPU, :CPU) to target
# other hardware without touching any scene script — they all read `DEVICE`.
global DEVICE = GPUSelect.Backend(:Lava)
set_theme!()
