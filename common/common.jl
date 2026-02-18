using AMDGPU
using GLMakie, RayMakie, Hikari
using FileIO, ImageShow

global DEVICE = AMDGPU.ROCBackend()
set_theme!()
