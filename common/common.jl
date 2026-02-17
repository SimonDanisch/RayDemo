using AMDGPU
using GLMakie, TraceMakie, Hikari
using FileIO, ImageShow

global DEVICE = AMDGPU.ROCBackend()
set_theme!()
