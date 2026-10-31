# Live WGLMakie + Bonito app: the ReynkoViz "advected-streaklet" flow mode, demoed on the
# WaterLily dolphin sim. Streaklets are temporally-coherent pathline trails advected through
# the (time-varying) velocity field, vorticity/inflow-seeded, killed on body-entry/outflow.
# Colour = |u| (turbo), head-bright→tail-faint fade. Rendered as a single `lines!` (backend
# agnostic) + the body as an `sdf=0` isosurface.
#
# Run (Reynko env, which has WGLMakie/Bonito/JLD2):
#   julia --project=/sim/Programmieren/Reynko
#   julia> include("/sim/Programmieren/VulkanDev/RayDemo/Waterlily/dolphin_streaklet_app.jl")
#   julia> serve_dolphin_app()          # prints a URL; open it in a browser
using WGLMakie, GeometryBasics, LinearAlgebra, JLD2, Bonito
using GeometryBasics: Point3f, Vec3f
WGLMakie.activate!()

const STEPS = "/sim/Programmieren/VulkanDev/RayDemo/Waterlily/dolphin_smoke_L256_steps"

# ── generic streaklet engine (grid-sampled velocity) ────────────────────────────────────
mutable struct Swarm{FV}
    fv::FV; N::Int; TRAIL::Int; nsub::Int
    P::Matrix{Point3f}; W::Matrix{Float32}; age::Vector{Int}; life::Vector{Int}
    t::Float32; fade::Int; wscale::Float32
end
struct GridField
    usd::Vector{Array{Float32,4}}; sdd::Vector{Array{Float32,3}}
    tt::Vector{Float32}; lo::Point3f; hi::Point3f; vscale::Float32
end

@inline function tl_u(A::Array{Float32,4}, p::Point3f)
    nx,ny,nz,_=size(A); x=clamp(p[1],1f0,Float32(nx)); y=clamp(p[2],1f0,Float32(ny)); z=clamp(p[3],1f0,Float32(nz))
    i=clamp(unsafe_trunc(Int,x),1,nx-1); j=clamp(unsafe_trunc(Int,y),1,ny-1); k=clamp(unsafe_trunc(Int,z),1,nz-1); fx=x-i; fy=y-j; fz=z-k
    @inbounds begin
        c(a,b,cc)=Point3f(A[a,b,cc,1],A[a,b,cc,2],A[a,b,cc,3])
        x00=c(i,j,k)*(1-fx)+c(i+1,j,k)*fx; x10=c(i,j+1,k)*(1-fx)+c(i+1,j+1,k)*fx
        x01=c(i,j,k+1)*(1-fx)+c(i+1,j,k+1)*fx; x11=c(i,j+1,k+1)*(1-fx)+c(i+1,j+1,k+1)*fx
    end; (x00*(1-fy)+x10*fy)*(1-fz)+(x01*(1-fy)+x11*fy)*fz
end
@inline function tl_s(A::Array{Float32,3}, p::Point3f)
    nx,ny,nz=size(A); x=clamp(p[1],1f0,Float32(nx)); y=clamp(p[2],1f0,Float32(ny)); z=clamp(p[3],1f0,Float32(nz))
    i=clamp(unsafe_trunc(Int,x),1,nx-1); j=clamp(unsafe_trunc(Int,y),1,ny-1); k=clamp(unsafe_trunc(Int,z),1,nz-1); fx=x-i; fy=y-j; fz=z-k
    @inbounds begin
        x00=A[i,j,k]*(1-fx)+A[i+1,j,k]*fx; x10=A[i,j+1,k]*(1-fx)+A[i+1,j+1,k]*fx
        x01=A[i,j,k+1]*(1-fx)+A[i+1,j,k+1]*fx; x11=A[i,j+1,k+1]*(1-fx)+A[i+1,j+1,k+1]*fx
    end; (x00*(1-fy)+x10*fy)*(1-fz)+(x01*(1-fy)+x11*fy)*fz
end
function frame_idx(fv::GridField, t)
    t<=fv.tt[1] && return (1,1,0f0); t>=fv.tt[end] && return (length(fv.tt),length(fv.tt),0f0)
    i=searchsortedlast(fv.tt,t); (i,i+1,(t-fv.tt[i])/(fv.tt[i+1]-fv.tt[i]))
end
function velocity(fv::GridField, r::Point3f, t::Float32)
    i,j,a = frame_idx(fv,t); fv.vscale .* ((1f0-a).*tl_u(fv.usd[i],r) .+ a.*tl_u(fv.usd[j],r))
end
body_sdf(fv::GridField, r::Point3f, t::Float32) = tl_s(fv.sdd[frame_idx(fv,t)[1]], r .- Point3f(0.5f0))

@inline function rk4p(fv, r::Point3f, t::Float32, h::Float32)
    k1=velocity(fv,r,t); k2=velocity(fv,r+0.5f0h.*k1,t+0.5f0h)
    k3=velocity(fv,r+0.5f0h.*k2,t+0.5f0h); k4=velocity(fv,r+h.*k3,t+h)
    r+(h/6f0).*(k1+2f0.*k2+2f0.*k3+k4)
end
function seed_spread(fv,t)
    for _ in 1:30; p=fv.lo.+(fv.hi.-fv.lo).*Point3f(rand(),rand(),rand()); body_sdf(fv,p,t)>0.2f0 && return p; end
    fv.lo.+(fv.hi.-fv.lo).*Point3f(rand(),rand(),rand())
end
function seed_inflow(fv,t)
    for _ in 1:20; p=Point3f(fv.lo[1]+(fv.hi[1]-fv.lo[1])*rand(),fv.lo[2]+1f0+6f0*rand(),fv.lo[3]+(fv.hi[3]-fv.lo[3])*rand()); body_sdf(fv,p,t)>0.5f0 && return p; end
    Point3f(fv.lo[1]+(fv.hi[1]-fv.lo[1])*rand(),2f0,fv.lo[3]+(fv.hi[3]-fv.lo[3])*rand())
end
function init_swarm(fv;N=500,TRAIL=22,nsub=3,fade=6)
    t0=fv.tt[1]; ws=sort!([norm(velocity(fv,fv.lo.+(fv.hi.-fv.lo).*Point3f(rand(),rand(),rand()),t0)) for _ in 1:4000]); wscale=ws[round(Int,0.9*length(ws))]
    sw=Swarm{typeof(fv)}(fv,N,TRAIL,nsub,Matrix{Point3f}(undef,TRAIL,N),zeros(Float32,TRAIL,N),zeros(Int,N),rand(400:1000,N),t0,fade,wscale)
    for j in 1:N; s=seed_spread(fv,t0); sw.P[:,j].=(s,); sw.W[:,j].=norm(velocity(fv,s,t0)); sw.age[j]=rand(0:sw.life[j]); end; sw
end
function advance!(sw,t_target)
    fv=sw.fv; lo=fv.lo; hi=fv.hi; h=(t_target-sw.t)/sw.nsub
    for _ in 1:sw.nsub
        @inbounds for j in 1:sw.N
            nh=rk4p(fv,sw.P[sw.TRAIL,j],sw.t,h)
            out=nh[1]<lo[1]||nh[1]>hi[1]||nh[2]<lo[2]||nh[2]>hi[2]||nh[3]<lo[3]||nh[3]>hi[3]
            inbody=(!out)&&body_sdf(fv,nh,sw.t+h)<0f0; sw.age[j]+=1
            if out||inbody||sw.age[j]≥sw.life[j]
                s=seed_inflow(fv,sw.t); sw.P[:,j].=(s,); sw.W[:,j].=norm(velocity(fv,s,sw.t)); sw.age[j]=0; sw.life[j]=rand(400:1000)
            else
                @views sw.P[1:end-1,j].=sw.P[2:end,j]; @views sw.W[1:end-1,j].=sw.W[2:end,j]
                sw.P[sw.TRAIL,j]=nh; sw.W[sw.TRAIL,j]=norm(velocity(fv,nh,sw.t+h))
            end
        end; sw.t+=h
    end
end
function snapshot!(pts,us,al,sw::Swarm)
    empty!(pts); empty!(us); empty!(al)
    @inbounds for j in 1:sw.N
        fin=clamp(sw.age[j]/sw.fade,0f0,1f0); fout=clamp((sw.life[j]-sw.age[j])/sw.fade,0f0,1f0); lifef=min(fin,fout)
        for p in 1:sw.TRAIL
            push!(pts,sw.P[p,j]); push!(us,sw.W[p,j]); push!(al,(p/sw.TRAIL)^1.3f0*lifef)
        end
        push!(pts,Point3f(NaN)); push!(us,0f0); push!(al,0f0)
    end
end
const CMAP = Makie.to_colormap(:turbo)
function compose(us,al,wscale)
    c=Vector{RGBAf}(undef,length(us))
    @inbounds for i in eachindex(us)
        u=clamp(us[i]/wscale,0f0,1f0); col=CMAP[clamp(round(Int,u*(length(CMAP)-1))+1,1,length(CMAP))]
        c[i]=RGBAf(col.r,col.g,col.b,0.95f0*al[i])
    end; c
end

# ── load dolphin frames + run the forward pass (Option B: precompute per timestep) ──────
function load_and_generate(; stride_t=2, stride_s=2, N=500, TRAIL=22, nsub=3)
    files = sort(filter(startswith("frame_"), readdir(STEPS)))
    selt = 1:stride_t:length(files); s=stride_s
    U=Vector{Array{Float32,4}}(undef,length(selt)); S=Vector{Array{Float32,3}}(undef,length(selt)); T=Vector{Float32}(undef,length(selt))
    for (i,f) in enumerate(selt)
        jldopen(joinpath(STEPS,files[f]),"r") do d
            U[i]=d["u"][1:s:end,1:s:end,1:s:end,:]; S[i]=d["sdf"][1:s:end,1:s:end,1:s:end]; T[i]=d["sim_t"]
        end
    end
    nx,ny,nz,_ = size(U[1])
    fv = GridField(U,S,T, Point3f(1,1,1), Point3f(nx,ny,nz), Float32(256/s))   # vscale = L/stride
    sw = init_swarm(fv;N=N,TRAIL=TRAIL,nsub=nsub)
    pos=Vector{Point3f}[]; usc=Vector{Float32}[]; alp=Vector{Float32}[]
    pts=Point3f[]; us=Float32[]; al=Float32[]
    for (i,t) in enumerate(fv.tt); i>1&&advance!(sw,t); snapshot!(pts,us,al,sw); push!(pos,copy(pts)); push!(usc,copy(us)); push!(alp,copy(al)); end
    return (; pos, usc, alp, wscale=sw.wscale, S, tt=T)
end

# ── the live Bonito app: 3D scene + time slider (server-side → real scrubbing/animation) ─
function dolphin_app(data)
    App() do
        sl = Bonito.Slider(1:length(data.pos))
        fig = Figure(size=(900,760), backgroundcolor=RGBf(0.02,0.02,0.05))
        ls = LScene(fig[1,1]; show_axis=false)
        P = map(i->data.pos[i], sl.value); C = map(i->compose(data.usc[i],data.alp[i],data.wscale), sl.value)
        B = map(i->data.S[i], sl.value)
        ny = size(data.S[1],2)
        xe=(1.5f0, Float32(size(data.S[1],1))+0.5f0); ye=(1.5f0, Float32(ny)+0.5f0); ze=(1.5f0, Float32(size(data.S[1],3))+0.5f0)
        lines!(ls, P; color=C, linewidth=2.0, transparency=true)
        contour!(ls, xe,ye,ze, B; levels=[0f0], colormap=[RGBAf(0.82,0.84,0.9,1)], alpha=0.5)
        update_cam!(ls.scene, Vec3f(250,35,155), Vec3f(0.5*xe[2],0.5*ye[2],0.5*ze[2]), Vec3f(0,0,1))
        return DOM.div(DOM.div("time: ", sl; style="color:#ccc;padding:6px"), fig)
    end
end

const DEMO = Ref{Any}()
function serve_dolphin_app(; port=8791, host="0.0.0.0")
    isassigned(DEMO) || (DEMO[] = load_and_generate())
    app = dolphin_app(DEMO[])
    server = Bonito.Server(app, host, port)
    println("Dolphin streaklet app live at  http://localhost:$port")
    return server
end
