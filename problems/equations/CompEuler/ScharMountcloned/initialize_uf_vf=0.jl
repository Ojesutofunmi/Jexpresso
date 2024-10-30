using Base
using Interpolations

function initialize(SD::NSD_2D, PT::CompEuler, mesh::St_mesh, inputs::Dict, OUTPUT_DIR::String, TFloat)
    """

                """
    @info " Initialize fields for 2D CompEuler with θ equation ........................ "
    

    
    qvars = ("dρ", "dρu", "dρv", "dρθ")
    q = define_q(SD, mesh.nelem, mesh.npoin, mesh.ngl, qvars, TFloat, inputs[:backend]; neqs=length(qvars))
    
    if (inputs[:backend] == CPU()) 
        PhysConst = PhysicalConst{Float64}()
        θref = 280.0 #K
        θ0 = 280.0
        T0   = θ0
        p0   = 100000.0
    
        N    = 0.01#PhysConst.g/sqrt(PhysConst.cp*T0)
        N2   = N*N
    
        #readinf the data from file
        data = readdlm("velocity_data.csv", ',')
        yf = data[2:end, 1]          # Skip header and get distance
        uf = data[2:end, 2]         # Skip header and get u component
       # vf = data[2:end, 3]         # Skip header and get v component

        u_interp = LinearInterpolation(yf, uf, extrapolation_bc=Flat())

        for iel_g = 1:mesh.nelem
            for j=1:mesh.ngl, i=1:mesh.ngl
            
                ip = mesh.connijk[iel_g,i,j]
                y = mesh.y[ip]
                θ    = θref*exp(N2*y/PhysConst.g)         
                p    = p0*(1.0 + PhysConst.g2*(exp(-y*N2/PhysConst.g) - 1.0)/(PhysConst.cp*θref*N2))^PhysConst.cpoverR
                ρ    = perfectGasLaw_θPtoρ(PhysConst; θ=θ,    Press=p) #kg/m³
                ρref = perfectGasLaw_θPtoρ(PhysConst; θ=θ, Press=p) #kg/m³
                u = u_interp(y)
                v = 0.0
                if inputs[:SOL_VARS_TYPE] == PERT()
                    q.qn[ip,1] = ρ-ρref
                    q.qn[ip,2] = ρ*u - ρref*u
                    q.qn[ip,3] = ρ*v - ρref*v
                    q.qn[ip,4] = ρ*θ - ρref*θ
                    q.qn[ip,end] = p
                else
                    q.qn[ip,1] = ρ
                    q.qn[ip,2] = ρ*u
                    q.qn[ip,3] = ρ*v
                    q.qn[ip,4] = ρ*θ
                    q.qn[ip,end] = p
                end
                #Store initial background state for plotting and analysis of pertuebations
                q.qe[ip,1] = ρref
                q.qe[ip,2] = ρref*u
                q.qe[ip,3] = ρref*v
                q.qe[ip,4] = ρref*θ
                q.qe[ip,end] = p
            end
        end
        outvarsref = ("rho_ref", "u_ref", "v_ref", "theta_ref", "p_ref")    
        write_vtk_ref(SD, mesh, q.qe, "REFERENCE_state", inputs[:output_dir]; nvar=length(q.qe[1,:]), outvarsref=outvarsref)

    else
        if (inputs[:SOL_VARS_TYPE] == PERT())
            lpert = true
        else
            lpert = false
        end
        PhysConst = PhysicalConst{TFloat}()
        θref = TFloat(280.0) #K
        θ0 = TFloat(280.0)
        T0   = θ0
        p0   = TFloat(100000.0)

        N    = TFloat(0.01)
        N2   = TFloat(N*N)

        k = initialize_gpu!(inputs[:backend])
        k(q.qn, q.qe, mesh.x, mesh.y, θref, θ0, T0, p0, N, N2, PhysConst, lpert; ndrange = (mesh.npoin))
    end
    @info "Initialize fields for system of 2D CompEuler with θ equation ........................ DONE"
    
    return q
end

@kernel function initialize_gpu!(qn, qe, x, y, θref, θ0, T0, p0, N, N2, PhysConst, lpert)
    ip = @index(Global, Linear)

    x = x[ip]
    y = y[ip]
    T = eltype(x)

    θ    = θref*exp(N2*y/PhysConst.g)
    p    = p0*(T(1.0) + PhysConst.g2*(exp(-y*N2/PhysConst.g) - T(1.0))/(PhysConst.cp*θref*N2))^PhysConst.cpoverR

    ρ    = perfectGasLaw_θPtoρ(PhysConst; θ=θ,    Press=p) #kg/m³
    ρref = perfectGasLaw_θPtoρ(PhysConst; θ=θ, Press=p)

    u = T(10.0)
    v = T(0.0)

    qn[ip,1] = ρ-ρref
    qn[ip,2] = ρ*u - ρref*u
    qn[ip,3] = ρ*v - ρref*v
    qn[ip,4] = ρ*θ - ρref*θ
    qn[ip,end] = p

    qe[ip,1] = ρref
    qe[ip,2] = ρref*u
    qe[ip,3] = ρref*v
    qe[ip,4] = ρref*θ
    qe[ip,end] = p

end
