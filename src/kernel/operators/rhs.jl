include("../abstractTypes.jl")
include("../mesh/mesh.jl")
include("../mesh/metric_terms.jl")
include("../../problems/AbstractProblems.jl")
#include("../../io/print_matrix.jl")

#---------------------------------------------------------------------------
# Fetch problem name to access the user_rhs functions
#---------------------------------------------------------------------------
problem_name = ARGS[1]
user_flux_dir = string("../../problems/", problem_name, "/user_flux.jl")
user_source_dir = string("../../problems/", problem_name, "/user_source.jl")
include(user_flux_dir)
include(user_source_dir)
#---------------------------------------------------------------------------
function rhs!(du, u, params, time)

    #SD::NSD_1D, QT::Inexact, PT::Wave1D, mesh::St_mesh, metrics::St_metrics, M, De, u)
    T       = params.T
    SD      = params.SD
    QT      = params.QT
    PT      = params.PT
    neqns   = params.neqns
    basis   = params.basis
    mesh    = params.mesh
    metrics = params.metrics
    inputs  = params.inputs
    ω       = params.ω
    M       = params.M
    De      = params.De
    Le      = params.Le
    Δt      = params.Δt
    RHS = build_rhs(SD, QT, PT, u, neqns, basis, ω, mesh, metrics, M, De, Le, time, inputs, Δt, T)    
    for i=1:neqns
       idx = (i-1)*mesh.npoin
       du[idx+1:i*mesh.npoin] .= RHS[:,i]
    end  
    return du #This is already DSSed
end



function build_rhs(SD::NSD_1D, QT::Inexact, PT::AdvDiff, qp::Array, neqns, basis, ω, mesh::St_mesh, metrics::St_metrics, M, De, Le, time, inputs, Δt, T)

    Fuser = user_flux(T, SD, qp, mesh)
    
    #
    # Linear RHS in flux form: f = u*u
    #  
    RHS = zeros(mesh.npoin)
    qe  = zeros(mesh.ngl)
    fe  = zeros(mesh.ngl)
    for iel=1:mesh.nelem
        for i=1:mesh.ngl
            I = mesh.conn[i,iel]
            qe[i] = qp[I,1]
            fe[i] = Fuser[I]
        end
        for i=1:mesh.ngl
            I = mesh.conn[i,iel]
            for j=1:mesh.ngl
                RHS[I] = RHS[I] - De[i,j,iel]*fe[j] + inputs[:νx]*Le[i,j,iel]*qe[j]
            end
        end
    end
    
    # M⁻¹*rhs where M is diagonal
    RHS .= RHS./M

    apply_periodicity!(SD, RHS, qp, mesh, inputs, QT, metrics, basis.ψ, basis.dψ, ω, 0, neqns)
    
    return RHS
    
end

function build_rhs(SD::NSD_1D, QT::Exact, PT::AdvDiff, qp::Array, neqns, basis, ω, mesh::St_mesh, metrics::St_metrics, M, De, Le, time, inputs, Δt, T) nothing end


function build_rhs(SD::NSD_2D, QT::Inexact, PT::AdvDiff, qp::Array, neqns, basis, ω, mesh::St_mesh, metrics::St_metrics, M, De, Le, time, inputs, Δt, T)
    
    F      = zeros(mesh.ngl, mesh.ngl, mesh.nelem)
    G      = zeros(mesh.ngl, mesh.ngl, mesh.nelem)
    rhs_el = zeros(mesh.ngl, mesh.ngl, mesh.nelem)
    
    #B.C.
    apply_boundary_conditions!(SD, rhs_el, qp, mesh, inputs, QT, metrics, basis.ψ, basis.dψ, ω, time, neqns)   
    Fuser, Guser = user_flux(T, SD, qp, mesh)
    
    for iel=1:mesh.nelem
        for i=1:mesh.ngl
            for j=1:mesh.ngl
                ip = mesh.connijk[i,j,iel]
                
                F[i,j,iel] = Fuser[ip]
                G[i,j,iel] = Guser[ip]
            end
        end
    end
    
   # for ieq = 1:neqns
        for iel=1:mesh.nelem
            for i=1:mesh.ngl
                for j=1:mesh.ngl
                    
                    dFdξ = 0.0
                    dFdη = 0.0
                    dGdξ = 0.0
                    dGdη = 0.0
                    for k = 1:mesh.ngl
                        dFdξ = dFdξ + basis.dψ[k, i]*F[k,j,iel]
                        dFdη = dFdη + basis.dψ[k, j]*F[i,k,iel]

                        dGdξ = dGdξ + basis.dψ[k, i]*G[k,j,iel]
                        dGdη = dGdη + basis.dψ[k, j]*G[i,k,iel]
                    end
                    dFdx = dFdξ*metrics.dξdx[i,j,iel] + dFdη*metrics.dηdx[i,j,iel]
                    dGdy = dGdξ*metrics.dξdy[i,j,iel] + dGdη*metrics.dηdy[i,j,iel]
                    rhs_el[i, j, iel] -= ω[i]*ω[j]*metrics.Je[i,j,iel]*(dFdx + dGdy)
                end
            end
        end
    #end
    #show(stdout, "text/plain", el_matrices.D)

    #Build rhs_el(diffusion)
    rhs_diff_el = build_rhs_diff(SD, QT, PT, qp,  neqns, basis, ω, inputs[:νx], inputs[:νy], mesh, metrics, T)
    
    
    #DSS(rhs_el)
    RHS = DSS_rhs(SD, rhs_el + rhs_diff_el, mesh.connijk, mesh.nelem, mesh.npoin,neqns, mesh.nop, T)
    divive_by_mass_matrix!(RHS, M, QT,neqns)

    
    return RHS
    
end

function build_rhs(SD::NSD_2D, QT::Exact, PT::AdvDiff, qp::Array, neqns, basis, ω, mesh::St_mesh, metrics::St_metrics, M, De, Le, time, inputs, Δt, T) nothing end

function build_rhs(SD::NSD_2D, QT::Inexact, PT::LinearCLaw, qp::Array, neqns, basis, ω, mesh::St_mesh, metrics::St_metrics, M, De, Le, time, inputs, Δt, T)    
    
    F    = zeros(mesh.ngl,mesh.ngl,mesh.nelem, neqns)
    G    = zeros(mesh.ngl,mesh.ngl,mesh.nelem, neqns)

    rhs_el = zeros(mesh.ngl,mesh.ngl,mesh.nelem, neqns)
    qq = zeros(mesh.npoin,neqns)
    for i=1:neqns
        idx = (i-1)*mesh.npoin
        qq[:,i] .= qp[idx+1:i*mesh.npoin]
    end
    Fuser, Guser = user_flux(T, SD, qq, mesh)
    dFdx = zeros(neqns)
    dFdξ = zeros(neqns)
    dGdξ = zeros(neqns)
    dGdy = zeros(neqns)
    dFdη = zeros(neqns)
    dGdη = zeros(neqns)
    
    for iel=1:mesh.nelem

        for i=1:mesh.ngl
            for j=1:mesh.ngl
                ip = mesh.connijk[i,j,iel]
                F[i,j,iel,1] = Fuser[ip,1]
                F[i,j,iel,2] = Fuser[ip,2]
                F[i,j,iel,3] = Fuser[ip,3]

                G[i,j,iel,1] = Guser[ip,1]
                G[i,j,iel,2] = Guser[ip,2]
                G[i,j,iel,3] = Guser[ip,3]

            end
        end

        for i=1:mesh.ngl
            for j=1:mesh.ngl
                dFdξ = zeros(T, neqns)
                dFdη = zeros(T, neqns)
                dGdξ = zeros(T, neqns) 
                dGdη = zeros(T, neqns)
                for k = 1:mesh.ngl
                    dFdξ[1:neqns] .= dFdξ[1:neqns] .+ basis.dψ[k,i]*F[k,j,iel,1:neqns]
                    dFdη[1:neqns] .= dFdη[1:neqns] .+ basis.dψ[k,j]*F[i,k,iel,1:neqns]
                    
                    dGdξ[1:neqns] .= dGdξ[1:neqns] .+ basis.dψ[k,i]*G[k,j,iel,1:neqns]
                    dGdη[1:neqns] .= dGdη[1:neqns] .+ basis.dψ[k,j]*G[i,k,iel,1:neqns]
                end
                dFdx .= dFdξ[1:neqns]*metrics.dξdx[i,j,iel] .+ dFdη[1:neqns]*metrics.dηdx[i,j,iel]
                dGdy .= dGdξ[1:neqns]*metrics.dξdy[i,j,iel] .+ dGdη[1:neqns]*metrics.dηdy[i,j,iel]
                rhs_el[i,j,iel,1:neqns] .-= ω[i]*ω[j]*metrics.Je[i,j,iel]*(dFdx[1:neqns] .+ dGdy[1:neqns])
            end
        end
    end
    rhs_diff_el = build_rhs_diff(SD, QT, PT, qp,  neqns, basis, ω, inputs[:νx], inputs[:νy], mesh, metrics, T)
    apply_boundary_conditions!(SD, rhs_el, qq, mesh, inputs, QT, metrics, basis.ψ, basis.dψ, ω, Δt*(floor(time/Δt)), neqns)
    for i=1:neqns
        idx = (i-1)*mesh.npoin
        qp[idx+1:i*mesh.npoin] .= qq[:,i]
    end
    RHS = DSS_rhs(SD, rhs_el .+ rhs_diff_el, mesh.connijk, mesh.nelem, mesh.npoin, neqns, mesh.nop, T)
    divive_by_mass_matrix!(RHS, M, QT,neqns)
    return RHS
end

function build_rhs(SD::NSD_1D, QT::Inexact, PT::ShallowWater, qp::Array, neqns, basis, ω, mesh::St_mesh, metrics::St_metrics, M, De, Le, time, inputs, Δt, T)

    F      = zeros(mesh.ngl,mesh.nelem, neqns)
    F1     = zeros(mesh.ngl,mesh.nelem, neqns)
    rhs_el = zeros(mesh.ngl,mesh.nelem, neqns)
    qq     = zeros(mesh.npoin,neqns)
    for i=1:neqns
        idx = (i-1)*mesh.npoin
        qq[:,i] .= qp[idx+1:i*mesh.npoin]
    end
    qq[:,1] = max.(qq[:,1],0.001)
    Fuser, Fuser1 = user_flux(T, SD, qq, mesh)
    dFdx = zeros(neqns)
    dFdξ = zeros(neqns)
    gHsx = zeros(neqns)
    for iel=1:mesh.nelem

        for i=1:mesh.ngl
                ip = mesh.conn[i,iel]
                F[i,iel,1] = Fuser[ip,1]
                F[i,iel,2] = Fuser[ip,2]

                F1[i,iel,1] = Fuser1[ip,1]
                F1[i,iel,2] = Fuser1[ip,2]
                #@info Fuser[ip,1] + Fuser1[ip,1], Fuser[ip,2] + Fuser1[ip,2]
        end

        for i=1:mesh.ngl
                dFdξ = zeros(T, neqns)
                dFdξ1 = zeros(T, neqns)
                for k = 1:mesh.ngl
                    dFdξ[1:neqns] .= dFdξ[1:neqns] .+ basis.dψ[k,i]*F[k,iel,1:neqns]

                    dFdξ1[1:neqns] .= dFdξ1[1:neqns] .+ basis.dψ[k,i]*F1[k,iel,1:neqns]
                    #@info i,dFdξ[1:neqns], dFdξ1[1:neqns]
                end
                ip = mesh.conn[i,iel]
                x = mesh.x[ip]
                Hb = bathymetry(x)
                Hs = max(qq[ip,1] - Hb,0.001)
                gHsx[1] = 1.0
                gHsx[2] = Hs*9.81
                dFdx .= gHsx .* (dFdξ[1:neqns]) .+ dFdξ1[1:neqns]
                rhs_el[i,iel,1:neqns] .-= ω[i]*mesh.Δx[iel]/2*dFdx[1:neqns]
        end
    end
    rhs_diff_el = build_rhs_diff(SD, QT, PT, qp,  neqns, basis, ω, inputs[:νx], inputs[:νy], mesh, metrics, T)
    apply_boundary_conditions!(SD, rhs_el, qq, mesh, inputs, QT, metrics, basis.ψ, basis.dψ, ω, Δt*(floor(time/Δt)), neqns)
    for i=1:neqns
        idx = (i-1)*mesh.npoin
        qp[idx+1:i*mesh.npoin] .= qq[:,i]
    end
    RHS = DSS_rhs(SD, rhs_el .+ rhs_diff_el, mesh.connijk, mesh.nelem, mesh.npoin, neqns, mesh.nop, T)
    divive_by_mass_matrix!(RHS, M, QT,neqns)
    return RHS
end

function build_rhs(SD::NSD_2D, QT::Inexact, PT::ShallowWater, qp::Array, neqns, basis, ω, mesh::St_mesh, metrics::St_metrics, M, De, Le, time, inputs, Δt, T)
    F    = zeros(mesh.ngl,mesh.ngl,mesh.nelem, neqns)
    G    = zeros(mesh.ngl,mesh.ngl,mesh.nelem, neqns)
    F1    = zeros(mesh.ngl,mesh.ngl,mesh.nelem, neqns)
    G1    = zeros(mesh.ngl,mesh.ngl,mesh.nelem, neqns)
    rhs_el = zeros(mesh.ngl,mesh.ngl,mesh.nelem, neqns)
    qq = zeros(mesh.npoin,neqns)
    for i=1:neqns
        idx = (i-1)*mesh.npoin
        qq[:,i] .= qp[idx+1:i*mesh.npoin]
    end
    Fuser, Guser, Fuser1, Guser1 = user_flux(T, SD, qq, mesh)
    dFdx = zeros(neqns)
    dFdξ = zeros(neqns)
    dGdξ = zeros(neqns)
    dGdy = zeros(neqns)
    dFdη = zeros(neqns)
    dGdη = zeros(neqns)
    gHsx = zeros(neqns)
    gHsy = zeros(neqns)
    for iel=1:mesh.nelem

        for i=1:mesh.ngl
            for j=1:mesh.ngl
                ip = mesh.connijk[i,j,iel]
                F[i,j,iel,1] = Fuser[ip,1]
                F[i,j,iel,2] = Fuser[ip,2]
                F[i,j,iel,3] = Fuser[ip,3]
                
                F1[i,j,iel,1] = Fuser1[ip,1] 
                F1[i,j,iel,2] = Fuser1[ip,2] 
                F1[i,j,iel,3] = Fuser1[ip,3]  
               
                G[i,j,iel,1] = Guser[ip,1]
                G[i,j,iel,2] = Guser[ip,2]
                G[i,j,iel,3] = Guser[ip,3]

                G1[i,j,iel,1] = Guser1[ip,1] 
                G1[i,j,iel,2] = Guser1[ip,2] 
                G1[i,j,iel,3] = Guser1[ip,3] 
            end
        end

        for i=1:mesh.ngl
            for j=1:mesh.ngl
                dFdξ = zeros(T, neqns)
                dFdη = zeros(T, neqns)
                dGdξ = zeros(T, neqns)
                dGdη = zeros(T, neqns)
                dFdξ1 = zeros(T, neqns)
                dFdη1 = zeros(T, neqns)
                dGdξ1 = zeros(T, neqns)
                dGdη1 = zeros(T, neqns)
                for k = 1:mesh.ngl
                    dFdξ[1:neqns] .= dFdξ[1:neqns] .+ basis.dψ[k,i]*F[k,j,iel,1:neqns]
                    dFdη[1:neqns] .= dFdη[1:neqns] .+ basis.dψ[k,j]*F[i,k,iel,1:neqns]
                     
                    dFdξ1[1:neqns] .= dFdξ1[1:neqns] .+ basis.dψ[k,i]*F1[k,j,iel,1:neqns]
                    dFdη1[1:neqns] .= dFdη1[1:neqns] .+ basis.dψ[k,j]*F1[i,k,iel,1:neqns]    

                    dGdξ[1:neqns] .= dGdξ[1:neqns] .+ basis.dψ[k,i]*G[k,j,iel,1:neqns]
                    dGdη[1:neqns] .= dGdη[1:neqns] .+ basis.dψ[k,j]*G[i,k,iel,1:neqns]

                    dGdξ1[1:neqns] .= dGdξ1[1:neqns] .+ basis.dψ[k,i]*G1[k,j,iel,1:neqns]
                    dGdη1[1:neqns] .= dGdη1[1:neqns] .+ basis.dψ[k,j]*G1[i,k,iel,1:neqns]
                end
                ip = mesh.connijk[i,j,iel]
                x = mesh.x[ip]
                y = mesh.y[ip]
                Hb = bathymetry(x,y)
                Hs = max(qq[ip,1] - Hb,0.001)
                gHsx[1] = 1.0
                gHsx[2] = Hs * 9.81
                gHsx[3] = 1.0
                gHsy[1] = 1.0
                gHsy[2] = 1.0
                gHsy[3] = Hs * 9.81
                dFdx .= gHsx .* (dFdξ[1:neqns]*metrics.dξdx[i,j,iel] .+ dFdη[1:neqns]*metrics.dηdx[i,j,iel]) + dFdξ1[1:neqns]*metrics.dξdx[i,j,iel] .+ dFdη1[1:neqns]*metrics.dηdx[i,j,iel]
                dGdy .= gHsy .* (dGdξ[1:neqns]*metrics.dξdy[i,j,iel] .+ dGdη[1:neqns]*metrics.dηdy[i,j,iel]) + dGdξ1[1:neqns]*metrics.dξdy[i,j,iel] .+ dGdη1[1:neqns]*metrics.dηdy[i,j,iel]
                rhs_el[i,j,iel,1:neqns] .-= ω[i]*ω[j]*metrics.Je[i,j,iel]*(dFdx[1:neqns] .+ dGdy[1:neqns])
            end
        end
    end
    rhs_diff_el = build_rhs_diff(SD, QT, PT, qp,  neqns, basis, ω, inputs[:νx], inputs[:νy], mesh, metrics, T)
    apply_boundary_conditions!(SD, rhs_el, qq, mesh, inputs, QT, metrics, basis.ψ, basis.dψ, ω, Δt*(floor(time/Δt)), neqns)
    for i=1:neqns
        idx = (i-1)*mesh.npoin
        qp[idx+1:i*mesh.npoin] .= qq[:,i]
    end
    RHS = DSS_rhs(SD, rhs_el .+ rhs_diff_el, mesh.connijk, mesh.nelem, mesh.npoin, neqns, mesh.nop, T)
    divive_by_mass_matrix!(RHS, M, QT,neqns)
    return RHS
end

function build_rhs_diff(SD::NSD_1D, QT::Inexact, PT::AdvDiff, qp::Array, nvars, basis, ω, νx, νy, mesh::St_mesh, metrics::St_metrics, T)

    N           = mesh.ngl - 1
    qnel        = zeros(mesh.ngl, mesh.nelem)
    rhsdiffξ_el = zeros(mesh.ngl, mesh.nelem)
    
    #
    # Add diffusion ν∫∇ψ⋅∇q (ν = const for now)
    #
    for iel=1:mesh.nelem
        Jac = mesh.Δx[iel]/2.0
        dξdx = 2.0/mesh.Δx[iel]
        
        for i=1:mesh.ngl
            qnel[i,iel,1] = qp[mesh.connijk[i,iel], 1]
        end
        
        for k = 1:mesh.ngl
            ωJk = ω[k]*Jac
            
            dqdξ = 0.0
            for i = 1:mesh.ngl
                dqdξ = dqdξ + basis.dψ[i,k]*qnel[i,iel]
            end
            dqdx = dqdξ*dξdx            
            ∇ξ∇q = dξdx*dqdx
            
            for i = 1:mesh.ngl
                hll     = basis.ψ[k,k]
                dhdξ_ik = basis.dψ[i,k]
                
                rhsdiffξ_el[i, iel] -= ωJk * basis.dψ[i,k] * basis.ψ[k,k]*∇ξ∇q
            end
        end
    end
      
    return rhsdiffξ_el*νx
end

function build_rhs_diff(SD::NSD_2D, QT::Inexact, PT::AdvDiff, qp::Array, nvars, basis, ω, νx, νy, mesh::St_mesh, metrics::St_metrics, T)

    N = mesh.ngl - 1
    
    qnel = zeros(mesh.ngl,mesh.ngl,mesh.nelem)
    
    rhsdiffξ_el = zeros(mesh.ngl,mesh.ngl,mesh.nelem)
    rhsdiffη_el = zeros(mesh.ngl,mesh.ngl,mesh.nelem)
    
    #
    # Add diffusion ν∫∇ψ⋅∇q (ν = const for now)
    #
    for iel=1:mesh.nelem

        for j=1:mesh.ngl, i=1:mesh.ngl
            m = mesh.connijk[i,j,iel]            
            qnel[i,j,iel,1] = qp[m,1]
        end
        
        for k = 1:mesh.ngl, l = 1:mesh.ngl
            ωJkl = ω[k]*ω[l]*metrics.Je[k, l, iel]
            
            dqdξ = 0.0
            dqdη = 0.0
            for i = 1:mesh.ngl
                dqdξ = dqdξ + basis.dψ[i,k]*qnel[i,l,iel]
                dqdη = dqdη + basis.dψ[i,l]*qnel[k,i,iel]
            end
            dqdx = dqdξ*metrics.dξdx[k,l,iel] + dqdη*metrics.dηdx[k,l,iel]
            dqdy = dqdξ*metrics.dξdy[k,l,iel] + dqdη*metrics.dηdy[k,l,iel]
            
            ∇ξ∇q_kl = metrics.dξdx[k,l,iel]*dqdx + metrics.dξdy[k,l,iel]*dqdy
            ∇η∇q_kl = metrics.dηdx[k,l,iel]*dqdx + metrics.dηdy[k,l,iel]*dqdy
            
            for i = 1:mesh.ngl
                hll,     hkk     =  basis.ψ[l,l],  basis.ψ[k,k]
                dhdξ_ik, dhdη_il = basis.dψ[i,k], basis.dψ[i,l]
                
                rhsdiffξ_el[i,l,iel] -= ωJkl*dhdξ_ik*hll*∇ξ∇q_kl
                rhsdiffη_el[k,i,iel] -= ωJkl*hkk*dhdη_il*∇η∇q_kl
            end
        end
    end

    return (rhsdiffξ_el*νx + rhsdiffη_el*νy)
    
end

function build_rhs_diff(SD::NSD_2D, QT, PT::LinearCLaw, qp, neqns, basis, ω, νx, νy, mesh::St_mesh, metrics::St_metrics, T)
    
    N = mesh.ngl - 1

    qnel = zeros(mesh.ngl,mesh.ngl,mesh.nelem, neqns)

    rhsdiffξ_el = zeros(mesh.ngl, mesh.ngl, mesh.nelem, neqns)
    rhsdiffη_el = zeros(mesh.ngl, mesh.ngl, mesh.nelem, neqns)
    qq = zeros(mesh.npoin,neqns)

    #
    # qp[1:npoin]         <-- qq[1:npoin, "p"]
    # qp[npoin+1:2npoin]  <-- qq[1:npoin, "u"]
    # qp[2npoin+1:3npoin] <-- qq[1:npoin, "v"]
    #
    for i=1:neqns
        idx = (i-1)*mesh.npoin
        qq[:,i] = qp[idx+1:i*mesh.npoin]
    end
    #
    # Add diffusion ν∫∇ψ⋅∇q (ν = const for now)
    #
    for iel=1:mesh.nelem

        for j=1:mesh.ngl, i=1:mesh.ngl
            m = mesh.connijk[i,j,iel]
            qnel[i,j,iel,1:neqns] = qq[m,1:neqns]
        end

        for k = 1:mesh.ngl, l = 1:mesh.ngl
            ωJkl = ω[k]*ω[l]*metrics.Je[k, l, iel]

            for ieq = 1:neqns
                dqdξ = 0.0
                dqdη = 0.0
                for i = 1:mesh.ngl
                    dqdξ = dqdξ + basis.dψ[i,k]*qnel[i,l,iel,ieq]
                    dqdη = dqdη + basis.dψ[i,l]*qnel[k,i,iel,ieq]
                end
                dqdx = dqdξ*metrics.dξdx[k,l,iel] + dqdη*metrics.dηdx[k,l,iel]
                dqdy = dqdξ*metrics.dξdy[k,l,iel] + dqdη*metrics.dηdy[k,l,iel]

                ∇ξ∇q_kl = metrics.dξdx[k,l,iel]*dqdx + metrics.dξdy[k,l,iel]*dqdy
                ∇η∇q_kl = metrics.dηdx[k,l,iel]*dqdx + metrics.dηdy[k,l,iel]*dqdy

                for i = 1:mesh.ngl

                    hll,     hkk     = basis.ψ[l,l],  basis.ψ[k,k]
                    dhdξ_ik, dhdη_il = basis.dψ[i,k], basis.dψ[i,l]

                    rhsdiffξ_el[i,l,iel, ieq] -= ωJkl*dhdξ_ik*hll*∇ξ∇q_kl
                    rhsdiffη_el[k,i,iel, ieq] -= ωJkl*hkk*dhdη_il*∇η∇q_kl
                end
            end
        end
     end

    return (rhsdiffξ_el*νx + rhsdiffη_el*νy)

end

function build_rhs_diff(SD::NSD_1D, QT, PT::ShallowWater, qp, neqns, basis, ω, νx, νy, mesh::St_mesh, metrics::St_metrics, T)

    N = mesh.ngl - 1

    qnel = zeros(mesh.ngl, mesh.nelem, neqns)

    rhsdiffξ_el = zeros(mesh.ngl, mesh.nelem, neqns)
    qq = zeros(mesh.npoin,neqns)

    #
    # qp[1:npoin]         <-- qq[1:npoin, "p"]
    # qp[npoin+1:2npoin]  <-- qq[1:npoin, "u"]
    # qp[2npoin+1:3npoin] <-- qq[1:npoin, "v"]
    #
    for i=1:neqns
        idx = (i-1)*mesh.npoin
        qq[:,i] = qp[idx+1:i*mesh.npoin]
    end
    #
    # Add diffusion ν∫∇ψ⋅∇q (ν = const for now)
    #
    for iel=1:mesh.nelem
        Jac = mesh.Δx[iel]/2.0
        for i=1:mesh.ngl
            m = mesh.conn[i,iel]
            qnel[i,iel,1] = qq[m,1]
            qnel[i,iel,2] = qq[m,2]/qq[m,1]
        end

        for k = 1:mesh.ngl
            ωJkl = ω[k]*Jac

            for ieq = 1:neqns
                dqdξ = 0.0
                for i = 1:mesh.ngl
                    dqdξ = dqdξ + basis.dψ[i,k]*qnel[k,iel,ieq]
                end
                dqdx = νx * (dqdξ)
                if (ieq > 1)
                    ip = mesh.conn[k,iel]
                    x = mesh.x[ip]
                    Hb = bathymetry(x)
                    Hs = qq[ip,1] - Hb
                    dqdx = dqdx * Hs
                end

                ∇ξ∇q_kl =  dqdx

                for i = 1:mesh.ngl

                    hkk     = basis.ψ[k,k]
                    dhdξ_ik = basis.dψ[i,k]

                    rhsdiffξ_el[i,iel,ieq] -= ωJkl*dhdξ_ik*hkk*∇ξ∇q_kl
                end
            end
        end
     end

    return (rhsdiffξ_el)

end

function build_rhs_diff(SD::NSD_2D, QT, PT::ShallowWater, qp, neqns, basis, ω, νx, νy, mesh::St_mesh, metrics::St_metrics, T)
    
    N = mesh.ngl - 1

    qnel = zeros(mesh.ngl,mesh.ngl,mesh.nelem, neqns)

    rhsdiffξ_el = zeros(mesh.ngl, mesh.ngl, mesh.nelem, neqns)
    rhsdiffη_el = zeros(mesh.ngl, mesh.ngl, mesh.nelem, neqns)
    qq = zeros(mesh.npoin,neqns)

    #
    # qp[1:npoin]         <-- qq[1:npoin, "p"]
    # qp[npoin+1:2npoin]  <-- qq[1:npoin, "u"]
    # qp[2npoin+1:3npoin] <-- qq[1:npoin, "v"]
    #
    for i=1:neqns
        idx = (i-1)*mesh.npoin
        qq[:,i] = qp[idx+1:i*mesh.npoin]
    end
    #
    # Add diffusion ν∫∇ψ⋅∇q (ν = const for now)
    #
    for iel=1:mesh.nelem

        for j=1:mesh.ngl, i=1:mesh.ngl
            m = mesh.connijk[i,j,iel]
            qnel[i,j,iel,1] = qq[m,1]
            qnel[i,j,iel,2] = qq[m,2]/qq[m,1]
            qnel[i,j,iel,3] = qq[m,3]/qq[m,1]
        end

        for k = 1:mesh.ngl, l = 1:mesh.ngl
            ωJkl = ω[k]*ω[l]*metrics.Je[k, l, iel]

            for ieq = 1:neqns
                dqdξ = 0.0
                dqdη = 0.0
                for i = 1:mesh.ngl
                    dqdξ = dqdξ + basis.dψ[i,k]*qnel[i,l,iel,ieq]
                    dqdη = dqdη + basis.dψ[i,l]*qnel[k,i,iel,ieq]
                end
                dqdx = νx * (dqdξ*metrics.dξdx[k,l,iel] + dqdη*metrics.dηdx[k,l,iel])
                dqdy = νy * (dqdξ*metrics.dξdy[k,l,iel] + dqdη*metrics.dηdy[k,l,iel])
                if (ieq > 1)
                    ip = mesh.connijk[k,l,iel]
                    x = mesh.x[ip]
                    y = mesh.y[ip]
                    Hb = bathymetry(x,y)
                    Hs = qq[ip,1] - Hb
                    dqdx = dqdx * Hs
                    dqdy = dqdy * Hs
                end                

                ∇ξ∇q_kl =  (metrics.dξdx[k,l,iel]*dqdx + metrics.dξdy[k,l,iel]*dqdy)
                ∇η∇q_kl =  (metrics.dηdx[k,l,iel]*dqdx + metrics.dηdy[k,l,iel]*dqdy)

                for i = 1:mesh.ngl

                    hll,     hkk     = basis.ψ[l,l],  basis.ψ[k,k]
                    dhdξ_ik, dhdη_il = basis.dψ[i,k], basis.dψ[i,l]
     
                    rhsdiffξ_el[i,l,iel, ieq] -= ωJkl*dhdξ_ik*hll*∇ξ∇q_kl
                    rhsdiffη_el[k,i,iel, ieq] -= ωJkl*hkk*dhdη_il*∇η∇q_kl
                end
            end
        end
     end

    return (rhsdiffξ_el + rhsdiffη_el)

end

function build_rhs_source(SD::NSD_2D,
                          QT::Inexact,
                          q::Array,
                          mesh::St_mesh,
                          M::AbstractArray, #M is sparse for exact integration
                          T)

    S = user_source(q, mesh, T)
    
    return M.*S    
end

function build_rhs_source(SD::NSD_2D,
                          QT::Exact,
                          q::Array,
                          mesh::St_mesh,
                          M::Matrix, #M is sparse for exact integration
                          T)

    S = user_source(q, mesh, T)
    
    return M*S   
end

