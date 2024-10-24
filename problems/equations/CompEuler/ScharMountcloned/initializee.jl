using Base
using DelimitedFiles
using Jexpresso  # Assuming Jexpresso is the module where NSD_2D, CompEuler, etc., are defined

# Function to read velocity data from a file
function read_velocity(file_path::String)
    data = readdlm(file_path, ',')
    distance = data[2:end, 1]  # Skip header and get distance
    u = data[2:end, 2]         # Skip header and get u component
    v = data[2:end, 3]         # Skip header and get v component
    return distance, u, v
end

# Function to initialize fields for 2D CompEuler with θ equation
function initialize(SD::NSD_2D, PT::CompEuler, mesh::St_mesh, inputs::Dict, OUTPUT_DIR::String, TFloat, u::Vector{TFloat}, v::Vector{TFloat})
    """
    Initialize fields for 2D CompEuler with θ equation
    """
    @info " Initialize fields for 2D CompEuler with θ equation ........................ "

    qvars = ("dρ", "dρu", "dρv", "dρθ")
    q = define_q(SD, mesh.nelem, mesh.npoin, mesh.ngl, qvars, TFloat, inputs[:backend]; neqs=length(qvars))

    if (inputs[:backend] == CPU())
        PhysConst = PhysicalConst{Float64}()
        θref = 280.0 # K
        θ0 = 280.0
        T0 = θ0
        p0 = 100000.0

        N = 0.01
        N2 = N * N

        for iel_g = 1:mesh.nelem
            for j = 1:mesh.ngl, i = 1:mesh.ngl
                ip = mesh.connijk[iel_g, i, j]
                y = mesh.y[ip]
                θ = θref * exp(N2 * y / PhysConst.g)
                p = p0 * (1.0 + PhysConst.g2 * (exp(-y * N2 / PhysConst.g) - 1.0) / (PhysConst.cp * θref * N2))^PhysConst.cpoverR
                ρ = perfectGasLaw_θPtoρ(PhysConst; θ = θ, Press = p) # kg/m³
                ρref = perfectGasLaw_θPtoρ(PhysConst; θ = θ, Press = p) # kg/m³

                if inputs[:SOL_VARS_TYPE] == PERT()
                    q.qn[ip, 1] = ρ - ρref
                    q.qn[ip, 2] = ρ * u[ip] - ρref * u[ip]
                    q.qn[ip, 3] = ρ * v[ip] - ρref * v[ip]
                    q.qn[ip, 4] = ρ * θ - ρref * θ
                    q.qn[ip, end] = p
                else
                    q.qn[ip, 1] = ρ
                    q.qn[ip, 2] = ρ * u[ip]
                    q.qn[ip, 3] = ρ * v[ip]
                    q.qn[ip, 4] = ρ * θ
                    q.qn[ip, end] = p
                end

                # Store initial background state for plotting and analysis of perturbations
                q.qe[ip, 1] = ρref
                q.qe[ip, 2] = ρref * u[ip]
                q.qe[ip, 3] = ρref * v[ip]
                q.qe[ip, 4] = ρref * θ
                q.qe[ip, end] = p
            end
        end
    else
        if (inputs[:SOL_VARS_TYPE] == PERT())
            lpert = true
        else
            lpert = false
        end
        PhysConst = PhysicalConst{TFloat}()
        θref = TFloat(280.0) # K
        θ0 = TFloat(280.0)
        T0 = θ0
        p0 = TFloat(100000.0)

        N = TFloat(0.01)
        N2 = TFloat(N * N)

        k = initialize_gpu!(inputs[:backend])
        k(q.qn, q.qe, mesh.x, mesh.y, θref, θ0, T0, p0, N, N2, PhysConst, lpert, u, v; ndrange = (mesh.npoin))
    end
    @info "Initialize fields for system of 2D CompEuler with θ equation ........................ DONE"

    return q
end

@kernel function initialize_gpu!(qn, qe, x, y, θref, θ0, T0, p0, N, N2, PhysConst, lpert, u, v)
    ip = @index(Global, Linear)

    x = x[ip]
    y = y[ip]
    T = eltype(x)

    θ = θref * exp(N2 * y / PhysConst.g)
    p = p0 * (T(1.0) + PhysConst.g2 * (exp(-y * N2 / PhysConst.g) - T(1.0)) / (PhysConst.cp * θref * N2))^PhysConst.cpoverR

    ρ = perfectGasLaw_θPtoρ(PhysConst; θ = θ, Press = p) # kg/m³
    ρref = perfectGasLaw_θPtoρ(PhysConst; θ = θ, Press = p)

    qn[ip, 1] = ρ - ρref
    qn[ip, 2] = ρ * u[ip] - ρref * u[ip]
    qn[ip, 3] = ρ * v[ip] - ρref * v[ip]
    qn[ip, 4] = ρ * θ - ρref * θ
    qn[ip, end] = p

    qe[ip, 1] = ρref
    qe[ip, 2] = ρref * u[ip]
    qe[ip, 3] = ρref * v[ip]
    qe[ip, 4] = ρref * θ
    qe[ip, end] = p
end

# Example usage
distance, u, v = read_velocity("velocity_data.csv")
q = initialize(SD, PT, mesh, inputs, OUTPUT_DIR, Float64, u, v)
