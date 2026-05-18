"""
    multi_i_ode!(du, u, p, t)
Defines the ODE system for a multi-compartment infection model with exposed and multiple infectious stages.

# Arguments
    - `du`: A vector to store the derivatives of the state variables.
    - `u`: A vector containing the current values of the state variables, ordered as (E, I₁, I₂, I₃, I₄, I₅, I₆, I₇, R₁, R₂, R, I).  I is the total number of infectious individuals, calculated as the sum of I₁ through I₇.
    - `p`: A tuple containing the parameters of the model, ordered as (αs, γ, ν₁, ν₂, ν₃, ν₄, ν₅, ν₆, ν₇, η₁, η₂, param_change_times).
    - `t`: The current time point for which the derivatives are being calculated.   

# Returns   
    - The function updates the `du` vector in place with the derivatives of the state variables according to the defined ODE system.
"""
function multi_i_ode!(
    du,
    u,
    p,
    t
)
    (; αs, γ, νs, ηs, param_change_times) = p

    n_I = length(νs)
    n_R = length(ηs)

    E_idx = 1
    I_idx = 2:(n_I + 1)
    R_stage_idx = (n_I + 2):(n_I + n_R + 1)
    R_idx = n_I + n_R + 2

    ind_t = max(searchsortedlast(param_change_times, t), 1)
    α = αs[ind_t]

    E = u[E_idx]
    Is = @view u[I_idx]

    I_total = sum(Is)
    exposed_rate = α * I_total

    @inbounds begin
        du[E_idx] = exposed_rate - γ * E
        du[I_idx[1]] = γ * E - νs[1] * u[I_idx[1]]
        for j in 2:n_I
            du[I_idx[j]] =
                νs[j - 1] * u[I_idx[j - 1]] -
                νs[j] * u[I_idx[j]]
        end
        I_to_R_rate = νs[end] * u[I_idx[end]]
        if n_R == 0
            du[R_idx] = I_to_R_rate
        else
            du[R_stage_idx[1]] =
                I_to_R_rate -
                ηs[1] * u[R_stage_idx[1]]
            for j in 2:n_R
                du[R_stage_idx[j]] =
                    ηs[j - 1] * u[R_stage_idx[j - 1]] -
                    ηs[j] * u[R_stage_idx[j]]
            end
            du[R_idx] = ηs[end] * u[R_stage_idx[end]]
        end
    end
end


function setup_multi_i_ode_problem(;
    obstime_wastewater,
    Rₜ_module,
    init_compartment_module,
    γ,
    ν,
    η
)

    n_I = init_compartment_module.ode_config.n_I
    n_R = init_compartment_module.ode_config.n_R

    νs = fill(n_I * ν, n_I)

    if n_R == 0
        ηs = Float64[]
    else
        ηs = fill(n_R * η, n_R)
    end

    αs = Rₜ_module.Rₜ .* ν

    u0 = vcat(
        init_compartment_module.compartment₁.E,
        init_compartment_module.compartment₁.I,
        init_compartment_module.compartment₁.R_stage,
        init_compartment_module.compartment₁.R
    )

    p = (
        αs = αs,
        γ = γ,
        νs = νs,
        ηs = ηs,
        param_change_times = Rₜ_module.timebreaks
    )

    tspan = (
        minimum(obstime_wastewater),
        maximum(obstime_wastewater)
    )

    return ODEProblem(
        multi_i_ode!,
        u0,
        tspan,
        p,
        saveat = obstime_wastewater
    )
end


function unpack_multi_i_solution(sol_array, init_compartment_module)

    n_I = init_compartment_module.ode_config.n_I
    n_R = init_compartment_module.ode_config.n_R

    E = sol_array[1, :]

    I = [
        sol_array[j + 1, :]
        for j in 1:n_I
    ]

    R_stage = [
        sol_array[n_I + j + 1, :]
        for j in 1:n_R
    ]

    R = sol_array[n_I + n_R + 2, :]

    I_means = vec(sum(sol_array[2:(n_I + 1), :], dims = 1))

    return (
        states = (
            E = E,
            I = I,
            R_stage = R_stage,
            R = R
        ),
        I_means = I_means
    )
end