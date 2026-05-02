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

    (αs, γ, ν₁, ν₂, ν₃, ν₄, ν₅, ν₆, ν₇, η₁, η₂, param_change_times) = p
    (E, I₁, I₂, I₃, I₄, I₅, I₆, I₇, R₁, R₂, R) = u

    ind_t = max(searchsortedlast(param_change_times, t), 1)
    α = αs[ind_t]

    I = I₁ + I₂ + I₃ + I₄ + I₅ + I₆ + I₇

    exposed_rate = α * I
    I₁_to_I₂_rate = ν₁ * I₁
    I₂_to_I₃_rate = ν₂ * I₂
    I₃_to_I₄_rate = ν₃ * I₃
    I₄_to_I₅_rate = ν₄ * I₄
    I₅_to_I₆_rate = ν₅ * I₅
    I₆_to_I₇_rate = ν₆ * I₆
    I₇_to_R₁_rate = ν₇ * I₇
    R₁_to_R₂_rate = η₁ * R₁
    R₂_to_R_rate = η₂ * R₂

    @inbounds begin
        du[1] = exposed_rate - γ * E  # dE/dt
        du[2] = γ * E - I₁_to_I₂_rate  # dI₁/dt
        du[3] = I₁_to_I₂_rate - I₂_to_I₃_rate  # dI₂/dt
        du[4] = I₂_to_I₃_rate - I₃_to_I₄_rate  # dI₃/dt
        du[5] = I₃_to_I₄_rate - I₄_to_I₅_rate  # dI₄/dt
        du[6] = I₄_to_I₅_rate - I₅_to_I₆_rate  # dI₅/dt
        du[7] = I₅_to_I₆_rate - I₆_to_I₇_rate  # dI₆/dt
        du[8] = I₆_to_I₇_rate - I₇_to_R₁_rate  # dI₇/dt
        du[9] = I₇_to_R₁_rate - R₁_to_R₂_rate  # dR₁/dt
        du[10] = R₁_to_R₂_rate - R₂_to_R_rate  # dR₂/dt
        du[11] = R₂_to_R_rate  # dR/dt
    end

end