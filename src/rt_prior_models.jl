"""
    Rₜ_rw_prior_model(...)
A random walk prior model for Rₜ, where the log of Rₜ follows a random walk with specified priors on the initial value and the standard deviation of the random walk increments.

# Arguments
- `timebreaks`: An array of times (in days) where Rₜ is allowed to change.  Typically, these would be weekly change points.
- `log_R₁_prior`: An array of length 2 specifying the mean and standard deviation of the prior on log(R₁).  In the form (mean = ..., sd = ...).
- `σ_Rₜ_prior`: An array of length 2 specifying the mean and standard deviation of the prior on σ_Rₜ on the log scale.  In the form (mean = ..., sd = ...).

# Returns
- A named tuple containing:
    - `Rₜ`: A vector of Rₜ values corresponding to each time break point.
    - `σ_Rₜ`: The standard deviation of the random walk increments for log(Rₜ).
    - `timebreaks`: The input time break points for Rₜ changes.
"""
@model function Rₜ_rw_prior_model(
    timebreaks;
    log_R₁_prior = (mean = log(1.0), sd = 0.1),
    σ_Rₜ_prior = (mean = log(0.1), sd = 0.1)
)
    # Prelims-----------------------------
     n_timebreaks = length(timebreaks)
     # PRIORS-----------------------------
     Rₜ_params_non_centered ~ MvNormal(zeros(n_timebreaks + 1), I) # +1 for σ

     # TRANSFORMATIONS-----------------------------
    σ_Rₜ_non_centered = Rₜ_params_non_centered[1]
    Rₜ_init_non_centered = Rₜ_params_non_centered[2]
    Rₜ_steps_non_centered = Rₜ_params_non_centered[3:end]
    R₁ = exp(log_R₁_prior.mean + log_R₁_prior.sd * Rₜ_init_non_centered)
    σ_Rₜ = exp(σ_Rₜ_non_centered * σ_Rₜ_prior.sd + σ_Rₜ_prior.mean)

    Rₜ_no_init = exp.(log(R₁) .+ cumsum(Rₜ_steps_non_centered) * σ_Rₜ)
    Rₜ = vcat(R₁, Rₜ_no_init)
     return (
        Rₜ = Rₜ,
        timebreaks = timebreaks,
        params = (
            σ_Rₜ = σ_Rₜ
        )
    )
end

"""
    Rₜ_ibm_prior_model(...)
An integrated Brownian motion (IBM) based prior model for Rₜ, where the log of Rₜ follows an integrated Brownian motion process, also known as Jessalyn's Prior.

# Arguments
- `timebreaks`: An array of times (in days) where Rₜ is allowed to change.  Typically, these would be weekly change points.
- `log_R₁_prior`: An array of length 2 specifying the mean and standard deviation of the prior on log(R₁).  In the form (mean = ..., sd = ...).
- `σ_R₁_prior`: An array of length 2 specifying the mean and standard deviation of the prior on σ_R₁ on the log scale.  In the form (mean = ..., sd = ...).
- `σ_Rₜ_prior`: An array of length 2 specifying the mean and standard deviation of the prior on σ_Rₜ on the log scale.  In the form (mean = ..., sd = ...).
- `log_R₁_prime_prior`: An array of length 2 specifying the mean and standard deviation of the prior on the initial slope of log(Rₜ) on the log scale.  In the form (mean = ..., sd = ...). 

# Returns
- A named tuple containing:
    - `Rₜ`: A vector of Rₜ values corresponding to each time break point.
    - `timebreaks`: The input time break points for Rₜ changes.
    - `params`: A named tuple containing the parameters σ_Rₜ, σ_R₁, and log_Rₜ_prime.
"""
@model function Rₜ_ibm_prior_model(
    timebreaks;
    log_R₁_prior = (mean = log(1.0), sd = 0.1),
    σ_R₁_prior = (mean = log(0.4), sd = 0.25),
    σ_Rₜ_prior = (mean = log(0.075), sd = 0.04),
    log_R₁_prime_prior = (mean = 0.0, sd = 0.01)
)
    # Prelims-----------------------------
    n_timebreaks = length(timebreaks)

    # PRIORS-----------------------------
    σ_R₁_non_centered ~ Normal()
    σ_Rₜ_non_centered ~ Normal()
    log_R₁_non_centered ~ Normal()
    slope₁_non_centered ~ Normal()
    Z₁ ~ filldist(Normal(), 2) # For the initial slope and value
    Z ~ filldist(Normal(), 2, n_timebreaks - 1) # For the increments

    # TRANSFORMATIONS-----------------------------
    σ_R₁ = exp(σ_R₁_prior.mean + σ_R₁_prior.sd * σ_R₁_non_centered)
    σ_Rₜ = exp(σ_Rₜ_prior.mean + σ_Rₜ_prior.sd * σ_Rₜ_non_centered)
    log_R₁ = log_R₁_prior.mean + log_R₁_prior.sd * log_R₁_non_centered
    log_R₁_prime = log_R₁_prime_prior.mean + log_R₁_prime_prior.sd * slope₁_non_centered

    μ₁ = [log_R₁_prime, log_R₁]
    L₁ = [
        σ_R₁        0.0
        σ_R₁^3 / 2  σ_R₁^3 / (2 * sqrt(3))
    ]
    IBM₁ = μ₁ + L₁ * Z₁
    
    L = [
        σ_Rₜ        0.0
        σ_Rₜ^3 / 2  σ_Rₜ^3 / (2 * sqrt(3))
    ]
    ε = L * Z
    ε1 = vec(ε[1, :])
    ε2 = vec(ε[2, :])

    log_Rₜ_prime = vcat(IBM₁[1], IBM₁[1] .+ cumsum(ε1))

    increments = σ_Rₜ^2 .* log_Rₜ_prime[1:end-1] .+ ε2
    log_Rₜ = vcat(IBM₁[2], IBM₁[2] .+ cumsum(increments))

    Rₜ = exp.(log_Rₜ)

    return (
        Rₜ = Rₜ,
        timebreaks = timebreaks,
        params = (
            σ_Rₜ = σ_Rₜ,
            σ_R₁ = σ_R₁,
            log_Rₜ_prime = log_Rₜ_prime
        )
    )
end


@model function Rₜ_ibm_prior_model_loop(
    timebreaks;
    log_R₁_prior = (mean = log(1.0), sd = 0.1),
    σ_R₁_prior = (mean = log(0.4), sd = 0.25),
    σ_Rₜ_prior = (mean = log(0.075), sd = 0.04),
    log_R₁_prime_prior = (mean = 0.0, sd = 0.01)
)
    n_timebreaks = length(timebreaks)

    σ_R₁_non_centered ~ Normal()
    σ_Rₜ_non_centered ~ Normal()
    log_R₁_non_centered ~ Normal()
    slope₁_non_centered ~ Normal()

    Z₁ ~ filldist(Normal(), 2)
    Z ~ filldist(Normal(), 2, n_timebreaks - 1)

    σ_R₁ = exp(σ_R₁_prior.mean + σ_R₁_prior.sd * σ_R₁_non_centered)
    σ_Rₜ = exp(σ_Rₜ_prior.mean + σ_Rₜ_prior.sd * σ_Rₜ_non_centered)
    log_R₁ = log_R₁_prior.mean + log_R₁_prior.sd * log_R₁_non_centered
    log_R₁_prime = log_R₁_prime_prior.mean + log_R₁_prime_prior.sd * slope₁_non_centered

    μ₁ = [log_R₁_prime, log_R₁]
    L₁ = [
        σ_R₁                zero(σ_R₁)
        σ_R₁^3 / 2          σ_R₁^3 / (2 * sqrt(3))
    ]
    z₁ = μ₁ + L₁ * Z₁

    L = [
        σ_Rₜ                zero(σ_Rₜ)
        σ_Rₜ^3 / 2          σ_Rₜ^3 / (2 * sqrt(3))
    ]
    ε = L * Z

    log_Rₜ_prime = Vector{typeof(log_R₁)}(undef, n_timebreaks)
    log_Rₜ = Vector{typeof(log_R₁)}(undef, n_timebreaks)

    log_Rₜ_prime[1] = z₁[1]
    log_Rₜ[1] = z₁[2]

    for t in 2:n_timebreaks
        ε1 = ε[1, t - 1]
        ε2 = ε[2, t - 1]

        log_Rₜ_prime[t] = log_Rₜ_prime[t - 1] + ε1
        log_Rₜ[t] = log_Rₜ[t - 1] + σ_Rₜ^2 * log_Rₜ_prime[t - 1] + ε2
    end

    Rₜ = exp.(log_Rₜ)

    return (
        Rₜ = Rₜ,
        timebreaks = timebreaks,
        params = (
            σ_Rₜ = σ_Rₜ,
            σ_R₁ = σ_R₁,
            log_Rₜ_prime = log_Rₜ_prime
        )
    )
end