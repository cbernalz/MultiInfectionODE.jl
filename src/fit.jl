"""

    fit(...)
Sampler function to fit the model.  

# Arguments

"""
function fit(
    data_wastewater,
    obstime_wastewater,
    s,
    Rₜ_prior_model,
    γ_prior = (mean = log(1/7), sd = 0.25),
    ν_prior = (mean = log(1/7), sd = 0.25),
    η_prior = (mean = log(2/18), sd = 0.25),
    σ_ww_prior = (mean = log(0.1), sd = 0.25),
    compartment_priors = (
        E₁_prior = (mean = log(1000.0), sd = 0.5),
        I₁_prior = (mean = log(5000.0), sd = 0.5),
        I₂_prior = (mean = log(0.0), sd = 0.5),
        I₃_prior = (mean = log(0.0), sd = 0.5),
        I₄_prior = (mean = log(0.0), sd = 0.5),
        I₅_prior = (mean = log(0.0), sd = 0.5),
        I₆_prior = (mean = log(0.0), sd = 0.5),
        I₇_prior = (mean = log(0.0), sd = 0.5),
        R₁_prior = (mean = log(0.0), sd = 0.5),
        R₂_prior = (mean = log(0.0), sd = 0.5)
    ),
    priors_only::Bool=false,
    n_samples::Int64=500, n_chains::Int64=1,
    n_discard_initial::Int64=0, seed::Int64=2024,
    init_params=nothing
)

    # Obstime processing-----------------------------
    obstime_wastewater = convert(Vector{Float64}, obstime_wastewater)

    # Model and Sampling-----------------------------
    my_model = multi_infection_ode_model(
        data_wastewater = data_wastewater,
        obstime_wastewater = obstime_wastewater,
        s = s,
        Rₜ_prior_model = Rₜ_prior_model,
        γ_prior = γ_prior,
        ν_prior = ν_prior,
        η_prior = η_prior,
        σ_ww_prior = σ_ww_prior,
        compartment_priors = compartment_priors
    )
    # Sampling
    if priors_only
        Random.seed!(seed)
        samples = sample(my_model, Prior(), MCMCThreads(), 400, n_chains)
    else
        Random.seed!(seed)
        # Optimize
        if init_params === nothing
            samples = sample(my_model, NUTS(), MCMCThreads(), n_samples, n_chains, discard_initial = n_discard_initial)
        else
            samples = sample(my_model, NUTS(), MCMCThreads(), n_samples, n_chains, discard_initial = n_discard_initial, init_params = init_params)
        end
    end 

    return(samples)
end











