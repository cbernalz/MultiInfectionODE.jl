
function MultiInfectionODE_fit_generate(
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
    );
    n_samples::Int64 = 500, n_chains::Int64 = 1,
    n_discard_initial::Int64 = 0, seed::Int64 = 2024,
    init_params = nothing,
    forecast::Bool = false, forecast_days::Int64 = 14
)

    ## Obstime processing -----------------------------
    obstime_wastewater = convert(Vector{Float64}, obstime_wastewater)

    if forecast
        last_value = obstime_wastewater[end]

        obstime_wastewater_pred = vcat(
            obstime_wastewater,
            collect((last_value + 1):(last_value + forecast_days))
        )

        data_wastewater_fit = data_wastewater

        # Placeholder values only so generated_quantities can run on forecast times.
        # Predictive simulation uses my_model_predictive with missing observations.
        data_wastewater_gq = vcat(
            data_wastewater,
            repeat([data_wastewater[end]], forecast_days)
        )

        missing_data_wastewater = repeat([missing], length(obstime_wastewater_pred))
    else
        obstime_wastewater_pred = obstime_wastewater
        data_wastewater_fit = data_wastewater
        data_wastewater_gq = data_wastewater
        missing_data_wastewater = repeat([missing], length(obstime_wastewater_pred))
    end


    ## Models ------------------------------
    my_model_fit = multi_infection_ode_model(
        data_wastewater = data_wastewater_fit,
        obstime_wastewater = obstime_wastewater,
        s = s,
        Rₜ_prior_model = Rₜ_prior_model,
        γ_prior = γ_prior,
        ν_prior = ν_prior,
        η_prior = η_prior,
        σ_ww_prior = σ_ww_prior,
        compartment_priors = compartment_priors
    )

    my_model_gq = multi_infection_ode_model(
        data_wastewater = data_wastewater_gq,
        obstime_wastewater = obstime_wastewater_pred,
        s = s,
        Rₜ_prior_model = Rₜ_prior_model,
        γ_prior = γ_prior,
        ν_prior = ν_prior,
        η_prior = η_prior,
        σ_ww_prior = σ_ww_prior,
        compartment_priors = compartment_priors
    )

    my_model_predictive = multi_infection_ode_model(
        data_wastewater = missing_data_wastewater,
        obstime_wastewater = obstime_wastewater_pred,
        s = s,
        Rₜ_prior_model = Rₜ_prior_model,
        γ_prior = γ_prior,
        ν_prior = ν_prior,
        η_prior = η_prior,
        σ_ww_prior = σ_ww_prior,
        compartment_priors = compartment_priors
    )


    ## Posterior Sampling ------------------------------
    println("Sampling from posterior...")
    Random.seed!(seed)

    if init_params === nothing
        samples = sample(
            my_model_fit,
            NUTS(),
            MCMCThreads(),
            n_samples,
            n_chains;
            discard_initial = n_discard_initial
        )
    else
        samples = sample(
            my_model_fit,
            NUTS(),
            MCMCThreads(),
            n_samples,
            n_chains;
            discard_initial = n_discard_initial,
            init_params = init_params
        )
    end

    posterior_samples_df = DataFrame(samples)
    println("Posterior sampling complete.")


    ## Posterior Predictive and Generated Quantities ------------------------------
    posterior_gq_raw = generated_quantities(my_model_gq, samples)
    posterior_indices_to_keep = .!isnothing.(posterior_gq_raw)
    posterior_samples_keep = ChainsCustomIndex(samples, posterior_indices_to_keep)

    Random.seed!(seed)
    posterior_predictive = predict(my_model_predictive, posterior_samples_keep)

    println("Generating posterior quantities...")
    posterior_gq = generated_quantities(my_model_gq, posterior_samples_keep)
    println("Generation of posterior quantities complete.")

    chains = DataFrame(posterior_samples_keep).chain
    iters = DataFrame(posterior_samples_keep).iteration
    posterior_gq_augmented = [
        merge(posterior_gq[i], (
            chain = chains[i],
            iteration = iters[i]
        ))
        for i in eachindex(posterior_gq)
    ]

    ## Prior Sampling ------------------------------
    println("Sampling from prior...")
    Random.seed!(seed)

    prior_samples = sample(
        my_model_gq,
        Prior(),
        MCMCThreads(),
        400,
        1
    )

    prior_samples_df = DataFrame(prior_samples)
    println("Prior sampling complete.")


    ## Prior Predictive and Generated Quantities ------------------------------
    prior_gq_raw = generated_quantities(my_model_gq, prior_samples)
    prior_indices_to_keep = .!isnothing.(prior_gq_raw)
    prior_samples_keep = ChainsCustomIndex(prior_samples, prior_indices_to_keep)

    Random.seed!(seed)
    prior_predictive = predict(my_model_predictive, prior_samples_keep)

    println("Generating prior quantities...")
    prior_gq = generated_quantities(my_model_gq, prior_samples_keep)
    println("Generation of prior quantities complete.")

    chains = DataFrame(prior_samples_keep).chain
    iters = DataFrame(prior_samples_keep).iteration
    prior_gq_augmented = [
        merge(prior_gq[i], (
            chain = chains[i],
            iteration = iters[i]
        ))
        for i in eachindex(prior_gq)
    ]

    ## Return results ------------------------------
    return (
        posterior_predictive = DataFrame(posterior_predictive),
        posterior_generated_quantities = posterior_gq,
        posterior_samples = posterior_samples_df,

        prior_predictive = DataFrame(prior_predictive),
        prior_generated_quantities = prior_gq,
        prior_samples = prior_samples_df
    )

end