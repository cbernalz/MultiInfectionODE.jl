"""
    generate_pq_pp(...)

# Arguments

# Returns
- Posterior Quanteties and Posterior Predictive Distribution.
"""
function generate_pq_pp(
    samples,
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
    seed::Int64=2024,
    forecast::Bool=false, forecast_days::Int64=14
)

    # Obstime processing-----------------------------
    obstime_wastewater = convert(Vector{Float64}, obstime_wastewater)

    if forecast
        last_value = obstime_wastewater[end]
        obstime_wastewater = vcat(obstime_wastewater,(last_value+1):(last_value+forecast_days))
        missing_data_wastewater = repeat([missing], length(obstime_wastewater))
        data_wastewater = vcat(data_wastewater, repeat([data_wastewater[end]], forecast_days))
    else
        missing_data_wastewater = repeat([missing], length(data_wastewater))
    end
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
    my_model_forecast_missing = multi_infection_ode_model(
        data_wastewater = missing_data_wastewater,
        obstime_wastewater = obstime_wastewater,
        s = s,
        Rₜ_prior_model = Rₜ_prior_model,
        γ_prior = γ_prior,
        ν_prior = ν_prior,
        η_prior = η_prior,
        σ_ww_prior = σ_ww_prior,
        compartment_priors = compartment_priors
    )

    samples_df = DataFrame(samples)

    indices_to_keep = .!isnothing.(generated_quantities(my_model, samples))
    samples_randn = ChainsCustomIndex(samples, indices_to_keep)

    Random.seed!(seed)
    predictive_randn = predict(my_model_forecast_missing, samples_randn)
    Random.seed!(seed)
    println("Generating quantities...")
    gq_randn = generated_quantities(my_model, samples_randn)
    results = [DataFrame(predictive_randn), gq_randn, samples_df]

    return(results)

end


