

@model function multi_infection_ode_model(;
    data_wastewater,
    obstime_wastewater,
    s,
    Rₜ_prior_model,
    init_compartment_prior_model,
    γ_prior = (mean = log(1/7), sd = 0.25),
    ν_prior = (mean = log(1/7), sd = 0.25),
    η_prior = (mean = log(2/18), sd = 0.25),
    σ_ww_prior = (mean = log(0.1), sd = 0.25)
)

    # PRIORS-----------------------------
    γ_non_centered ~ Normal()
    ν_non_centered ~ Normal()
    η_non_centered ~ Normal()
    Rₜ_module ~ Turing.to_submodel(Rₜ_prior_model)
    init_compartment_module ~ Turing.to_submodel(init_compartment_prior_model)

    σ_ww_non_centered ~ Normal()

    # TRANSFORMATIONS-----------------------------
    trans = likelihood_helper(
        obstime_wastewater,
        s,
        γ_prior,
        ν_prior,
        η_prior,
        σ_ww_prior,
        γ_non_centered,
        ν_non_centered,
        η_non_centered,
        σ_ww_non_centered,
        Rₜ_module,
        init_compartment_module
    )

    # Reject if the helper function failed and skip sample
    if !trans.success
        println("Likelihood Helper Failed for current parameters. Skipping sample...")
        Turing.@addlogprob! -Inf
        return
    end

    # Likelihood calculations------------
    for i in 1:length(obstime_wastewater)
        data_wastewater[i] ~ Normal(trans.log_W_means[i], trans.σ_ww)   
    end

    return (
        log_W_means = trans.log_W_means, I_means = trans.I_means,
        αₜ  = trans.αₜ,
        Rₜ = trans.Rₜ, Rₜ_params = trans.Rₜ_params,
        σ_ww = trans.σ_ww,
        ode_parameters = trans.ode_parameters,
        compartment₁ = trans.compartment₁, compartment₁_params = trans.compartment₁_params,
        ode_solution = trans.ode_solution
    )

end