
@model function exp_init_compartment_prior_model(;
    ode_config = (
        n_I = 7,
        n_R = 2,
        include_R_shedding = true,
        shedding_weights = nothing
    ),
    α_prior = (mean = log(0.03), sd = 0.25),
    i0_prior = (mean = log(100.0), sd = 0.5),
    max_age = 50.0
)

    α_non_centered ~ Normal()
    i0_non_centered ~ Normal()

    α = exp(α_prior.mean + α_prior.sd * α_non_centered)
    i0 = exp(i0_prior.mean + i0_prior.sd * i0_non_centered)

    n_I = ode_config.n_I
    n_R = ode_config.n_R

    n_compartments = 1 + n_I + n_R

    bin_edges = collect(range(0.0, max_age, length = n_compartments + 1))
    lower_bounds = bin_edges[1:end-1]
    upper_bounds = bin_edges[2:end]

    probs = exp.(-α .* lower_bounds) .- exp.(-α .* upper_bounds)
    probs_norm = probs ./ sum(probs)

    comp_counts = i0 .* probs_norm

    E = comp_counts[1]
    I = comp_counts[2:(n_I + 1)]

    if n_R == 0
        R_stage = Float64[]
    else
        R_stage = comp_counts[(n_I + 2):(n_I + n_R + 1)]
    end

    if ode_config.shedding_weights === nothing
        if ode_config.include_R_shedding
            shedding_weights_final = ones(n_I + n_R)
        else
            shedding_weights_final = ones(n_I)
        end
    else
        shedding_weights_final = ode_config.shedding_weights
    end

    return (
        compartment₁ = (
            E = E,
            I = I,
            R_stage = R_stage,
            R = 0.0
        ),
        ode_config = (
            n_I = n_I,
            n_R = n_R,
            include_R_shedding = ode_config.include_R_shedding,
            shedding_weights = shedding_weights_final
        ),
        params = (
            α = α,
            i0 = i0
        )
    )
end

@model function custom_init_compartment_prior_model(;
    ode_config = (
        n_I = 7,
        n_R = 2,
        include_R_shedding = true,
        shedding_weights = nothing
    ),
    E_prior = (mean = log(700.0), sd = 0.25),
    I_priors = (mean = log(100.0), sd = 0.25)
)

    n_I = ode_config.n_I
    n_R = ode_config.n_R

    E_non_centered ~ Normal()
    I_non_centered = Vector{Real}(undef, n_I)
    for j in 1:n_I
        I_non_centered[j] ~ Normal()
    end

    E = exp(E_prior.mean + E_prior.sd * E_non_centered)

    I = [
        exp(I_priors.mean + I_priors.sd * I_non_centered[j])
        for j in 1:n_I
    ]

    if ode_config.shedding_weights === nothing
        if ode_config.include_R_shedding
            shedding_weights_final = ones(n_I + n_R)
        else
            shedding_weights_final = ones(n_I)
        end
    else
        shedding_weights_final = ode_config.shedding_weights
    end

    return (
        compartment₁ = (
            E = E,
            I = I,
            R_stage = zeros(n_R),
            R = 0.0
        ),
        ode_config = (
            n_I = n_I,
            n_R = n_R,
            include_R_shedding = ode_config.include_R_shedding,
            shedding_weights = shedding_weights_final
        ),
        params = (params = "NA")
    )
end