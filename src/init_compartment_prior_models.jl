
@model function exp_init_compartment_prior_model(;
    α_prior = (mean = log(0.03), sd = 0.25),
    i0_prior = (mean = log(100.0), sd = 0.5),
    max_age = 50.0
)
    compartments = 8

    # PRIORS-----------------------------
    α_non_centered ~ Normal()
    i0_non_centered ~ Normal()

    # TRANSFORMATIONS------------------------------
    α = exp(α_prior.mean + α_prior.sd * α_non_centered)
    i0 = exp(i0_prior.mean + i0_prior.sd * i0_non_centered)

    bin_edges = collect(range(0.0, max_age, length=compartments+1))
    lower_bounds = bin_edges[1:end-1]
    upper_bounds = bin_edges[2:end]

    probs = exp.(-α .* lower_bounds) .- exp.(-α .* upper_bounds)
    probs_norm = probs ./ sum(probs)

    comp_counts = i0 .* probs_norm

    return (
        compartment₁ = (
            E = comp_counts[1],
            I₁ = comp_counts[2],
            I₂ = comp_counts[3],
            I₃ = comp_counts[4],
            I₄ = comp_counts[5],
            I₅ = comp_counts[6],
            I₆ = comp_counts[7],
            I₇ = comp_counts[8],
            R₁ = 0.0,
            R₂ = 0.0
        ),
        params = (
            α = α,
            i0 = i0
        )
    )

end

@model function custom_init_compartment_prior_model(;
    compartment₁_prior = (
        E = (mean = log(100.0), sd = 0.5),
        I₁ = (mean = log(50.0), sd = 0.5),
        I₂ = (mean = log(30.0), sd = 0.5),
        I₃ = (mean = log(20.0), sd = 0.5),
        I₄ = (mean = log(10.0), sd = 0.5),
        I₅ = (mean = log(5.0), sd = 0.5),
        I₆ = (mean = log(3.0), sd = 0.5),
        I₇ = (mean = log(2.0), sd = 0.5),
        R₁ = (mean = log(1.0), sd = 0.5),
        R₂ = (mean = log(1.0), sd = 0.5)
    )
)
    # PRIORS-----------------------------
    E_non_centered ~ Normal()
    I₁_non_centered ~ Normal()
    I₂_non_centered ~ Normal()
    I₃_non_centered ~ Normal()
    I₄_non_centered ~ Normal()
    I₅_non_centered ~ Normal()
    I₆_non_centered ~ Normal()
    I₇_non_centered ~ Normal()
    R₁_non_centered ~ Normal()
    R₂_non_centered ~ Normal()

    # TRANSFORMATIONS-----------------------------
    E = exp(compartment₁_prior.E.mean + compartment₁_prior.E.sd * E_non_centered)
    I₁ = exp(compartment₁_prior.I₁.mean + compartment₁_prior.I₁.sd * I₁_non_centered)
    I₂ = exp(compartment₁_prior.I₂.mean + compartment₁_prior.I₂.sd * I₂_non_centered)
    I₃ = exp(compartment₁_prior.I₃.mean + compartment₁_prior.I₃.sd * I₃_non_centered)
    I₄ = exp(compartment₁_prior.I₄.mean + compartment₁_prior.I₄.sd * I₄_non_centered)
    I₅ = exp(compartment₁_prior.I₅.mean + compartment₁_prior.I₅.sd * I₅_non_centered)
    I₆ = exp(compartment₁_prior.I₆.mean + compartment₁_prior.I₆.sd * I₆_non_centered)
    I₇ = exp(compartment₁_prior.I₇.mean + compartment₁_prior.I₇.sd * I₇_non_centered)
    R₁ = exp(compartment₁_prior.R₁.mean + compartment₁_prior.R₁.sd * R₁_non_centered)
    R₂ = exp(compartment₁_prior.R₂.mean + compartment₁_prior.R₂.sd * R₂_non_centered)

    return (
        compartment₁ = (
            E = E,
            I₁ = I₁,
            I₂ = I₂,
            I₃ = I₃,
            I₄ = I₄,
            I₅ = I₅,
            I₆ = I₆,
            I₇ = I₇,
            R₁ = R₁,
            R₂ = R₂
        ),
        params = nothing
    )

end