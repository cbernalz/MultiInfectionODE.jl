"""
    likelihood_helper(...)
Helper function to compute the likelihood components for the MvFInfectionAge model. This function simulates the PDE given the parameters and returns the necessary components for the likelihood calculation in the Turing model.

# Arguments
    - `obstimes_wastewater`: Vector of observation times for the wastewater data.
    - `s`: Vector of shedding coefficients for each compartment.
    - `γ_prior`: Prior parameters for the γ parameter (mean and sd on log scale).
    - `ν_prior`: Prior parameters for the ν parameter (mean and sd on log scale).
    - `η_prior`: Prior parameters for the η parameter (mean and sd on log scale).
    - `σ_ww_prior`: Prior parameters for the σ_ww parameter (mean and sd on log scale).
    - `compartment_priors`: A named tuple containing the prior parameters for each compartment (E₁, I₁, I₂, I₃, I₄, I₅, I₆, I₇, R₁, R₂), each specified as a tuple of (mean, sd) on the log scale.
    - `γ_non_centered`: Sampled value for non-centered γ.
    - `ν_non_centered`: Sampled value for non-centered ν.
    - `η_non_centered`: Sampled value for non-centered η.
    - `σ_ww_non_centered`: Sampled value for non-centered σ_ww.
    - `compartment_priors_non_centered`: Sampled values for non-centered compartment initial conditions (E₁, I₁, I₂, I₃, I₄, I₅, I₆, I₇, R₁, R₂).
    - `Rₜ_module`: The submodel for Rₜ, which provides the Rₜ values and parameters.

# Returns
"""
function likelihood_helper(
        obstime_wastewater,
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

    try
        # Transform parameters
        γ = exp(γ_prior.mean + γ_prior.sd * γ_non_centered)
        ν = exp(ν_prior.mean + ν_prior.sd * ν_non_centered)
        η = exp(η_prior.mean + η_prior.sd * η_non_centered)
        σ_ww = exp(σ_ww_prior.mean + σ_ww_prior.sd * σ_ww_non_centered)

        prob = setup_multi_i_ode_problem(
            obstime_wastewater = obstime_wastewater,
            Rₜ_module = Rₜ_module,
            init_compartment_module = init_compartment_module,
            γ = γ,
            ν = ν,
            η = η
        )

        sol = solve(prob, Tsit5(); verbose = false)

        if sol.retcode != ReturnCode.Success
            @warn "ODE solver failed." retcode = sol.retcode
            return (success = false,)
        end

        sol_array = Array(sol)
        sol_array .= clamp.(sol_array, 1, 1e10)

        unpacked = unpack_multi_i_solution(sol_array, init_compartment_module)
        w_mean = wastewater_mean(sol_array, init_compartment_module)
        log_W_means = log.(clamp.(w_mean, 1e-12, Inf))

        return (
            success = true,
            log_W_means = log_W_means,
            I_means = unpacked.I_means,
            αₜ = prob.p.αs,
            Rₜ = Rₜ_module.Rₜ,
            Rₜ_params = Rₜ_module.params,
            ode_parameters = (γ = γ, ν = ν, η = η),
            σ_ww = σ_ww,
            compartment₁ = init_compartment_module.compartment₁,
            compartment₁_params = init_compartment_module.params,
            ode_solution = (t = sol.t, states = unpacked.states)
        )

    catch e
        println("likelihood_helper failed:")
        showerror(stdout, e)
        println()
        Base.show_backtrace(stdout, catch_backtrace())
        println()
        return (success = false,)

    end




end