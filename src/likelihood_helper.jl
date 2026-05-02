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
        s,
        γ_prior,
        ν_prior,
        η_prior,
        σ_ww_prior,
        compartment_priors,
        γ_non_centered,
        ν_non_centered,
        η_non_centered,
        σ_ww_non_centered,
        compartment_priors_non_centered,
        Rₜ_module,
)

    try
        # Transform parameters
        γ = exp(γ_prior.mean + γ_prior.sd * γ_non_centered)
        ν = exp(ν_prior.mean + ν_prior.sd * ν_non_centered)
        η = exp(η_prior.mean + η_prior.sd * η_non_centered)
        σ_ww = exp(σ_ww_prior.mean + σ_ww_prior.sd * σ_ww_non_centered)
        E₁ = exp(compartment_priors.E₁_prior.mean + compartment_priors.E₁_prior.sd * compartment_priors_non_centered[1])
        I₁ = exp(compartment_priors.I₁_prior.mean + compartment_priors.I₁_prior.sd * compartment_priors_non_centered[2])
        I₂ = exp(compartment_priors.I₂_prior.mean + compartment_priors.I₂_prior.sd * compartment_priors_non_centered[3])
        I₃ = exp(compartment_priors.I₃_prior.mean + compartment_priors.I₃_prior.sd * compartment_priors_non_centered[4])
        I₄ = exp(compartment_priors.I₄_prior.mean + compartment_priors.I₄_prior.sd * compartment_priors_non_centered[5])
        I₅ = exp(compartment_priors.I₅_prior.mean + compartment_priors.I₅_prior.sd * compartment_priors_non_centered[6])
        I₆ = exp(compartment_priors.I₆_prior.mean + compartment_priors.I₆_prior.sd * compartment_priors_non_centered[7])
        I₇ = exp(compartment_priors.I₇_prior.mean + compartment_priors.I₇_prior.sd * compartment_priors_non_centered[8])
        R₁ = exp(compartment_priors.R₁_prior.mean + compartment_priors.R₁_prior.sd * compartment_priors_non_centered[9])
        R₂ = exp(compartment_priors.R₂_prior.mean + compartment_priors.R₂_prior.sd * compartment_priors_non_centered[10])

        # Simulate the model
        ν₁ = 7 * ν
        ν₂ = 7 * ν
        ν₃ = 7 * ν
        ν₄ = 7 * ν
        ν₅ = 7 * ν
        ν₆ = 7 * ν
        ν₇ = 7 * ν 

        αs = Rₜ_module.Rₜ .* ν
        u0 = [E₁, I₁, I₂, I₃, I₄, I₅, I₆, I₇, R₁, R₂, 0.0]  
        p = (αs, γ, ν₁, ν₂, ν₃, ν₄, ν₅, ν₆, ν₇, η, η, Rₜ_module.timebreaks)
        tspan = (minimum(obstime_wastewater), maximum(obstime_wastewater))
        prob = ODEProblem(multi_i_ode!, u0, tspan, p, saveat = obstime_wastewater)

        sol = solve(prob, Tsit5(); verbose=false)
        # If the ODE solver fails, reject the sample by adding -Inf to the likelihood
        if sol.retcode != ReturnCode.Success
            throw(ArgumentError("ODE solver failed!!!"))
        end
        
        sol_array = Array(sol)
        sol_array .= clamp.(sol_array, 1, 1e10)
        sol_nt = (
            E₁ = sol_array[1, :],
            I₁ = sol_array[2, :],
            I₂ = sol_array[3, :],
            I₃ = sol_array[4, :],
            I₄ = sol_array[5, :],
            I₅ = sol_array[6, :],
            I₆ = sol_array[7, :],
            I₇ = sol_array[8, :],
            R₁ = sol_array[9, :],
            R₂ = sol_array[10, :],
            W = sol_array[11, :]
        )

        I_means = sol_array[2, :] + sol_array[3, :] + sol_array[4, :] + sol_array[5, :] + sol_array[6, :] + sol_array[7, :] + sol_array[8, :]

        # Wastewater means with attached shedding to compartments
        w_mean = 
            s[1] .* sol_array[2, :] .+ # I1
            s[2] .* sol_array[3, :] .+ # I2
            s[3] .* sol_array[4, :] .+ # I3
            s[4] .* sol_array[5, :] .+ # I4
            s[5] .* sol_array[6, :] .+ # I5
            s[6] .* sol_array[7, :] .+ # I6
            s[7] .* sol_array[8, :] .+ # I7
            s[8] .* sol_array[9, :] .+ # R1
            s[9] .* sol_array[10, :]    # R2
        log_W_means = log.(w_mean) 


        return (
            success = true,
            log_W_means = log_W_means,
            I_means = I_means,
            αₜ = αs,
            Rₜ = Rₜ_module.Rₜ,
            Rₜ_params = Rₜ_module.params,
            ode_parameters = (γ = γ, ν = ν, η = η),
            σ_ww = σ_ww,
            compartment₁ = (E₁ = E₁, I₁ = I₁, I₂ = I₂, I₃ = I₃, I₄ = I₄, I₅ = I₅, I₆ = I₆, I₇ = I₇, R₁ = R₁, R₂ = R₂),
            ode_solution = (t = sol.t, states = sol_nt)
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