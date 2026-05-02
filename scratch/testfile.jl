

using Revise
using MultiInfectionODE
using CSV
using DataFrames
using Plots



_data_path = normpath(joinpath(@__DIR__, "MvF-Simulation-Exploration", "data", "sim-data-2.csv"))
sim_data = CSV.read(_data_path, DataFrame;)


T = 125.0   # total time
drop_initial = 25




#u0 = [1000.0, 5000.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
#αs = sim_data.rt[1:Int(T)] .* ν
#param_change_times = sim_data.time[1:Int(T)]
#p = (αs, γ, ν₁, ν₂, ν₃, ν₄, ν₅, ν₆, ν₇, η, η, param_change_times)
#tspan = (0.0, T)
#prob = ODEProblem(multi_i_ode!, u0, tspan, p, saveat = sim_data.time[1:Int(T)])
#sol = solve(prob, Tsit5())

#I_ode = sol[2, :] + sol[3, :] + sol[4, :] + sol[5, :] + sol[6, :] + sol[7, :] + sol[8, :]
#plot(sim_data.time[1:Int(T)], sim_data.I_total[1:Int(T)], label="Simulated Infections", xlabel="Time", ylabel="Infections")
#plot!(sim_data.time[1:Int(T)], I_ode, label="ODE Infections", xlabel="Time", ylabel="Infections")


γ_prior = (mean = log(1/7), sd = 0.25)
ν_prior = (mean = log(1/7), sd = 0.25)
η_prior = (mean = log(2/18), sd = 0.25)
σ_ww_prior = (mean = log(0.6), sd = 0.25)
compartment_priors = (
    E₁_prior = (mean = log(2250.0), sd = 0.25),
    I₁_prior = (mean = log(350.0), sd = 0.25),
    I₂_prior = (mean = log(350.0), sd = 0.25),
    I₃_prior = (mean = log(350.0), sd = 0.25),
    I₄_prior = (mean = log(350.0), sd = 0.25),
    I₅_prior = (mean = log(350.0), sd = 0.25),
    I₆_prior = (mean = log(350.0), sd = 0.25),
    I₇_prior = (mean = log(350.0), sd = 0.25),
    R₁_prior = (mean = log(3500.0), sd = 0.25),
    R₂_prior = (mean = log(3500.0), sd = 0.25)
)


log_R₁_prior = (mean = log(1.0), sd = 0.65)
σ_R₁_prior = (mean = log(0.4), sd = 0.25)
σ_Rₜ_prior = (mean = log(0.45), sd = 0.25)
log_R₁_prime_prior = (mean = 0.0, sd = 0.25)


data_wastewater = sim_data.obs_log_mean_copiesten[drop_initial+1:Int(T + drop_initial)]
obstime_wastewater = convert(Vector{Float64}, sim_data.time[drop_initial+1:Int(T + drop_initial)]) .- convert(Float64, drop_initial)

dumb_weights = [0.05, 0.4, 0.8, 0.4, 0.2, 0.1, 0.05, 0.025, 0.0125]
norm_weights = dumb_weights / sum(dumb_weights)
log_weights = log10.(norm_weights .* 1e7)
s = 10 .^ log_weights

weeks = ceil(Int, maximum(obstime_wastewater) / 7.0)
timebreaks = collect(1:7:(weeks*7))
timebreaks = convert(Vector{Float64}, timebreaks)
rt_prior_model_internal = rt_ibm_prior_model_loop(
    timebreaks;
    log_R₁_prior,
    σ_R₁_prior,
    σ_Rₜ_prior,
    log_R₁_prime_prior
)
#rt_prior_model_internal = rt_rw_prior_model(
#    timebreaks;
#    log_R₁_prior,
#    σ_Rₜ_prior
#)

samples = fit(
    data_wastewater,
    obstime_wastewater,
    s,
    rt_prior_model_internal,
    γ_prior,
    ν_prior,
    η_prior,
    σ_ww_prior,
    compartment_priors,
    false,
    50, 1,
    20, 2024,
    nothing
)



results = generate_pq_pp(
    samples,
    data_wastewater,
    obstime_wastewater,
    s,
    rt_prior_model_internal,
    γ_prior,
    ν_prior,
    η_prior,
    σ_ww_prior,
    compartment_priors;
)


_result_path = normpath(joinpath(@__DIR__, "MvF-Simulation-Exploration", "results", "MultiInfectionODE-test"))
#_result_path = normpath(joinpath(@__DIR__, "..", "..", "results", "MvFInfectionAge", "h=" * string(h)))

if !isdir(_result_path)
    mkpath(_result_path)
end
CSV.write(
joinpath(_result_path, "pp.csv"),
results[1]
)
CSV.write(
joinpath(_result_path, "gq.csv"),
results[2]
)
CSV.write(
joinpath(_result_path, "samples.csv"),
results[3]
)
