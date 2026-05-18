
"""
    ChainsCustomIndex(c::Chains, indices_to_keep::BitMatrix)

Reduce Chains object to only wanted indices. 

Function created by Damon Bayer. 
"""
function ChainsCustomIndex(c::Chains, indices_to_keep::BitMatrix)
    min_length = minimum(mapslices(sum, indices_to_keep, dims = 1))
  v = c.value
  new_v = copy(v.data)
  new_v_filtered = cat([new_v[indices_to_keep[:, i], :, i][1:min_length, :] for i in 1:size(v, 3)]..., dims = 3)
  aa = AxisArray(new_v_filtered; iter = v.axes[1].val[1:min_length], var = v.axes[2].val, chain = v.axes[3].val)

  Chains(aa, c.logevidence, c.name_map, c.info)
end



function wastewater_mean(sol_array, init_compartment_module)

    n_I = init_compartment_module.ode_config.n_I
    n_R = init_compartment_module.ode_config.n_R
    include_R_shedding = init_compartment_module.ode_config.include_R_shedding
    s = init_compartment_module.ode_config.shedding_weights

    I_rows = 2:(n_I + 1)

    if include_R_shedding
        expected_length = n_I + n_R

        if length(s) != expected_length
            throw(ArgumentError(
                "shedding_weights must have length n_I + n_R when include_R_shedding = true. " *
                "Got $(length(s)), expected $(expected_length)."
            ))
        end

        shedding_rows = 2:(n_I + n_R + 1)

    else
        expected_length = n_I

        if length(s) != expected_length
            throw(ArgumentError(
                "shedding_weights must have length n_I when include_R_shedding = false. " *
                "Got $(length(s)), expected $(expected_length)."
            ))
        end

        shedding_rows = I_rows
    end

    shedding_states = sol_array[shedding_rows, :]

    return vec(sum(s .* shedding_states, dims = 1))
end