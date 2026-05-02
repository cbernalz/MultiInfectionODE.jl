
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
