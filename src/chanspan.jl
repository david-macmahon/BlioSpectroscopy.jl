"""
    chanspan(chan, nchan) -> chan_range

Return a `Range` of `nchan` channel numbers centered on `chan`.
"""
function chanspan(chan, nchan)
    (0:nchan-1) .- nchan÷2 .+ chan
end

