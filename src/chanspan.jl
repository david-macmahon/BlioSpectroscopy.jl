"""
    chanspan(chan, nchan) -> chan_range

Return a `Range` of `nchan` channel numbers centered on `chan`.
"""
function chanspan(chan, nchan)
    (0:nchan-1) .- nchan÷2 .+ chan
end

"""
    centerspan(firstchan, lastchan) -> center, span
    centerspan(chanrange::AbstractRange) -> center, span

Returns the center fine channel and number of fine channels spanned from
`firstchan` to `lastchan`.
"""
function centerspan(firstchan, lastchan)
    @assert firstchan <= lastchan "invalid channel span"
    span = lastchan - firstchan + 1
    center = firstchan + fld(span, 2)
    center, span
end

function centerspan(chanrange::AbstractRange)
    centerspan(first(chanrange), last(chanrange))
end

