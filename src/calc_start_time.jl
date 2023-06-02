using Blio.GuppiRaw: getntime

export calc_start_time

"""
    calc_start_time(hdr) -> unix_seconds

Returns the time of the first sample of the GUPPI RAW data block corresponding
to `hdr` as the number of seconds since the Unix epoch.  The `hdr` argument can
be anything that can be indexed by `:tbin`, `:piperblk`, `:synctime`, `:pktidx`.
Commonly used types for `hdr` would be GuppiRaw.Header or DataFrameRow, but
`Dict{Symbol,Any}` and `NamedTuple` can be used as well.
"""
function calc_start_time(hdr)
    if all(haskey.(Ref(hdr), (:tbin, :piperblk, :synctime, :pktidx)))
        secperblk=hdr[:tbin] * getntime(hdr)
        secperpktidx = secperblk / hdr[:piperblk]
        hdr[:synctime] + hdr[:pktidx] * secperpktidx
    elseif all(haskey.(Ref(hdr), (:stt_imjd, :stt_smjd)))
        mjd1970 = 40587
        (hdr[:stt_imjd]-mjd1970) * (24*60*60) + hdr[:stt_smjd]
    else
        error("unable to calculate start time")
    end
end

