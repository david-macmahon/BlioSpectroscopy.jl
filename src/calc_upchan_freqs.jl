export calc_upchan_freqs

"""
    calc_upchan_freqs(hdr, nfpc=2^17) -> freqs::Range

Returns a Range of frequencies that correspond to up-channelizing the GUPPI RAW
data described by `hdr` by a factor of `nfpc`.  The `hdr` argument can be
anything that can be indexed by `:obsnchan`, `:nants` (defaults to 1 if not
present), `:obsfreq`, `:chan_bw`.  Commonly used types for `hdr` would be
`GuppiRaw.Header` or `DataFrameRow`, but `Dict{Symbol,Any}` and `NamedTuple` can
be used as well.
"""
function calc_upchan_freqs(hdr, nfpc=2^17)
    nants = haskey(hdr, :nants) ? hdr[:nants] : 1
    ncoarse = hdr[:obsnchan] ÷ nants
    fcoarse0 = hdr[:obsfreq] - (ncoarse-1)*hdr[:chan_bw]/2
    foff = hdr[:chan_bw]/nfpc
    fch1 = fcoarse0 - (nfpc÷2) * foff
    nfine = nfpc * ncoarse
    range(fch1, step=foff, length=nfine)
end

