using Blio: GuppiRaw
using Statistics: std

"""
    upchan_extract_guppiraw(hdr1, blks; cchan, fchan, nfchan=32, sideband=:native) -> (grh, data)

Uses `upchan_extract` to up-channelize the 3- or 4-dimensional data blocks in
`blks` and extract fine channels based on `cchan`, `fchan, and `nfchan`.  The
results from each block are treated as single time sample.  Time samples from
multiple blocks are concatenated (and scaled and quantized) to create an
`Array{Complex{Int8}}`` whose dimensions are permuted to match the GUPPI RAW
ordering of dimensions: `(npol, ntime, nchan, nant)`, where:

- `npol` is the number of polarizations,
- `ntime` is the number of output time samples (i.e. number of input blocks),
- `nfine` is the number of fine channels (i.e. number of time samples per input
  block)
- `nant` is the number of antennas.

If `sideband` is `:upper` or `:lower` and the original data is opposite, the
data array will be flipped around the frequency axis and the relevant fields in
the returned GuppiRaw.Header object (i.e. `:chan_bw` and `:obsbw`) will be
updated.

This function returns `(grh::GuppiRaw.Header, newblockk::Array{Complex{Int8}})`,
where `grh` is a copy of `hdr1` that has been modified to reflect the data in
`newblock`.
"""
function upchan_extract_guppiraw(hdr1, blks; cchan, fchan, nfchan=32, sideband=:native)
    # upchan_extract returns an Array dimensioned: `(time, chan, ant, pol)`
    vv_tcap = upchan_extract(blks; cchan, fchan, nfchan)

    # Requantize vv from Complex{Float32} to Complex{Int8}
    vvi8_tcap = requantize_to_i8(vv_tcap)
    # Permute dims to `(pol, time, chan, ant)`
    vvi8_ptca = permutedims(vvi8_tcap, (4,1,2,3))

    # Gather relevant metadata
    nfpc = size(first(blks), 2)
    grh = GuppiRaw.Header(hdr1)
    cfreqs = collect(Iterators.partition(calc_upchan_freqs(grh, nfpc), nfpc))
    ffreqs = cfreqs[cchan][chanspan(fchan, nfchan)]
    _, _, nc, na = size(vvi8_ptca)

    # Update grh
    grh[:blocsize] = sizeof(vvi8_ptca)
    grh[:nchan] = nc # non-standard, but present in COSMIC RAW files
    grh[:obsnchan] = nc * na
    grh[:chan_bw] /= nfpc
    grh[:tbin] = 1e-6/abs(grh[:chan_bw])
    grh[:obsbw] = nc * grh[:chan_bw]
    grh[:obsfreq] = sum(extrema(ffreqs)) / 2
    #NO!grh[:piperblk] *= length(blks)
    # Set pktstart and pktidx to -1 so that Rawx.jl will derive from
    # STT_IMJD/STT_SMJD instead
    grh[:pktstart] = -1
    grh[:pktidx] = -1

    # If sideband is opposire from requested (i.e. spectrally flipped), unflip
    if (sideband == :upper && grh[:chan_bw] < 0) || (sideband == :lower && grh[:chan_bw] > 0)
        grh[:chan_bw] *= -1
        grh[:obsbw] *= -1
        reverse!(vvi8_ptca, dims=3)
    end

    return grh, vvi8_ptca
end

"""
    requantize_to_i8(vv::Array{ComplexF32})::Array{Complex{Int8}}

Scale `vv` to be within [-127, 127], round, and convert to
`Array{Complex{Int8}}`.
"""
function requantize_to_i8(vv::Array{ComplexF32})
    vvf32 = reinterpret(Float32, vv)
    sigma = std(vvf32)
    nsigma = maximum(abs.(extrema(vvf32))) / sigma
    scale = 120 ÷ nsigma
    convert(Array{Complex{Int8}}, round.(scale .* vv ./ sigma))
end

"""
    write_raw_file(filename, grh::GuppiRaw.Header, data)

Write a single block GUPPI RAW file with `grh` as the header and `data` as the
data block.  No validation of header fields in `grh` is performed, but a warning
will be displayed if `grh[:blocsize] != sizeof(data)`.
"""
function write_raw_file(filename, grh, data)
    if grh[:blocsize] != sizeof(data)
        @warn "blocsize mismatch ($(grh[:blocsize]) != $(sizeof(data)))"
    end

    open(filename, "w") do io
        write(io, grh)
        write(io, data)
    end
end
