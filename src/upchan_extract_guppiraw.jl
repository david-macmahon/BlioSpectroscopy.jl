using Glob: glob
using Blio: GuppiRaw
using DataFrames: DataFrame
using Statistics: std

"""
    upchan_extract_guppiraw(globpattern, DIR; cchan, fchan, nfchan=32, sideband=:native, pbkwargs...) -> (grh, data)

For all files in `DIR` matching `globpattern`, use CUDA to up-channelize coarse
channel `cchan` of 3- or 4-dimensional GUPPI RAW data blocks along the time
axis.  For each block, `fftshift` the resulting fine channels, then select
`nfchan` channels around fine channel `fchan`.  The results from each
block are treated as single time sample.  Time samples from multiple blocks are
concatenated (and scaled and quantized) to create an `Array{Complex{Int8}}``
whose dimensions are permuted to match the GUPPI RAW ordering of dimensions:
`(npol, ntime, nchan, nant)`, where:

- `npol` is the number of polarizations,
- `ntime` is the number of output time samples (i.e. number of input blocks),
- `nfine` is the number of fine channels (i.e. number of time samples per input
  block)
- `nant` is the number of antennas.

If `sideband` is `:upper` or `:lower` and the original data is opposite, the
data array will be flipped around the frequency axis and the relevant fields in
the returned GuppiRaw.Header object (i.e. `:chan_bw` and `:obsbw`) will be
updated.

This function returns `nothing` if no files in `DIR` match or a
`(GuppiRaw::Header, Array{Complex{Int8}})`` tuple if one or more files in `DIR`
match.

`pbkwargs` are passed through to ProgressBar.  This can useful in notebooks
where the progress bar may be unwanted (e.g. by passing
`output_stream=devnull`).
"""
function upchan_extract_guppiraw(globpattern, DIR=getrawroot(); cchan, fchan, nfchan=32, sideband=:native, pbkwargs...)
    # Get the list of RAW file names
    rawnames = sort(glob(globpattern, DIR))
    if isempty(rawnames)
        @info "no files matching $globpattern in $DIR"
        return nothing
    end

    hdrs, blks = GuppiRaw.load(rawnames, DataFrame)

    grh = GuppiRaw.Header(hdrs[1,:])
    # upchan_extract returns an Array dimensioned: `(time, chan, ant, pol)`
    vv_tcap = upchan_extract(blks; cchan, fchan, nfchan, pbkwargs...)

    # Requantize vv from Complex{Float32} to Complex{Int8}
    vvi8_tcap = requantize_to_i8(vv_tcap)
    # Permute dims to `(pol, time, chan, ant)`
    vvi8_ptca = permutedims(vvi8_tcap, (4,1,2,3))

    # Gather relevant metadata
    nfpc = size(blks[1], 2)
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

    # Finalize blks to release mmap'd regions
    foreach(finalize, blks)

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
