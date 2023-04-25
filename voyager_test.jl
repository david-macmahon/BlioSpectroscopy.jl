using Blio
using Glob
using DataFrames
using Plots
using HDF5
using StatsBase
using ProgressBars
#using FFTW
using LinearAlgebra # for mul!
using CUDA
using CUDA.CUFFT
CUDA.allowscalar(false)

if isdir("/mnt/buf1/voyager/GUPPI")
    @info "Running on COSMIC/VLA"
    # For COSMIC/VLA 60045 data, the buf0 data has Voyager on the edge of coarse
    # channel 16 (and 17 thanks to aliasing).  The buf1 data has voyager near
    # the center of coarse channel 13 (peak is at fine channel 61586, but center
    # of peaks is 61585) on cosmic-gpu-2.
    DIR = "/mnt/buf1/voyager/GUPPI"
    CUDA.device!(5)
elseif isdir("/datax/dibas")
    @info "Running on BL/GBT"
    # For GBT data BLP33, coarse channel 7:
    # - Scan 0112: fine channel 330857/2^20, 165430/2^19
    # - Scan 0116: fine channel 330996/2^20, 165508/2^19
    # The fine channel index for 2^20 fine channels are for the 0000.h5 product
    # which have different time resolution and different total time) than the
    # 2^19 fine channels of the "upchan_extract" with nfpc=2^19 and no time
    # interation.  This leads to slight offsets between the two.

    # Use glob pattern "BLP??/..." to work with any BLPxx
    DIR = "/datax/dibas/AGBT22B_999_41/GUPPI"
else
    @warn "running on unknown system"
end

"""
    loadrawobs(globpattern, dir=".") -> hdrs::DataFrame, blks::Vector{Array}

Loads all GUPPI RAW files in `dir` that match `globpattern`.  Returns a
DataFrame with one row per header and a Vector containing mmap'd Arrays for eash
data block.
"""
function loadrawobs(globpattern, dir=".")
    files = sort(glob(globpattern, dir))
    hdrblks = GuppiRaw.load.(files, DataFrame)
    hdrs = mapreduce(first, (a,b)->vcat(a, b; cols=:union), hdrblks)
    blks = mapreduce(last, vcat, hdrblks)
    hdrs, blks
end

"""
    calcfreqs(hdr, nfpc=2^17) -> freqs::Range

Returns a Range of frequencies that corrrespond to upchannelizing the GUPPI RAW
data described by `hdr` by a factor of `nfpc`.  The `hdr` argument can be
anything that can be indexed by `:obsnchan`, `:nants` (defaults to 1 if not
present), `:obsfreq`, `:chan_bw`.  Commonly used types for `hdr` would be
GuppiRaw.Header or DataFrameRow, but `Dict{Symbol,Any}` and `NamedTuple` can be
used as well.
"""
function calcfreqs(hdr, nfpc=2^17)
    nants = haskey(hdr, :nants) ? hdr[:nants] : 1
    ncoarse = hdr[:obsnchan] ÷ nants
    fcoarse0 = hdr[:obsfreq] - (ncoarse-1)*hdr[:chan_bw]/2
    foff = hdr[:chan_bw]/nfpc
    fch1 = fcoarse0 - (nfpc÷2) * foff
    nfine = nfpc * ncoarse
    range(fch1, step=foff, length=nfine)
end

"""
    calctime(hdr) -> unix_seconds

Returns the time of the first sample of the GUPPI RAW data block corresponding
to `hdr` as the number of seconds since the Unix epoch.  The `hdr` argument can
be anything that can be indexed by `:tbin`, `:piperblk`, `:synctime`, `:pktidx`.
Commonly used types for `hdr` would be GuppiRaw.Header or DataFrameRow, but
`Dict{Symbol,Any}` and `NamedTuple` can be used as well.
"""
function calctime(hdr)
    secperblk=hdr[:tbin]*GuppiRaw.getntime(hdr)
    secperpktidx = secperblk / hdr[:piperblk]
    hdr[:synctime] + hdr[:pktidx] * secperpktidx
end

"""
    spectroscopy(blks; nint=32)

Uses CUDA to upchannelize GUPPI RAW data blocks stored as 4 dimnesional Arrays
in iterable `blks` along the time axis, detects (i.e. converts to power), and
integrates `nint` blocks into each output time sample and returns a 5
dimensional Array.

The input Arrays are dimensioned as: `(npol, ntime, ncoarse, nant)`.

The output Array is dimensioned as: `(nfine, ncoarse, nant, npol, nint)`

where `npol` is the number of polarizations, `ntime` is the number of time
samples per block, `ncoarse` is the number of (coarse) frequency channels per
antenna per block, `nant` is the number of antennas, `nfine` is the number of
(fine) frequency channels in the output, and `nint` is the number of output
integrations.  Note that `nfine == ntime` and `nint` is the time axis of the
output data.
"""
function spectroscopy(blks; nint=32)
    np = size(blks[1], 1)
    nt = size(blks[1], 2)
    nc = size(blks[1], 3)
    na = ndims(blks[1]) < 4 ? 1 : size(blks[1], 4)

    ni = length(blks) ÷ nint
    gpuint = CuArray{Complex{Int8}}(undef, np, nt, nc, na)
    fftin = CuArray{ComplexF32}(undef, nt, nc, na, np)
    plan = plan_fft(fftin, 1)
    fftout = similar(fftin)
    pblk = similar(fftout, Float32, nt, nc, na, np); pblk .= 0
    # Output arrays
    pblkcpu = Array{Float32}(undef, size(pblk))
    pwrints = Array{Float32}(undef, nt, nc, na, np, ni)

    # For each block
    i = 1
    for (b, blk) in ProgressBar(enumerate(blks))
        # Copy block to gpuint
        copyto!(gpuint, blk)

        # Copy gpuint to fftin
        for (f, g) in zip(eachslice(fftin; dims=4), eachslice(gpuint; dims=1))
            copyto!(f, g)
        end

        # FFT fftin to fftout
        mul!(fftout, plan, fftin)

        # Accumulate power into pblk
        pblk .+= abs2.(fftout) 

        if b % nint == 0
            copyto!(pblkcpu, pblk)
            fftshift!(@view(pwrints[:,:,:,:,i]), pblkcpu, 1)
            pblk .= 0
            i += 1
        end
    end

    # Wait for final batch to finish (still needed?)
    CUDA.synchronize()

    # Return pwrints as an Array{Float64}
    convert(Array{Float64}, pwrints)
end

"""
    getkurtosis(pwr::AbstractArray{<:Real,5})

For 5-dimensional array `pwr`, compute the kurtosis along the 5th dimension.
"""
function getkurtosis(pwr::AbstractArray{<:Real,5})
    nf, nc, na, np, nt = size(pwr)
    k = map(kurtosis, eachrow(reshape(pwr, :, nt)))
    reshape(k, nf, nc, na, np)
end

"""
    chanspan(fchan, nfchan) -> chan_range

Return a `Range` of `nfchan` channels centered on `fchan`.
"""
function chanspan(fchan, nfchan)
    (0:nfchan-1) .- nfchan÷2 .+ fchan
end

"""
    upchan_extract(blks; cchan=16, fchan=124871, nfchan=32)

Uses CUDA to up-channelize coarse channel `cchan` in 4-dimensional GUPPI RAW
data blocks in `blks` along the time axis.  For each block, fftshift the
resulting fine channels, then select `nfchan` channels around fine channel
`fchan`.  The resulting data fram each block are treated as time samples and
retuned in an output Array that is dimensioned as: `(ntime, nfine, nant, npol)`,
where `ntime` is the number of output time samples (i.e. number of input
blocks), `nfine` is the number of fine channels (i.e. number of time samples per
input block), `nant` is the number of antennas, and `npol` is the number of
polarizations.
"""
function upchan_extract(blks; cchan=16, fchan=124871, nfchan=32)
    np = size(blks[1], 1)
    nt = size(blks[1], 2)
    nc = size(blks[1], 3)
    na = ndims(blks[1]) < 4 ? 1 : size(blks[1], 4)
    nb = length(blks)
    # _ptca indicates (pol, time, chan, ant) indexing
    gpuint_ptca = CuArray{Complex{Int8}}(undef, np, nt, nc, na)
    # _pta indicates (pol, time, ant) indexing
    gpuint_pta = @view gpuint_ptca[:, :, cchan, :]
    # _tap indicates (time, ant, pol) indexing
    fftin_tap = CuArray{ComplexF32}(undef, nt, na, np)
    plan = plan_fft(fftin_tap, 1)
    # _fap indicates (fine, ant, pol) indexing
    fftout_unshifted_fap = similar(fftin_tap)
    fftout_fap = similar(fftin_tap)
    # _fap_sel indicates selected subset of _fap
    fftout_fap_sel = @view fftout_fap[chanspan(fchan, nfchan), :, :]
    # Output arrays
    # _tfap indicated (time, fine, ant, pol)
    voutgpu_tfap = CuArray{ComplexF32}(undef, nb, nfchan, na, np)
    voutcpu_tfap = Array{ComplexF32}(undef, nb, nfchan, na, np)

    # For each block
    for (b, blk) in ProgressBar(enumerate(blks))
        # Copy block to gpuint
        copyto!(gpuint_ptca, blk)

        # Copy gpuint_pta to fftin_tap
        for (f, g) in zip(eachslice(fftin_tap; dims=3), eachslice(gpuint_pta; dims=1))
            copyto!(f, g)
        end

        # FFT fftin_tap to fftout_unshifted_fap
        mul!(fftout_unshifted_fap, plan, fftin_tap)

        # FFT shift into fftout_fap
        fftshift!(fftout_fap, fftout_unshifted_fap, 1)

        # Copy fftout_fap_sel to voutgputtfap
        copyto!(@view(voutgpu_tfap[b,:,:,:]), fftout_fap_sel)
    end

    # Wait for final batch to finish (still needed?)
    CUDA.synchronize()

    # Copy voutgpu_tfap to voutcpu_tfap
    copy!(voutcpu_tfap, voutgpu_tfap)

    # Return voutcpu_tfap
    voutcpu_tfap
end

function plotspan(qf, pk, antnames; r=45000, foff=10^6/2^17, kwargs...)
    fchan = pk[1]
    cchan = pk[2]
    ant = pk[3]
    r1 = min(fchan-1, r)
    r2 = min(size(qf,1)-fchan, r)
    rr = -r1:r2
    @show rr rr.*foff./1e3
    scatter(rr.*foff./1e3, log10.(qf[fchan.+rr, cchan, ant]);
        title="$(antnames[ant]) c$(cchan) f$(fchan)",
        xlabel="Frequency offset [kHz]",
        ylabel="Power [log10(counts)]",
        xlims=(-r,r).*abs(foff)./1e3,
        ms=2, msw=0, legend=false, kwargs...
    )
end

function plotspan_linear(qf, pk, antnames; r=45000, foff=10^6/2^17)
    fchan = pk[1]
    cchan = pk[2]
    ant = pk[3]
    r1 = min(fchan-1, r)
    r2 = min(size(qf,1)-fchan, r)
    rr = -r1:r2
    scatter(rr.*foff./1e3, (qf[fchan.+rr, cchan, ant]),
        title="$(antnames[ant]) c$(cchan) f$(fchan)",
        xlabel="Frequency offset [kHz]",
        ylabel="Power [σ above mean]",
        xlims=(-r,r).*foff./1e3,
        ms=2, msw=0, legend=false
    )
end

# This should probably be part of Blio.GuppiRaw
function guppi_raw_header(h)
    grh = GuppiRaw.Header()
    grh[:backend] = h[:backend]
    for (k,v) in zip(names(h), values(h))
        k == "backend" && continue
        grh[Symbol(k)] = v
    end
    grh
end

function upchan_extract_guppiraw(globpattern, DIR; cchan, fchan, nfchan=32, sideband=:native)
    #DIR = "/mnt/buf1/voyager/GUPPI"
    #cchan = 13
    #fchan = 61585

    #hdrs, blks = loadrawobs("*60045*.4.1.*.????.raw", DIR)
    hdrs, blks = loadrawobs(globpattern, DIR)

    grh = guppi_raw_header(hdrs[1,:])
    # upchan_extract returns an Array dimensioned: `(time, chan, ant, pol)`
    vv_tcap = upchan_extract(blks; cchan, fchan, nfchan)

    # Requantize vv from Complex{Float32} to Complex{Int8}
    vvi8_tcap = requantize_to_i8(vv_tcap)
    # Permute dims to `(pol, time, chan, ant)`
    vvi8_ptca = permutedims(vvi8_tcap, (4,1,2,3))

    # Gather relevant metadata
    nfpc = size(blks[1], 2)
    cfreqs = collect(Iterators.partition(calcfreqs(grh, nfpc), nfpc))
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
    # TODO Fix ra/dec and ra_str/dec_str

    # If lower sideband (i.e. spectrally flipped), unflip
    if (sideband == :upper && grh[:chan_bw] < 0) || (sideband == :lower && grh[:chan_bw] > 0)
        grh[:chan_bw] *= -1
        grh[:obsbw] *= -1
        reverse!(vvi8_ptca, dims=3)
    end

    # Finalize blks to release mmap'd regions
    foreach(finalize, blks)

    return grh, vvi8_ptca
end

function requantize_to_i8(vv::Array{ComplexF32})
    vvf32 = reinterpret(Float32, vv)
    sigma = std(vvf32)
    nsigma = maximum(abs.(extrema(vvf32))) / sigma
    scale = 120 ÷ nsigma
    convert(Array{Complex{Int8}}, round.(scale .* vv ./ sigma))
end

function write_raw_file(filename, grh, data)
    io = open(filename, "w")
    write(io, grh)
    write(io, data)
    close(io)
end

function lo(k,n,N=2^17)::Vector{ComplexF32}
    cispi.((0:n-1) .* (-2f0*k/N))
end

function lotest(bb, b, k)
    bb .= b .* lo(k, size(b, 1))
    # Get vector/scalar ratio per antenna
    1 - mean(abs.(sum(bb, dims=1))/sum(abs.(bb), dims=1))
end

"""
    fqav(A, n::Integer; f=sum)

Reduce every `n` elements of the first dimension of `A` to a single value using
function `f`.
"""
function fqav(A, n::Integer; f=sum)
    n <= 1 && return A
    sz = (n, :, size(A)[2:end]...)
    dropdims(f(reshape(A,sz), dims=1), dims=1)
end

"""
    fqav(A::AbstractRange, n::Integer)

Return a range whose elements are the mean of every `n` elements of `r`.
"""
function fqav(r::AbstractRange, n::Integer)
    n <= 1 && return r
    fch1 = first(r) + (n-1)*step(r)/2
    foff = n * step(r)
    nchan = length(r) ÷ n
    range(fch1; step=foff, length=nchan)
end