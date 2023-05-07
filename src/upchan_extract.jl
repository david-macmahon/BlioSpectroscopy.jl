using CUDA
using CUDA.CUFFT
using LinearAlgebra: mul!

"""
    upchan_extract(blks; cchan, fchan, nfchan=32)

Uses CUDA to up-channelize coarse channel `cchan` of 3- or 4-dimensional GUPPI
RAW data blocks in `blks` along the time axis.  For each block, `fftshift` the
resulting fine channels, then select `nfchan` channels around fine channel
`fchan`.  The resulting data from each block are treated as time samples and
returned in an output Array that is dimensioned as: `(ntime, nfine, nant,
npol)`, where `ntime` is the number of output time samples (i.e. number of input
blocks), `nfine` is the number of fine channels (i.e. number of time samples per
input block), `nant` is the number of antennas, and `npol` is the number of
polarizations.
"""
function upchan_extract(blks; cchan, fchan, nfchan=32)
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
    for (b, blk) in enumerate(blks)
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

