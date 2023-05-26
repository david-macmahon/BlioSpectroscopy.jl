using CUDA
using CUDA.CUFFT
using LinearAlgebra: mul!

"""
    spectroscopy(blks; nint=32)

Uses CUDA to up-channelize GUPPI RAW data blocks stored as 3- or 4-dimensional
Arrays in iterable `blks` along the time axis, detects (i.e. converts to power),
and integrates `nint` blocks into each output time sample and returns a 5
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
    np, nt, nc, na = size.(Ref(first(blks)), (1,2,3,4))
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
    toutidx = 1
    for (b, blk) in enumerate(blks)
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
            fftshift!(@view(pwrints[:,:,:,:,toutidx]), pblkcpu, 1)
            pblk .= 0
            toutidx += 1
        end
    end

    # Wait for final batch to finish (still needed?)
    CUDA.synchronize()

    # Return `pwrints` as an Array{Float64}
    convert(Array{Float64}, pwrints)
end

