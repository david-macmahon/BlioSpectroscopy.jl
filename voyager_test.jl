using Blio
using BlioSpectroscopy
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

# For COSMIC/VLA 60045 data on cosmic-gpu-2:/mnt/buf1/voyager/GUPPI, the buf0
# data has Voyager on the edge of coarse channel 16 (and 17 thanks to aliasing).
# The buf1 data has voyager near the center of coarse channel 13 (peak is at
# fine channel 61586, but center of peaks is 61585) on cosmic-gpu-2.
#
# For cosmic-gpu-2:/mnt/buf1/voyager/GUPPI data of 60060, the carrier starts
# at/near fine channel 76137.

# For GBT 60055 data on BLP33:/datax/dibas/AGBT22B_999_41/GUPPI, coarse channel
# 7:
#   - Scan 0112: fine channel 330857/2^20, 165430/2^19
#   - Scan 0116: fine channel 330996/2^20, 165508/2^19
# The fine channel index for 2^20 fine channels are for the 0000.h5 product
# which have different time resolution and different total time) than the
# 2^19 fine channels of the "upchan_extract" with nfpc=2^19 and no time
# integration.  This leads to slight offsets between the two.

@info "Running on $(getsitedesc())"
DIR = getrawroot()
CUDA.device!(getcudadev())

"""
    getkurtosis(pwr::AbstractArray{<:Real,5})

For 5-dimensional array `pwr`, compute the kurtosis along the 5th dimension.
"""
function getkurtosis(pwr::AbstractArray{<:Real,5})
    nf, nc, na, np, nt = size(pwr)
    k = map(kurtosis, eachrow(reshape(pwr, :, nt)))
    reshape(k, nf, nc, na, np)
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