using CUDA
using CUDA.CUFFT
using LinearAlgebra: mul!
import StatsBase: kurtosis

"""
    spectrokurtosoid(blks; nint=32) -> pwr, pwr2

Uses CUDA to up-channelize GUPPI RAW data blocks stored as 3- or 4-dimensional
Arrays in iterable `blks` along the time axis, detects (i.e. converts to power),
and integrates `nint` blocks into each output time sample and returns two 5
dimensional Arrays: one containing "sum of power" and the other containing "sum
of squared power".  The can be used to calculate kurtosis with or without
further integration in time and/or frequency and/or antennas (incoherent sum).
See `kurtosify` and `incokurtosify`

The input Arrays are dimensioned as: `(npol, ntime, ncoarse, nant)`.

The output Arrays are dimensioned as: `(nfine, ncoarse, nant, npol, nint)`

where `npol` is the number of polarizations, `ntime` is the number of time
samples per block, `ncoarse` is the number of (coarse) frequency channels per
antenna per block, `nant` is the number of antennas, `nfine` is the number of
(fine) frequency channels in the output, and `nint` is the number of output
integrations.  Note that `nfine == ntime` and `nint` is the time axis of the
output data.
"""
function spectrokurtosoid(blks; nint=32)
    np, nt, nc, na = size.(Ref(first(blks)), (1,2,3,4))
    ni = length(blks) ÷ nint
    gpuint = CuArray{Complex{Int8}}(undef, np, nt, nc, na)
    fftin = CuArray{ComplexF32}(undef, nt, nc, na, np)
    plan = plan_fft(fftin, 1)
    fftout = similar(fftin)
    pblk = similar(fftout, Float32, nt, nc, na, np); pblk .= 0
    p2blk = similar(fftout, Float32, nt, nc, na, np); p2blk .= 0
    # Output arrays
    pblkcpu = Array{Float32}(undef, size(pblk))
    p2blkcpu = Array{Float32}(undef, size(p2blk))
    pwrints = Array{Float32}(undef, nt, nc, na, np, ni)
    pwr2ints = Array{Float32}(undef, nt, nc, na, np, ni)

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

        # Accumulate power into pblk, p2blks
        # TODO replace with custom kernel to avoid duplicate effort
        pblk .+= abs2.(fftout) ./ (2*nint)
        p2blk .+= (real.(fftout).^4 .+ imag.(fftout).^4) ./ (2*nint)

        if b % nint == 0
            copyto!(pblkcpu, pblk)
            copyto!(p2blkcpu, p2blk)
            fftshift!(@view(pwrints[:,:,:,:,toutidx]), pblkcpu, 1)
            fftshift!(@view(pwr2ints[:,:,:,:,toutidx]), p2blkcpu, 1)
            pblk .= 0
            p2blk .= 0
            toutidx += 1
        end
    end

    # Wait for final batch to finish (still needed?)
    CUDA.synchronize()

    # Return `pwrints` as an Array{Float64}
    #convert(Array{Float64}, pwrints)
    pwrints, pwr2ints
end

"""
    ddsum([f,] x; dims)

Sum `x` along `dims` and then drop `dims`.  If `f` is given, apply `f` to each
element of `x` beforehand.
"""
function ddsum(x; dims)
    dropdims(sum(x; dims); dims)
end

function ddsum(f, x; dims)
    dropdims(sum(f, x; dims); dims)
end

"""
    abs4(z::Complex)

Returns `real(z)^4 + imag(z)^4`.  This is very much like `abs2(z)`, but raising
real and imaginary to the power of 4 rather than 2.  NB: This is very different
from `abs2(z)^2`.  Be careful when using `Complex{<:Integer}` types because they
are prone to (silent!) overflow.
"""
function abs4(z::Complex)
    real(z)^4 + imag(z)^4
end

function abs4(T::Type, z::Complex)
    abs4(convert(Complex{T}, z))
end

"""
    pick(elements::Tuple, i) -> (picked, remaining)
    pick(elements::CartesianIndex, i) -> (picked, remaining)

Pick `elements[i]` from `elements`, returning that element and a tuple of all
remaining elements.

    elements = (10,20,30)
    pick(elements, 2) # Returns (20, (10, 30))
"""
function pick(elements::Tuple, i::Integer)
    picked = elements[i]
    remaining = (elements[1:i-1]..., elements[i+1:end]...)
    (picked, remaining)
end

function pick(ci::CartesianIndex, i::Integer)
    pick(Tuple(ci), i)
end

"""
    kurtosis(p::Real, p2::Real; nsum::Real=1)::Real

Given the fourth central moment `p2` and the second central moment `p`,
calculate excess kurtosis as:

    p2 / p^2 * scale - 3

The `nsum` parameter is intended for use when `p` and `p2` have been summed,
but not yet normalized by the number of samples summed.
"""
function kurtosis(p::Real, p2::Real; nsum::Real=1)
    p2 / p^2 * nsum - 3
end

#=
function kurtosify!(dst, pwr, pwr2; scale=1)
    dst .= (pwr2 ./ (pwr.^2)) .* scale .- 3
end

function kurtosify(pwr, pwr2; scale=1)
    @assert size(pwr) == size(pwr2)
    dst = similar(pwr2)
    kurtosify!(dst, pwr, pwr2; scale)
end

function kurtosify(pwrpwr2::Tuple; scale=1)
    pwr, pwr2 = pwrpwr2
    kurtosify(pwr, pwr2; scale)
end
=#

"""
    incokurtosis(pwr, pwr2; dims=(3,5), scale=1)

Sum `pwr` and `pwr2` Arrays (e.g. as returned by `spectrokurtosoid`) over
dimensions specified by `dims` and then compute kurtosis of each
fine channel for each coarse channel and polarization.

5-dimensional `pwr` and `pwr2` are indexed as `[fine,coarse,ant,pol,time]` (as
returned by `spectrokurtosoid`).

The returned 3-dimensional kurtosis Array is indexed as `[fine, coarse, pol]`.

The `scale` keyword argument can be used if additional integration has been done
to `pwr` and `pwr2` before calling this function.
"""
function incokurtosis(pwr, pwr2; dims=(3,5), nsum=1)
    inco = ddsum(pwr; dims)
    inco2 = ddsum(pwr2; dims)
    nsum *= prod(size.(Ref(pwr), dims))
    kurtosis.(inco, inco2; nsum)
end

"""
    clusterize(fchans, maxgap=64)

Assign "nearby" fine channel indices in sorted input Vector `fchans` to
clusters, where nearby is considered to by fine channels less than `maxgap` fine
channels apart.  `fchans` must be sorted to get valid results, but this
condition is not validated by this function.
"""
function clusterize(fchans; maxgap=64)
    breaks = findall(>(maxgap), diff(fchans))
    # Handle case where there are no breaks (i.e. all one cluster)
    first_length = isempty(breaks) ? 0 : first(breaks)
    first_cluster = Iterators.repeated(1, first_length)
    middle_clusters = Iterators.repeated.(2:length(breaks), diff(breaks))
    last_length = isempty(breaks) ? length(fchans) : length(fchans)-last(breaks)
    last_cluster = Iterators.repeated(length(breaks)+1, last_length)
    Iterators.flatten((first_cluster,
                       Iterators.flatten(middle_clusters),
                       last_cluster)) |> collect
end

"""
    kurtosismap(ks; fdim=1, klim=3, maxgap=64)

Takes Array `ks` as returned by `kurtosis` or `incokurtosis`.  Finds all
indices in `ks` whose value is greater than `klim`.  Returns a Dict mapping
tuples of unique non-fine-channel dimensions of found indices to 2-tuples
consisting of a sorted Vector of corresponding fine channels and a Vector of
cluster assignments as determined by `clusterize`.  The `maxgap` keyword
argument is passed to `clusterize`.
"""
function kurtosismap(ks; fdim=1, klim=3, maxgap=64)
    pks = findall(>(klim), ks)
    d = Dict{Tuple, Vector{Int}}()
    for pk in pks
        fchan, key = pick(pk, fdim)
        haskey(d, key) || (d[key] = Int[])
        push!(d[key], fchan)
    end
    for v in values(d)
        sort!(v)
    end
    dc = Dict{Tuple, NTuple{2, Vector{Int}}}()
    for (k,v) in d
        dc[k] = (v, clusterize(v; maxgap))
    end
    return dc
end
