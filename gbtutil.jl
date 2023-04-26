using BlioSpectroscopy
using HDF5, H5Zbitshuffle
using StatsBase
using Plots
using Glob

function hirescoarsekmax(scan, tsamp::Integer=1; nfpc=2^20, h5dir=geth5root())
    scanz = lpad(scan, 4, '0')
    h5names = glob("*_$(scanz).rawspec.0000.h5", h5dir)
    if isempty(h5names)
        @warn "scan $scanz not found in $h5dir"
        return nothing
    end
    d = h5open(h5names[1]) do h5
        h5["data"][:,1,tsamp]
    end
    dcoarse = reshape(d, nfpc, :)
    # De-spike
    dcoarse[nfpc÷2+1,:] = dcoarse[nfpc÷2,:]
    kurtosis.(eachcol(dcoarse))
end

function hirescoarsepeaks(scan, tsamp::Integer=1; nfpc=2^20, h5dir=geth5root())
    scanz = lpad(scan, 4, '0')
    h5names = glob("*_$(scanz).rawspec.0000.h5", h5dir)
    if isempty(h5names)
        @warn "scan $scanz not found in $h5dir"
        return nothing
    end
    d = h5open(h5names[1]) do h5
        h5["data"][:,1,tsamp]
    end
    dcoarse = reshape(d, nfpc, :)
    # De-spike
    dcoarse[nfpc÷2+1,:] = dcoarse[nfpc÷2,:]
    vec.(findmax(dcoarse, dims=1))
end

function hireswf(scan, fchan; cchan=1, nfchan=128, nfpc=2^20, h5dir=geth5root())
    scanz = lpad(scan, 4, '0')
    h5names = glob("*_$(scanz).rawspec.0000.h5", h5dir)
    if isempty(h5names)
        @warn "scan $scanz not found in $h5dir"
        return nothing
    end

    r = range(-nfchan÷2, length=nfchan)
    fchan += (cchan-1) * nfpc

    h5d, foff = h5open(h5names[1]) do h5
        h5["data"][r.+fchan,1,:], abs(h5["data"]["foff"][]) * 1e6
    end

    heatmap(r.*foff, 1:size(h5d,2), log10.(h5d'), yflip=true,
        title="Scan $scanz centered at fine channel $fchan", titlefontsize=12,
        xlabel="Frequency offset [Hz]", ylabel="Time sample"
    )
end

function hireswf(scan, center::CartesianIndex{2}; nfchan=128, nfpc=2^20, h5dir=geth5root())
    hireswf(scan, center[1]; cchan=center[2], nfchan, nfpc, h5dir)
end
