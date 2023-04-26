using BlioSpectroscopy
using HDF5, H5Zbitshuffle
using Plots
using Glob

function hireswf(scan, fchan, nfchan=128; h5dir=geth5root())
    scanz = lpad(scan, 4, '0')
    h5names = glob("*_$(scanz).rawspec.0000.h5", h5dir)
    if isempty(h5names)
        @warn "scan $scanz not found in $h5dir"
        return nothing
    end

    r = range(-nfchan÷2, length=nfchan)

    h5d, foff = h5open(h5names[1]) do h5
        h5["data"][r.+fchan,1,:], abs(h5["data"]["foff"][]) * 1e6
    end

    heatmap(r.*foff, 1:size(h5d,2), log10.(h5d'), yflip=true,
        title="Scan $scanz centered at fine channel $fchan", titlefontsize=12,
        xlabel="Frequency offset [Hz]", ylabel="Time sample"
    )
end
