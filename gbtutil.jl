using HDF5, H5Zbitshuffle
using Plots

function hireswf(scan, fchan, nfchan=128; h5dir="/datax/dibas/AGBT22B_999_42/GUPPI")
    scanz = lpad(scan, 4, '0')
    h5=h5open(glob("BLP??/*_$(scanz).rawspec.0000.h5", h5dir)[1])
    r = range(-nfchan÷2, length=nfchan)
    h5d = h5["data"][r.+fchan,1,:]
    foff = abs(h5["data"]["foff"][]) * 1e6
    heatmap(r.*foff, 1:size(h5d,2), log10.(h5d'), yflip=true,
        title="Scan $scanz centered at fine channel $fchan", titlefontsize=12,
        xlabel="Frequency offset [Hz]", ylabel="Time sample"
    )
end