module BlioSpectroscopy
    export calc_start_time
    include("calc_start_time.jl")

    export calc_upchan_freqs
    include("calc_upchan_freqs.jl")

    export chanspan
    include("chanspan.jl")

    export spectroscopy
    include("spectroscopy.jl")
    
    export upchan_extract_guppiraw, write_raw_file
    include("upchan_extract_guppiraw.jl")

    export upchan_extract
    include("upchan_extract.jl")
end # module BlioSpectroscopy