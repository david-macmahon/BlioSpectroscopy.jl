using Preferences

getrawroot(default::AbstractString=".") = @load_preference("RAWROOT", default)
setrawroot(dir::AbstractString) = @set_preferences!("RAWROOT"=>dir)

geth5root(default::AbstractString=".") = @load_preference("H5ROOT", default)
seth5root(dir::AbstractString) = @set_preferences!("H5ROOT"=>dir)

getcudadev() = @load_preference("CUDADEV", 0)
setcudadev(devnum::Integer) = @set_preferences!("CUDADEV"=>devnum)

getsitedesc() = @load_preference("SITEDESC", "unknown site")
setsitedesc(s::AbstractString) = @set_preferences!("SITEDESC"=>s)