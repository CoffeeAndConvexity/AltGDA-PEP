using HDF5, JLD 
using CSV, DataFrames

current_dir = pwd()
if current_dir[end-8:end] ≠ "/PEP/data"
    current_dir = current_dir * "/PEP/data"
end
println("$(current_dir)")

for alg in [:AltGDA, :SimGDA]
    df_combined = DataFrame()
    for subset_N in [5:30, 31:35, 36:40]
        record_res = load("$(current_dir)/$(alg)_$(first(subset_N))_$(last(subset_N)).jld")["data"]
        df = DataFrame(N=record_res["N"], optimal_η=record_res["η"], optimal_obj=record_res["optimal_obj"])
        append!(df_combined, df)
    end
    CSV.write("$(current_dir)/$(alg).csv", df_combined)
end