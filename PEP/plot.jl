using HDF5, JLD 
using Plots
using CSV, DataFrames, CSVFiles


# Plot both optimal performance measure in one plot
start_N, end_N = 5, 50  # inclusive bounds
current_dir = pwd()
if current_dir[end-8:end] ≠ "/data"
    current_dir = current_dir * "/data"
end
println("$(current_dir)")

record_csv_AltGDA = load("$(current_dir)/AltGDA.csv")
record_csv_SimGDA = load("$(current_dir)/SimGDA.csv")

current_dir = current_dir * "/../figures"

# attain the first colomn of record_csv_AltGDA
record_csv_AltGDA = DataFrame(record_csv_AltGDA)
record_csv_SimGDA = DataFrame(record_csv_SimGDA)

# Plot optimal_η vs N in a scatter plot separately, change the color of the xlabel to blue
scatter(record_csv_AltGDA.N, record_csv_AltGDA.optimal_η, label="AltGDA", markershape=:circle, xlabel="T", ylabel="Optimized Step Size (η)", title="Optimized Step Size vs N", xguidefont = font(12, "blue"), fontlabels = font(20))
savefig("$(current_dir)/AltGDA_optimized_η_vs_N.png")
scatter(record_csv_SimGDA.N, record_csv_SimGDA.optimal_η, label="SimGDA", markershape=:diamond, xlabel="T", ylabel="Optimized Step Size (η)", title="Optimized Step Size vs N")
savefig("$(current_dir)/SimGDA_optimized_η_vs_N.png")

# Plot optimal_η vs N in a scatter plot
scatter(record_csv_AltGDA.N, record_csv_AltGDA.optimal_η, label="AltGDA", markershape=:circle, xlabel="T", ylabel="Optimized Step Size (η)", title="Optimized Step Size vs T", xguidefont = font(12, "blue"), fontlabels = font(20))
scatter!(record_csv_SimGDA.N, record_csv_SimGDA.optimal_η, label="SimGDA", markershape=:diamond, xlabel="T", ylabel="Optimized Step Size (η)", xguidefont = font(12, "blue"), fontlabels = font(20), titlefontcolor = "blue")
savefig("$(current_dir)/optimized_η_vs_N.png")

# Plot optimal_obj vs N in a scatter plot
scatter(record_csv_AltGDA.N, record_csv_AltGDA.optimal_obj, label="AltGDA", xscale=:log10, yscale=:log10, markershape=:circle, xlabel="T", ylabel="Optimized Objective Value", title="Optimized Objective Value vs T", xguidefont = font(12, "blue"), fontlabels = font(20))
scatter!(record_csv_SimGDA.N, record_csv_SimGDA.optimal_obj, label="SimGDA", xscale=:log10, yscale=:log10, markershape=:diamond, xguidefont = font(12, "blue"), fontlabels = font(20), titlefontcolor = "blue")
savefig("$(current_dir)/optimized_obj_vs_N.png")