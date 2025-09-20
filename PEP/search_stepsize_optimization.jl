using HDF5, JLD 
using Plots
using CSV, DataFrames

include("sdp.jl")

algos = [:AltGDA, :SimGDA]
start_N = 5
end_N = 30

L = 1

function optimal_obj_ηc(N, L, alg, η_c, performance_measure)
    η = 1 / (η_c * L)  # α = β = η

    # Parameters
    D_x_input, D_u_input, R_x_input, R_u_input = sqrt(2.0), sqrt(2.0), 1.0, 1.0
    q_input = 1.0

    # Feasible stepsize generation
    α_alg = η * OffsetArray(ones(N), 0:N-1)
    β_alg = η * OffsetArray(ones(N), 0:N-1)
    ι_x_input, ι_u_input, α_input, ϕ_input, β_input, ψ_input = feasible_stepsize_generator(N, α_alg, β_alg, alg=alg) 

    # Solve primal with feasible stepsize
    sol_primal_with_known_stepsizes = solve_primal_with_known_stepsizes(
        N, α_input, β_input, ϕ_input, ψ_input, (1, 0, 0, 0), D_x_input, D_u_input, R_x_input, R_u_input, L, q_input, ι_x_input, ι_u_input; 
        show_output=:off, radius_constr=:on, diam_constr=:off, simplex_specific_constraints=:off, minimize_printing=:on
    )

    primal_obj_star, G_xv_star, G_uy_star, ν_star = sol_primal_with_known_stepsizes

    return η_c, primal_obj_star 
end

for alg in algos
    
    function global_search_optimal_ηc(N, L, alg, param_min_η_c, param_max_η_c, performance_measure; num_search_points=50)
        println("N: ", N, "; searching η_c in [", param_min_η_c, ", ", param_max_η_c, "] ... ")
        η_c_list = LinRange(param_min_η_c, param_max_η_c, num_search_points)
        opt_obj_list = [optimal_obj_ηc(N, L, alg, η_c, performance_measure)[2] for η_c in η_c_list]

        min_obj = minimum(opt_obj_list)
        opt_ηc_idx = argmin(opt_obj_list)
        η_c = η_c_list[opt_ηc_idx]

        return 1 / (η_c * L), min_obj
    end

    # main 
    println("*************************************************************************")

    # create an empty dictionary to store the results
    res = Dict{Any, Any}()
    res["N"] = start_N:end_N
    res["η"] = zeros(Float64, end_N - start_N + 1)
    res["optimal_obj"] = zeros(Float64, end_N - start_N + 1)

    # binary search for the optimal η_c
    for N in start_N:end_N
        if alg == :AltGDA
            min_η_c, max_η_c = 0.5, 1.0  # subject to tuning
        elseif alg == :SimGDA
            min_η_c, max_η_c = 0.3, 6.0  # subject to tuning
        end

        min_η_c_cur_state, max_η_c_cur_state = min_η_c, max_η_c
        num_grid = 20
        grid_width = (max_η_c_cur_state - min_η_c_cur_state) / num_grid
        η, optimal_obj = 0.0, 0.0
        while 1 / (L * min_η_c_cur_state) - 1 / (L * max_η_c_cur_state) > 1e-3
            η, optimal_obj = global_search_optimal_ηc(N, L, alg, min_η_c_cur_state, max_η_c_cur_state, :avg; num_search_points=num_grid)
            η_c = 1 / (η * L)
            min_η_c_cur_state, max_η_c_cur_state = η_c - grid_width, η_c + grid_width
            grid_width = (max_η_c_cur_state - min_η_c_cur_state) / num_grid
        end

        res["η"][N - start_N + 1] = η
        res["optimal_obj"][N - start_N + 1] = optimal_obj
        println("N: ", N, "; optimal η: ", η, "; optimal obj: ", optimal_obj)
    end

    save("($alg).jld", "data", res)

    # load the results and plot
    record_res = load("($alg).jld")["data"]

    scatter(record_res["N"], record_res["η"], label="optimal η", xscale=:log10, yscale=:log10, xlabel="N", ylabel="optimal η", title="optimal η vs N")
    savefig("($alg)_optimal_η_vs_N.png")
    scatter(record_res["N"], record_res["optimal_obj"], label="optimal obj", xscale=:log10, yscale=:log10, xlabel="N", ylabel="optimal obj", title="optimal obj vs N")
    savefig("($alg)_optimal_obj_vs_N.png")

    record_res = load("($alg).jld")["data"]
    df = DataFrame(N=record_res["N"], optimal_η=record_res["η"], optimal_obj=record_res["optimal_obj"])
    CSV.write("($alg)_results.csv", df)

    println("*************************************************************************")
    println()
end

# Plot both optimal performance measure in one plot
record_res_AltGDA = load("(AltGDA).jld")["data"]
record_res_SimGDA = load("(SimGDA).jld")["data"]
scatter(record_res_AltGDA["N"], record_res_AltGDA["optimal_obj"], label="AltGDA", xscale=:log10, yscale=:log10, xlabel="N", ylabel="optimal obj", title="optimal obj vs N")
scatter!(record_res_SimGDA["N"], record_res_SimGDA["optimal_obj"], label="SimGDA")
savefig("Both_optimal_obj_vs_N.png")