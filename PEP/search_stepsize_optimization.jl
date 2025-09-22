using HDF5, JLD 

include("sdp.jl")

algos = [:AltGDA, :SimGDA]
start_N, end_N = 41, 43  # inclusive bounds

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

    print("*")

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
            min_η_c, max_η_c = 0.7, 0.9  # subject to tuning
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

    # save results to the data folder in the current directory
    current_dir = pwd()
    if !isdir("$(current_dir)/PEP/data")
        mkdir("$(current_dir)/PEP/data")
    end
    save("$(current_dir)/PEP/data/$(alg)_$(start_N)_$(end_N).jld", "data", res)

    println("*************************************************************************")
    println()
end
