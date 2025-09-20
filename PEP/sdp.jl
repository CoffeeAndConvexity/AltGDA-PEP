using JuMP, MosekTools, Mosek, LinearAlgebra, OffsetArrays, Gurobi, Ipopt, JLD2, Distributions, OrderedCollections, BenchmarkTools

include("utils.jl")

function solve_primal_with_known_stepsizes(
        N,  # number of iterations
        α, β, ϕ, ψ,  # stepsize matrices
        c,  # c = [c_avg, c_min, c_lst, c_iia], only one entry of c is 1 and the rest are 0
        D_x_input, D_u_input,  # diameter of the sets X, U, i.e., ||x-y||^2 ≤ D_x^2, ||u-v||^2 ≤ D_u^2, where x, y ∈ X, u, v ∈ U
        R_x_input, R_u_input,  # radius of the sets X, U, i.e., ||x||^2 ≤ R_x^2, ||u||^2 ≤ R_u^2, where x ∈ X, u ∈ U
        L,  # maximum singular value of A
        q_input,  # term used in the definition of the weighted average
        ι_x_input, ι_u_input  # initial point x_0, u_0 used in x_i, u_i or not
        ; 
        # options
        # =======
        show_output = :on, # options are :on and :off
        radius_constr = :on, # options are :on and :off
        diam_constr = :off, # options are :on and :off
        simplex_specific_constraints = :off, # options are :on and :off, 
        minimize_printing = :off, 
        mosek_params_pfeastol = 1e-4,
        mosek_params_dfeastol = 1e-4
    )

    # Create index set of points in the primal set and dual set

    I_N = filter(x -> x != -1, -3:N) 

    # Default values of diameter and radius for the primal and dual player sets
    # ========================================================================

    if simplex_specific_constraints == :on
        D_x, D_u, R_x, R_u = sqrt(2), sqrt(2), 1, 1
    else
        D_x, D_u, R_x, R_u = D_x_input, D_u_input, R_x_input, R_u_input
    end

    # Generate the bold vector selector matrices
    # ==========================================
    𝐱, 𝐟′, 𝐯 = generator_function_Hxv(N, α, ϕ, ι_x_input; q = q_input, input_type = :stepsize_constant)
    𝐮, 𝐡′, 𝐲 = generator_function_Huy(N, β, ψ, ι_u_input; q = q_input, input_type = :stepsize_constant)
     
    # Define the model 
    # ===============
    model_primal_PEP_with_known_stepsizes = Model(optimizer_with_attributes(Mosek.Optimizer))
    set_attribute(model_primal_PEP_with_known_stepsizes, "MSK_DPAR_INTPNT_CO_TOL_PFEAS", mosek_params_pfeastol)
    set_attribute(model_primal_PEP_with_known_stepsizes, "MSK_DPAR_INTPNT_CO_TOL_DFEAS", mosek_params_dfeastol)

    # Add the variables 
    # =================
    dim_G_xv, dim_G_uy = (2 * N) + 11, (2 * N) + 11

    # positive smidefinite variable G_{xv} and G_{uy}
    @variable(model_primal_PEP_with_known_stepsizes, G_xv[1:dim_G_xv, 1:dim_G_xv], PSD)
    @variable(model_primal_PEP_with_known_stepsizes, G_uy[1:dim_G_uy, 1:dim_G_uy], PSD)

    # scalar variable ν free
    @variable(model_primal_PEP_with_known_stepsizes, ν)

    # Add the objective
    # =================

    # weights of performance measures (usually 1 for some measure and 0 for others)
    c_avg, c_min, c_lst, c_iia = c
    @objective(model_primal_PEP_with_known_stepsizes, Max, 
        c_avg * tr(G_uy * (sum(E_mat_h(i, -2, 𝐮, 𝐲) for i in 1:N) / N)) 
        + c_min * ν 
        + c_lst * tr(G_uy * E_mat_h(N, -2, 𝐮, 𝐲)) 
        + c_iia * tr(G_uy * E_mat_h(-3, -2, 𝐮, 𝐲))
    )

    # Add the constraints 
    # ===================

    # epigraph constraint for the average performance measure: ν ≤ 𝐭𝐫(G_{u, y} E^[h]_{i, -2}), i ∈ [1: N]
    if c_min >= 1e-6 
        @constraint(model_primal_PEP_with_known_stepsizes, con_epigraph[i = 1:N], ν ≤ tr(G_uy * E_mat_h(i, -2, 𝐮, 𝐲)))
    end

    # indicator constraint for primal set
    # 𝐭𝐫(G_{x, v} A^[f]_{i, j}) ≤ 0, i, j ∈ I_N
    # ============================================
    @constraint(model_primal_PEP_with_known_stepsizes, con_indicator_primal[i in I_N, j in I_N, i != j], tr(G_xv * A_mat_f(i, j, 𝐟′, 𝐱)) ≤ 0)


    # indicator constraint for dual player set
    # 𝐭𝐫(G_{u, y} A^[h]_{i, j}) ≤ 0, i, j ∈ I_N
    # ========================================== 
    @constraint(model_primal_PEP_with_known_stepsizes, con_indicator_dual[i in I_N, j in I_N, i != j], tr(G_uy * A_mat_h(i, j, 𝐡′, 𝐮)) ≤ 0)

    # the constraints related to modeling
    # Y = AX, V = A'U, σ_max(A) <= L 
    # =================================

    # X' V - Y' U = 0
    # 𝐭𝐫(G_{x, v} C^[f]_{i, j}) - 𝐭𝐫(G_{u, y} C^[h]_{i, j}) = 0, i, j ∈ [-3: N] 
    # ================================================================================
    @constraint(model_primal_PEP_with_known_stepsizes, con_matrix_1[i = -3:N, j = -3:N], tr(G_xv * C_mat_f(i, j, 𝐱, 𝐯)) - tr(G_uy * C_mat_h(i, j, 𝐮, 𝐲)) == 0)

    # 𝐘^⊤ G_{u, y} 𝐘 - L^2 𝐗^⊤ G_{x, v} 𝐗 ⪯ 0
    # ===============================================
    𝐲_nov = OffsetArrays.no_offset_view(𝐲)
    𝐱_nov = OffsetArrays.no_offset_view(𝐱)
    @constraint(model_primal_PEP_with_known_stepsizes, con_matrix_2, -(𝐲_nov'*G_uy*𝐲_nov) + (L^2*𝐱_nov'*G_xv*𝐱_nov) >= 0, PSDCone())

    # 𝐕^⊤ G_{x, v} 𝐕 - L^2 𝐔^⊤ G_{u, y} 𝐔 ⪯ 0
    # ===============================================
    𝐯_nov = OffsetArrays.no_offset_view(𝐯)
    𝐮_nov = OffsetArrays.no_offset_view(𝐮)
    @constraint(model_primal_PEP_with_known_stepsizes, con_matrix_3, -(𝐯_nov' * G_xv * 𝐯_nov) + (L^2 * 𝐮_nov' * G_uy * 𝐮_nov) in PSDCone())

    if diam_constr == :on
        if minimize_printing == :off
            @info "[🌹 ] Diameter constraints are turned on."
        end

        # diameter constraint for primal player set: 𝐭𝐫(G_{x, v} B^[f]_{i, j}) ≤ D_x^2, i, j ∈ I_N 
        @constraint(model_primal_PEP_with_known_stepsizes, con_diameter_primal[i in I_N, j in I_N, i != j], tr(G_xv * B_mat_f(i, j, 𝐱)) ≤ D_x^2)

        # diameter constraint for dual player set: 𝐭𝐫(G_{u, y} B^[h]_{i, j}) ≤ D_u^2, i, j ∈ I_N 
        @constraint(model_primal_PEP_with_known_stepsizes, con_diameter_dual[i in I_N, j in I_N, i != j], tr(G_uy * B_mat_h(i, j, 𝐮)) ≤ D_u^2)
    elseif diam_constr == :off
        if minimize_printing == :off
            @info "[🌹 ] Diameter constraints are turned off."
        end
    else
        @error "[🌹 ] Diameter constraints are not properly set."
    end

    if radius_constr == :on
        if minimize_printing == :off
            @info "[🌹 ] Radius constraints are turned on."
        end

        # Radius constraints for primal player set: 𝐭𝐫(G_{x, v} D^[f]_{i, i}) ≤ R_x^2, i ∈ I_N
        @constraint(model_primal_PEP_with_known_stepsizes, con_radius_primal[i in I_N], tr(G_xv * D_mat_f(i, i, 𝐱)) ≤ R_x^2)

        # Radius constraints for dual player set: 𝐭𝐫(G_{u, y} D^[h]_{i, i}) ≤ R_u^2, i ∈ I_N
        @constraint(model_primal_PEP_with_known_stepsizes, con_radius_dual[i in I_N], tr(G_uy * D_mat_h(i, i, 𝐮)) ≤ R_u^2)
    elseif radius_constr == :off
        if minimize_printing == :off
            @info "[🌹 ] Radius constraints are turned off."
        end
    else
        @error "[🌹 ] Radius constraints are not properly set."
    end

    if simplex_specific_constraints == :on
        if minimize_printing == :off
            @info "[🌹 ] Probability simplex constraints are turned on."
        end

        # Positivity constraints
        # ======================

        # constraint lower bound for probability simplex of dual player tr(G_uy * D_mat_h(i, j, 𝐮)) >= 0 for i,j in I_N
        @constraint(model_primal_PEP_with_known_stepsizes, con_lb_prob_simplex_dual[i in I_N, j in I_N, i != j], tr(G_uy * D_mat_h(i, j, 𝐮)) >= 0)

        # constraint upper bound for probability simplex of dual player tr(G_uy * D_mat_h(i, j, 𝐮)) <= 1 for i,j in I_N
        @constraint(model_primal_PEP_with_known_stepsizes, con_ub_prob_simplex_dual[i in I_N, j in I_N, i != j], tr(G_uy * D_mat_h(i, j, 𝐮)) <= 1)

        # add the constraint 
        # 0 <= tr G_{x,v} D^f_{i,j} <= 1, for i,j in [-3:N]

        # constraint lower bound for probability simplex of primal player tr G_{x,v} D^f_{i,j} >= 0
        @constraint(model_primal_PEP_with_known_stepsizes, con_lb_prob_simplex_primal[i in I_N, j in I_N, i != j], tr(G_xv * D_mat_f(i, j, 𝐱)) >= 0)

        # constraint lower bound for probability simplex of primal player tr G_{x,v} D^f_{i,j} >= 0
        @constraint(model_primal_PEP_with_known_stepsizes, con_ub_prob_simplex_primal[i in I_N, j in I_N, i != j], tr(G_xv * D_mat_f(i, j, 𝐱)) <= 1)

        # Hyperplane constraints
        # ======================

        # dual hyperplane constraint: tr(G_uy * D_mat_h(-1, i, 𝐮)) == 1, i in I_N 
        @constraint(model_primal_PEP_with_known_stepsizes, con_hyperplane_dual[i in I_N], tr(G_uy * D_mat_h(-1, i, 𝐮)) == 1)

        # primal hyperplane constraint: tr(G_xv * D_mat_f(-1, i, 𝐱)) == 1, i in I_N
        @constraint(model_primal_PEP_with_known_stepsizes, con_hyperplane_primal[i in I_N], tr(G_xv * D_mat_f(-1, i, 𝐱)) == 1)
    end

    #= Optimize =#
    if show_output ==:off 
        set_silent(model_primal_PEP_with_known_stepsizes)
    end


    # optimize the model

    # set_attribute(model_primal_PEP_with_known_stepsizes, "MSK_DPAR_INTPNT_CO_TOL_REL_GAP", 1e-3)
    optimize!(model_primal_PEP_with_known_stepsizes)

    #= Store and return the solution =#
    if termination_status(model_primal_PEP_with_known_stepsizes) ≠ MOI.OPTIMAL
        @warn "model_primal_PEP_with_known_stepsizes solving did not reach optimality; termination status = $(termination_status(model_primal_PEP_with_known_stepsizes))"
    end

    # Extract the optimal value of the objective function, and optimal solutions
    obj_star = objective_value(model_primal_PEP_with_known_stepsizes)

    G_xv_star = value.(G_xv)
    G_uy_star = value.(G_uy)
    ν_star = value(ν)

    if show_output == :on
        @info "[🚗 ] The optimal value of the objective function is $(obj_star)."
    end

    return obj_star, G_xv_star, G_uy_star, ν_star
end