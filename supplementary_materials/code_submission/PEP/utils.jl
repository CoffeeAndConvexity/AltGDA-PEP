using JuMP, MosekTools, OffsetArrays, Gurobi, Ipopt, JLD2, Distributions, OrderedCollections, BenchmarkTools
using LinearAlgebra
using PrettyTables
using LDLFactorizations
using Mosek

dict = Dict{Tuple{Int64, Int64}, Float64}()
dict[0, 0] = 1.0
dict
Containers.SparseAxisArray(dict)

# construct e_i in R^n
function e_i(n, i)
    e_i_vec = zeros(n, 1)
    e_i_vec[i] = 1
    return e_i_vec
end

function nov(x) # provides no offset view of x
    return OffsetArrays.no_offset_view(x)
end

# this symmetric outer product is used when a is constant, b is a JuMP variable
function ⊙(a,b)
    return ((a*b') .+ transpose(a*b')) ./ 2
end

# this symmetric outer product is for computing ⊙(a,a) where a is a JuMP variable
function ⊙(a)
    return a*transpose(a)
end

# In summary, `fix_if_small_coefficient` due to Oscar Dowson at JuMP below is a utility for reducing the complexity of an optimization model by setting variables that have very small coefficients in all the affine constraints.  
function fix_if_small_coefficient!(model; atol = 1e-6)

    @info "[🐶 ] Fixing variables that have coefficients less than $atol in all constraints"

    @show list_of_constraint_types(model)

    to_check = Set(all_variables(model))  # Create a set of all variables in the model to be checked

    for (F, S) in list_of_constraint_types(model)

        if F != AffExpr
            continue  # Skip non-affine constraints, only affine expressions are considered
        end

        for ci in all_constraints(model, F, S)  # Loop through all affine constraints

            object = constraint_object(ci)  # Get the constraint object for the current constraint
            for xi in to_check  # Iterate over each variable in the set to be checked

                if abs(coefficient(object.func, xi)) >= atol  # Check if the absolute coefficient value is above the threshold
                    pop!(to_check, xi)  # Remove the variable from the set if its coefficient is significant
                end
            end
        end
    end

    for xi in to_check  # Loop through remaining variables with small coefficients
        @info "[🐱 ] Fixing $xi to 0"  # Print a message that the variable is being fixed
        fix(xi, 0.0; force = true)  # Fix the variable at 0, forcing the fix despite potential model constraints
    end

    return  # Explicit return of nothing
end

## Data generator function for H_{x,v}
function generator_function_Hxv(N, α, ϕ, ι_x; q = 1, input_type = :stepsize_constant)
    # We want to create indicator matrices 𝐱, 𝐟′, 𝐯 such that we can select elements from H_{x, u} and construct encoder matrices;
    # N is #iterations; α is the converted stepsize matrix for 𝐟′; ϕ is the converted stepsize matrix for 𝐯  

    dim_𝐱 = 2 * N + 11
    dim_𝐟′ = 2 * N + 11
    dim_𝐯 = 2 * N + 11

    # Construct the 𝐱, 𝐟′, 𝐯 vectors 
    # 𝐱 = [𝐱_{-3} ∣ 𝐱_{-2} ∣ 𝐱_{-1} ∣ 𝐱_0 ∣ 𝐱_1 | … ∣ 𝐱_N] ∈ ℝ^{dim_𝐱, N+4}
    # 𝐟′ = [𝐟′_{-3} ∣ 𝐟′_{-2} ∣ 𝐟′_{-1} ∣ 𝐟′_0 ∣ 𝐟′_1 | … ∣ 𝐟′_N] ∈ ℝ^{dim_𝐟′, N+4}
    # 𝐯 = [𝐯_{-3} ∣ 𝐯_{-2} ∣ 𝐯_{-1} ∣ 𝐯_0 ∣ 𝐯_1 | … ∣ 𝐯_N] ∈ ℝ^{dim_𝐯, N+4}
    
    # [initialization] index: e.g., (1:dim_𝐱(default), -3:N(fitting 𝐱_{-3} - 𝐱_N))
    if input_type == :stepsize_constant 
          𝐱 = OffsetArray(Matrix{Float64}(undef, dim_𝐱, N+4), 1:dim_𝐱, -3:N) 
    elseif input_type == :stepsize_variable
        𝐱 = OffsetArray(Matrix{AffExpr}(undef, dim_𝐱, N+4), 1:dim_𝐱, -3:N) 
    else
        error("Invalid input_type")
    end

    𝐟′ = OffsetArray(Matrix{Float64}(undef, dim_𝐟′, N+4), 1:dim_𝐟′, -3:N) 
    𝐯 = OffsetArray(Matrix{Float64}(undef, dim_𝐯, N+4), 1:dim_𝐯, -3:N) 

    # Construct 𝐱_{-2} ∣ 𝐱_{-1} ∣ 𝐱_0
    𝐱[:,-2] = e_i(dim_𝐱, 1)
    𝐱[:,-1] = e_i(dim_𝐱, 2)
    𝐱[:,0] = e_i(dim_𝐱, 3)
    𝐱_0 = e_i(dim_𝐱, 3)
    
    # Construct 𝐟′_i and 𝐯_i, i = -3,…,N
    for i in -3:N
        𝐟′[:,i] = e_i(dim_𝐟′, 7 + i)
        𝐯[:,i] = e_i(dim_𝐯, 7 + (N + 4) + i)
    end

    # Construct 𝐱_i, i = 1,…,N
    for i in 1:N
        𝐱[:, i] = ι_x * 𝐱_0 - sum(α[i, j] * 𝐟′[:, j] for j in 0:i) - sum(ϕ[i, j] * 𝐯[:, j] for j in 0:i-1)
    end

    # Construct 𝐮_{-3}
    𝐱[:, -3] = sum(j^q * 𝐱[:, j] for j in 1:N) / sum(j^q for j in 1:N)
    
    return 𝐱, 𝐟′, 𝐯
end

## Data generator function for H_{u,y}
function generator_function_Huy(N, β, ψ, ι_u; q = 1, input_type = :stepsize_constant)   

    # Create 𝐮, 𝐡′, 𝐲 analogously
    dim_𝐮 = 2 * N + 11
    dim_𝐡′ = 2 * N + 11
    dim_𝐲 = 2 * N + 11

    # Construct the 𝐮, 𝐡′, 𝐲 vectors 
    # 𝐮 = [𝐮_{-3} ∣ 𝐮_{-2} ∣ 𝐮_{-1} ∣ 𝐮_0 ∣ 𝐮_1 | … ∣ 𝐮_N] ∈ ℝ^{dim_𝐮, N+4}
    # 𝐡′ = [𝐡′_{-3} ∣ 𝐡′_{-2} ∣ 𝐡′_{-1} ∣ 𝐡′_0 ∣ 𝐡′_1 | … ∣ 𝐡′_N] ∈ ℝ^{dim_𝐡′, N+4}
    # 𝐲 = [𝐲_{-3} ∣ 𝐲_{-2} ∣ 𝐲_{-1} ∣ 𝐲_0 ∣ 𝐲_1 | … ∣ 𝐲_N] ∈ ℝ^{dim_𝐲, N+4}
    
    # [initialization]
    if input_type == :stepsize_constant 
        𝐮 = OffsetArray(Matrix{Float64}(undef, dim_𝐮, N+4), 1:dim_𝐮, -3:N)
    elseif input_type == :stepsize_variable
        𝐮 = OffsetArray(Matrix{AffExpr}(undef, dim_𝐮, N+4), 1:dim_𝐮, -3:N)
    else
        error("Invalid input_type")
    end
     
    𝐡′ = OffsetArray(Matrix{Float64}(undef, dim_𝐡′, N+4), 1:dim_𝐡′, -3:N) 
    𝐲 = OffsetArray(Matrix{Float64}(undef, dim_𝐲, N+4), 1:dim_𝐲, -3:N) 

    # Construct 𝐮_{-2} ∣ 𝐮_{-1} ∣ 𝐮_0
    𝐮[:,-2] = e_i(dim_𝐮, 1)
    𝐮[:,-1] = e_i(dim_𝐮, 2)
    𝐮[:,0] = e_i(dim_𝐮, 3)
    𝐮_0 = e_i(dim_𝐮, 3)
    
    # Construct 𝐡′_i and 𝐲_i, i = -3,…,N
    for i in -3:N
        𝐡′[:,i] = e_i(dim_𝐡′, 7 + i)
        𝐲[:,i] = e_i(dim_𝐲, 7 + (N + 4) + i)
    end

    # Construct 𝐮_i, i = 1,…,N
    for i in 1:N
        𝐮[:, i] = ι_u*𝐮_0 - sum(β[i, j] * 𝐡′[:, j] for j in 0:i) - sum(ψ[i, j] * 𝐲[:, j] for j in 0:i)
    end

    # Construct 𝐮_{-3}
    𝐮[:, -3] = sum(j^q * 𝐮[:, j] for j in 1:N) / sum(j^q for j in 1:N)

    return 𝐮, 𝐡′, 𝐲
end

# Constructer for A^[f]_{i, j}
function A_mat_f(i, j, 𝐟′, 𝐱)
    return ⊙(𝐟′[:,j], 𝐱[:,i] - 𝐱[:,j])
end 

# Constructer for A^[h]_{i, j}
function A_mat_h(i, j, 𝐡′, 𝐮)
    return ⊙(𝐡′[:,j], 𝐮[:,i] - 𝐮[:,j])
end 

# Constructer for B^[f]_{i, j}
function B_mat_f(i, j, 𝐱)
    return ⊙(𝐱[:,i] - 𝐱[:,j], 𝐱[:,i] - 𝐱[:,j])
end 

# Constructer for B^[h]_{i, j}
function B_mat_h(i, j, 𝐮)
    return ⊙(𝐮[:,i] - 𝐮[:,j], 𝐮[:,i] - 𝐮[:,j])
end 

# Constructer for C^[f]_{i, j} and E^[f]_{i, j}
function C_mat_f(i, j, 𝐱, 𝐯)
    return ⊙(𝐱[:,i], 𝐯[:,j])
end 

function E_mat_f(i, j, 𝐱, 𝐯)
    return C_mat_f(i, j, 𝐱, 𝐯) - C_mat_f(j, i, 𝐱, 𝐯)
end

# Constructer for C^[h]_{i, j} and E^[h]_{i, j}
function C_mat_h(i, j, 𝐮, 𝐲)
    return ⊙(𝐲[:,i], 𝐮[:,j])
end 

function E_mat_h(i, j, 𝐮, 𝐲)
    return C_mat_h(i, j, 𝐮, 𝐲) - C_mat_h(j, i, 𝐮, 𝐲)
end

# Constructer for S^[f]_{i, j}
function S_mat_f(i, 𝐟′, 𝐯)
    return ⊙(𝐟′[:,i] + 𝐯[:,i], 𝐟′[:,i] + 𝐯[:,i])
end 

# Constructer for S^[h]_{i, j}
function S_mat_h(i, 𝐡′, 𝐲)
    return ⊙(𝐡′[:,i] - 𝐲[:,i], 𝐡′[:,i] - 𝐲[:,i])
end 

# Additional encoder matrices for finite radius constraint

# D_mat_f(i, j, 𝐱) = 𝐱[:,i] ⊙ 𝐱[:,j]
function D_mat_f(i, j, 𝐱)
    return ⊙(𝐱[:,i], 𝐱[:,j])
end

# D_mat_h(i, j, 𝐮) = 𝐮[:,i] ⊙ 𝐮[:,j]
function D_mat_h(i, j, 𝐮)
    return ⊙(𝐮[:,i], 𝐮[:,j])
end

# Additional encoder matrices for bounding normnal cones (recall only the directions matter)

# J_mat_f(i, 𝐟′) = 𝐟′[:,i] ⊙ 𝐟′[:,i]
function J_mat_f(i, 𝐟′)
    return ⊙(𝐟′[:,i], 𝐟′[:,i])
end

# J_mat_h(i, 𝐡′) = 𝐡′[:,i] ⊙ 𝐡′[:,i]
function J_mat_h(i, 𝐡′)
    return ⊙(𝐡′[:,i], 𝐡′[:,i])
end

## Generates feasible stepsize matrices for the input first-order algorithm 
function feasible_stepsize_generator(N, α_alg, β_alg; alg=:AltGDA)

    # 1, 1 for OMD type algorithms and 0, 0 for FTRL type algorithms
    ι_x_input, ι_u_input = 1, 1
    if alg == :FTRL || alg == :OFTRL
        ι_x_input, ι_u_input = 0, 0
    end

    # construct α, ϕ, β, ψ corresponding to Alternating Gradient Descent-ascent (AltGDA)  algorithm
    if alg == :AltGDA
        # α_alg: [0:N-1], β_alg: [0:N-1]
    
        # construct α, i.e., {α_{i,j}}_{i in 1:N, j in 0:i}	
        α = OffsetArray(zeros(N, N+1), 1:N, 0:N)
        for i in 1:N 
            for j in 1:i
                α[i,j] = α_alg[j - 1] 
            end
        end

        # construct β
        β = OffsetArray(zeros(N, N+1), 1:N, 0:N)
        for i in 1:N 
            for j in 1:i
                β[i,j] = β_alg[j - 1] 
            end
        end

        # construct ϕ, i.e., {ϕ_{i,j}}_{i in 1:N, j in 0:i-1}
        ϕ = OffsetArray(zeros(N, N), 1:N, 0:N-1)
        for i in 1:N 
            for j in 0:i-1
                ϕ[i,j] = α_alg[j] 
            end
        end
        
        # construct ψ
        ψ = OffsetArray(zeros(N, N+1), 1:N, 0:N)
        for i in 1:N
            for j in 1:i  # ALTERNATING 
                ψ[i,j] = - β_alg[j - 1]
            end
        end
    end
    
    # construct α, ϕ, β, ψ corresponding to Simultaneous Gradient Descent-ascent (SimGDA)  algorithm
    if alg == :SimGDA
        # α_alg: [0:N-1], β_alg: [0:N-1]
    
        # construct α i.e., {α_{i,j}}_{i in 1:N, j in 0:i}	
        α = OffsetArray(zeros(N, N+1), 1:N, 0:N)
        for i in 1:N 
            for j in 1:i
                α[i,j] = α_alg[j - 1] 
            end
        end

        # construct β
        β = OffsetArray(zeros(N, N+1), 1:N, 0:N)
        for i in 1:N 
            for j in 1:i
                β[i,j] = β_alg[j - 1] 
            end
        end

        # construct ϕ i.e., {ϕ_{i,j}}_{i in 1:N, j in 0:i-1}
        ϕ = OffsetArray(zeros(N, N), 1:N, 0:N-1)
        for i in 1:N 
            for j in 0:i-1
                ϕ[i,j] = α_alg[j] 
            end
        end
        
        # construct ψ
        ψ = OffsetArray(zeros(N, N+1), 1:N, 0:N)
        for i in 1:N
            for j in 0:i-1
                ψ[i,j] = - β_alg[j]
            end
        end
    end


    # Return the feasible stepsizes
    return ι_x_input, ι_u_input, α, ϕ, β, ψ
end