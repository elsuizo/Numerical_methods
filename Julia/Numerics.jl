__precompile__()
"""
classic numerical methods algorithms in julia 
"""
module Numeric


"""
`forward_subs(L, b)`

compute the forward substitution algorithm to solve linear system

input:
-----
L(array{number, 2}): lower triangle matrix
b(array{number, 1}): vector of coeficients

output:
------
x(array{number, 1}): vector solution of the system
"""
function forward_subs{T<:Number}(L::Array{T,2}, b::Array{T,1})
    
    n, m = size(L)
    if(n != m)
        error("The matrix must be square") 
    end
    if minimum(abs(diag(L))) == 0 
        error("The system is singular")
    end
    x = zeros(n)    
    x[1] = b[1] / L[1,1]
    # FIXME(elsuizo) cambiar L[i, 1:i]'[1:end] ⋅ x[1:i] para la version .5 que
    # si devuelve Array{Float64,1}
    for i = 2:n
        @show x[i] = (b[i] - (L[i, 1:i]'[1:end] ⋅ x[1:i])) / L[i, i]
    end
    return x
end


"""
`backward_subs(U, b)`

compute the backward substitution algorithm to solve linear system

input:
-----
U(array{number, 2}): upper triangle matrix
b(array{number, 1}): vector of coeficients

output:
------
x(array{number, 1}): vector solution of the system
"""
function backward_subs{T<:Number}(U::Array{T,2}, b::Array{T,1})
    n, m = size(U)
    
    if (n != m) 
        error("The matrix must be square")
    end
    
    if minimum(abs(diag(U))) == 0 
        error("The system is singular")
    end
    
    x = zeros(n)    
    # FIXME(elsuizo) cambiar L[i, 1:i]'[1:end] ⋅ x[1:i] para la version .5 que
    for i = n:-1:1
        x[i] = (b[i] - (U[i, i+1:n]'[1:end] ⋅ x[i+1:n])) / U[i, i] #
    end
    
    
    return x
end

"""
`lu_fact(A)`

Compute the LU factorization of a matrix A

input:
------
A(Array{Number, 2}) Matrix 

Outputs:
-------
L(Array{Number, 2}) Lower triangle matrix
U(Array{Number, 2}) Upper triangle matrix
"""
function lu_fact{T<:Number}(A::Array{T, 2})
    
    n, m = size(A)
    L = zeros(A) 
    U = zeros(A) 
    if (n != m) 
        error("The matrix must be square")
    end
    
    if minimum(abs(diag(A))) == 0 
        error("The system is singular")
    end
    for k in 1:n-1
        for i in k+1:n
            if (A[i, k] != zero(T))
                γ = A[i, k] / A[k, k]
                A[i, k+1:n] = A[i, k+1:n] - γ * A[k, k+1:n]
                A[i, k] = γ 
            end
        end
    end
    L = tril(A, -1) + eye(n)
    U = triu(A)
    return L, U

end



#-------------------------------------------------------------------------
# end of Numerics
#-------------------------------------------------------------------------
end
