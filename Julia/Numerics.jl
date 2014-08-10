#-------------------------------------------------------------------------
# Classic numerical methods algorithms in Julia 
#-------------------------------------------------------------------------
module Numeric

export forward_subs, backward_subs

#-------------------------------------------------------------------------
# Fordward substitution to solve linear systems
#-------------------------------------------------------------------------
function forward_subs(L, b)
    
    n, m = size(L)
    
    if n != m 
        error("The matrix must be square")
    end
    
    if minimum(abs(diag(L))) == 0 
        error("The system is singular")
    end
    
    for j = 1:n-1
    
        b[j] = b[j] / L[j,j]
        b[j+1:n] = b[j+1:n] - b[j] * L[j+1:n, j]
    end
    b[n] = b[n] / L[n,n]
    return b
end

#-------------------------------------------------------------------------
# Backward substitution for solve linear systems
#-------------------------------------------------------------------------
function backward_subs(U, b)
    n, m = size(U)
    
    if n != m 
        error("The matrix must be square")
    end
    
    if minimum(abs(diag(U))) == 0 
        error("The system is singular")
    end
    
    for j = n:-1:2
        b[j] = b[j] / U[j,j]
        b[1:j-1] = b[1:j-1] - b[j] * U[1:j-1,j]
    end
    
    b[1] = b[1] / U[1,1]
    
    return b
end


end
