using LinearAlgebra, NLPModels, SolverTools

# Input of an NLPModels compliant solver is:
function sqp(nlp :: AbstractNLPModel; # Only mandatory argument
             max_eval = 100,
             max_time = 3.0,
             atol = 1e-6,
             rtol = 1e-6,
             )

    start_time = time()

    nvar, ncon = nlp.meta.nvar, nlp.meta.ncon

    x = copy(nlp.meta.x0)
    y = ones(ncon)

    f(x) = obj(nlp, x)
    ∇f(x) = grad(nlp, x)
    H(x,y) = hess(nlp, x, y)
    c(x) = cons(nlp, x)
    J(x) = jac(nlp, x)

    Jx = J(x)
    dual = ∇f(x) + Jx' * y
    primal = c(x)

    ϵd = atol + rtol * norm(dual)
    ϵp = atol

    Δt = time() - start_time
    solved = norm(dual) < ϵd && norm(primal) < ϵp
    tired  = Δt > max_time || sum_counters(nlp) > max_eval
    iter = 0

    @info log_header([:iter, :fx, :dual, :primal],
                     [Int, Float64, Float64, Float64],
                     hdr_override=Dict(:fx => "f(x)",
                                       :dual => "‖∇ℓ(x,y)‖",
                                       :primal => "‖c(x)‖"))

    @info log_row(Any[iter, f(x), norm(dual), norm(primal)])

    while !(solved || tired)
        Hxy = H(x, y)
        W = [Hxy zeros(nvar, ncon); Jx zeros(ncon, ncon)]
        Δxy = -Symmetric(W, :L) \ [dual; primal]
        Δx = Δxy[1:nvar]
        Δy = Δxy[nvar+1:end]

        x += Δx
        y += Δy

        Jx = J(x)
        dual = ∇f(x) + Jx' * y
        primal = c(x)
        Δt = time() - start_time
        solved = norm(dual) < ϵd && norm(primal) < ϵp
        tired  = Δt > max_time || sum_counters(nlp) > max_eval
        iter += 1

        @info log_row(Any[iter, f(x), norm(dual), norm(primal)])
    end

    status = if solved
        :first_order
    elseif tired
        if Δt > max_time
            :max_time
        else
            :max_eval
        end
    else
        :unknown
    end

    return GenericExecutionStats(status, nlp;
                solution=x, objective=f(x), dual_feas=norm(dual),
                primal_feas=norm(primal), elapsed_time=Δt, iter=iter,
                solver_specific=Dict(:multiplers => y)) # should change to multipliers
end
