using CUTEst, NLPModelsIpopt, Plots, SolverBenchmark
gr(size=(1000, 800))

pnames = CUTEst.select(max_var=10, min_con=1, max_con=10,
                           only_free_var=true, only_equ_con=true)
problems = (CUTEstModel(p) for p in pnames)
solvers = Dict(:SQP => sqp,
                   :IPOPT => (nlp; kwargs...) -> ipopt(nlp, print_level=0; kwargs...))

stats = bmark_solvers(solvers, problems)
cost(df) = df.elapsed_time
performance_profile(stats, cost)
png("perf-prof")
