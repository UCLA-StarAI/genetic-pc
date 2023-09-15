module TrainPCGenomes

using LogicCircuits
using ProbabilisticCircuits
using ProbabilisticCircuits: CuBitsProbCircuit, loglikelihood, full_batch_em, 
mini_batch_em, update_parameters, EarlyStopPC, LikelihoodsLog, init_parameters

include("utils.jl")
include("data.jl")
include("learn.jl")


end # module TrainPCGenomes
