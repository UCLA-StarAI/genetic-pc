using CSV
using CUDA
using DataFrames
using MLBase
using Random
export data_gpu, data_cpu, transform_vcf_to_data, k_fold_cv, generate_artificial_genomes

BIO_DATASETS = ["805", "10K", "mhc", "chr5.sub"]

function load(name, type; data_dir, k_fold=nothing)
    filename = data_dir*"/$(name)_k=$k_fold.$type.data"
    dataframe = CSV.read(filename, DataFrame; header=false, 
        truestrings=["1", "2", "3"], falsestrings=["0"], types=Bool, strict=true)
    BitArray(Tables.matrix(dataframe))
end


function data_gpu(name; kwargs...)
    cu.(data_cpu(name; kwargs...))
end


function data_cpu(name; split=true, data_dir="./data1kg", k_fold)
    if name in BIO_DATASETS
        train = load(name, "train"; data_dir, k_fold)
        valid = load(name, "valid"; data_dir, k_fold)
        test = load(name, "test"; data_dir, k_fold)
        if !split
            train = [train; valid]
            valid = nothing
        end
        return train, valid, test
    end
end


function transform_vcf_to_data(name; data_dir)
    dataframe = CSV.read(data_dir*"/$name.vcf", DataFrame; header=false, delim="\t", drop=1:9)
    CSV.write(data_dir*"/$name.data.tmp", dataframe; header=false)
    file = data_dir*"/$name.data.tmp"
    cmd = `vim $file -c "%s/|/,/gi| wq"`
    run(cmd)
    dataframe = CSV.read(data_dir*"/$name.data.tmp", DataFrame; header=false, delim=",", transpose=true)
    CSV.write(data_dir*"/$name.data", dataframe; header=false)
    run(`rm $file`)
end


function k_fold_cv(name; k=5, seed=1337, valid_percent=0.1, data_dir="./")
    dataframe = CSV.read(data_dir*"/$name.data", DataFrame; header=false)
    n = size(dataframe)[1]
    Random.seed!(seed)
    datasets = []
    for (run_k, id) in enumerate(collect(Kfold(n, k)))
        is_train = falses(n)
        is_train[id] .= true
        is_test = .~is_train
        is_valid = rand(n) .< valid_percent
        is_train, is_valid = is_train .& .!is_valid, is_train .& is_valid
        @assert sum(is_test .& is_train .& is_valid) == 0
        @assert sum(is_test .| is_train .| is_valid) == n
        push!(datasets, (dataframe[is_train, :], dataframe[is_valid, :], dataframe[is_test, :]))
        CSV.write(data_dir*"/$(name)_k=$run_k.train.data", dataframe[is_train, :]; header=false, delim=" ")
        CSV.write(data_dir*"/$(name)_k=$run_k.valid.data", dataframe[is_valid, :]; header=false, delim=" ")
        CSV.write(data_dir*"/$(name)_k=$run_k.test.data", dataframe[is_test, :]; header=false, delim=" ")
    end
    datasets
end


function generate_artificial_genomes(bpc, nsample, num_var; seed=1337)
    Random.seed!(seed)

    # DataFrame
    cols = [:Type, :ID]
    col_types = [String, String]
    for i in 1:num_var
        push!(cols, Symbol("Column$i"))
        push!(col_types, Int64)
    end
    named_tuple = (; zip(cols, type[] for type in col_types )...)
    df = DataFrame(named_tuple)

    # sample
    states = ProbabilisticCircuits.sample(bpc, nsample, num_var, [Float64])
    samples = dropdims(to_cpu(states), dims=2)
    for i in 1:nsample
        push!(df, [["AG", "AG$i"]; samples[i, :]])
    end
    df
end