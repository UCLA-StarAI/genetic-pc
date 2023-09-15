using ArgParse
using TrainPCGenomes
using Random

if abspath(PROGRAM_FILE) == @__FILE__
    s = ArgParseSettings()
    @add_arg_table s begin
        "--datasetsname"
            help = "Dataset name"
            default = nothing
        "--data_dir"
            help = "Data directory"
            arg_type = String
            default = "./data1kg"
        "--num_k"
            help = "Number of k-fold cross validation"
            arg_type = Int64
            default = 5
        "--valid_percent"
            help = "Percentage of validation set splitted out from training set"
            arg_type = Float64
            default = 0.1
        "--seed"
            help = "Random seed"
            arg_type = Int64
            default = 1337
    end

    args = parse_args(ARGS, s)
    println(args)

    Random.seed!(args["seed"])

    # for name in ["mhc", "chr5.sub"]
    #     println("Transforming dataset $name from vcf")
    #     transform_vcf_to_data(name; 
    #         data_dir=args["data_dir"]);
    # end

    for name in ["805", "10K"] #, "mhc", "chr5.sub"]
        println("Spliting dataset $name")
        k_fold_cv(name; 
            k=args["num_k"], 
            seed=args["seed"], 
            valid_percent=args["valid_percent"], 
            data_dir=args["data_dir"])
    end
end