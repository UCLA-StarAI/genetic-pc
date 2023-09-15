using CSV
export train_hclt


function run_em(pc, train_x, valid_x, test_x; 
    verbose=false, batch_size=512, pseudocount=0.005, softness=0,
    num_epochs1 = 100, num_epochs2 = 100, num_epochs3 = 20, num_epochs4 = 20,
    param_inertia1 = 0.0, param_inertia2 = 0.9, param_inertia3 = 0.95, 
    param_inertia4 = 0.999,
    logiter=5, patience=20, warmup=100, is_earlystop=true)
    println("Number of free parameters: $(num_parameters(pc))")

    verbose && println("1. Moving circuit to GPU...");flush(stdout)
    bpc = CuBitsProbCircuit(pc)
    verbose && report_ll(bpc, train_x, valid_x, test_x; batch_size)

    verbose && println("2. First minibatch...\n")
    callback = nothing
    if is_earlystop
        callback = EarlyStopPC(LikelihoodsLog(valid_x, test_x, logiter); patience, warmup)
    else
        callback = LikelihoodsLog(valid_x, test_x, logiter)
    end

    num_epochs1 > 0 && mini_batch_em(bpc, train_x, num_epochs1; batch_size, pseudocount, 
                softness, param_inertia = param_inertia1, param_inertia_end = param_inertia2,
                callbacks=Any[callback], verbose=true)
    verbose && report_ll(bpc, train_x, valid_x, test_x; batch_size)

    verbose && println("3. Second minibatch... ")
    num_epochs2 > 0 && mini_batch_em(bpc, train_x, num_epochs2; batch_size, pseudocount, 
                softness, param_inertia = param_inertia2, param_inertia_end = param_inertia3,
                callbacks=Any[callback], verbose=true)
    verbose && report_ll(bpc, train_x, valid_x, test_x; batch_size)

    verbose && println("4. Third minibatch... ")
    num_epochs3 > 0 && mini_batch_em(bpc, train_x, num_epochs3; batch_size, pseudocount, 
                softness, param_inertia = param_inertia3, param_inertia_end = param_inertia4,
                callbacks=Any[callback], verbose=true)
    verbose && report_ll(bpc, train_x, valid_x, test_x; batch_size)

    verbose && println("5. Full bacth... ")
    num_epochs4 > 0 && full_batch_em(bpc, train_x, num_epochs4; batch_size, pseudocount, softness,
                callbacks=Any[callback],  verbose=true)
    verbose && report_ll(bpc, train_x, valid_x, test_x; batch_size)

    verbose && println("5. Update parameters")
    ProbabilisticCircuits.update_parameters(bpc)
    bpc
end


function train_hclt(; datasetname, latents, structure="hclt",
    k_fold=nothing, nosplit=false,
    batch_size=512, pseudocount=0.005, softness=0,
    num_epochs1=100, num_epochs2=100, num_epochs3=20, num_epochs4=20,
    param_inertia1=0.2, param_inertia2=0.9, param_inertia3=0.95, param_inertia4=0.999,
    verbose=true, is_earlystop, logiter=5, patience=20, warmup=100, other...)

    println("Loading dataset $datasetname...")
    train_x, valid_x, test_x = data_gpu(datasetname; split=!nosplit, k_fold)
    data_summary(datasetname, train_x, valid_x, test_x)
    num_var = size(train_x, 2)

    if structure == "hclt"
        println("Initialize hclt...")
        pc = hclt(train_x, latents; pseudocount)
    else
        error("Structure $structure not found!")
    end
    init_parameters(pc; perturbation = 0.4)

    println("Running EM...")
    bpc = run_em(pc, train_x, valid_x, test_x; 
            verbose, batch_size, pseudocount, softness,
            num_epochs1, num_epochs2, num_epochs3, num_epochs4, 
            param_inertia1, param_inertia2, param_inertia3, param_inertia4, 
            logiter, patience, warmup, is_earlystop)
    evaluate_pc(pc, train_x, valid_x, test_x; latents, batch_size)
    pc, bpc, num_var
end


function run()
    seed = 1337
    Random.seed!(seed)
    CUDA.seed!(seed)
    train_hclt(; datasetname="805", structure="hclt", latents=4, 
        k_fold=1, nosplit=false, is_earlystop=true,
        batch_size=512, pseudocount=0.005, softness=0,
        num_epochs1=5, num_epochs2=5, num_epochs3=5, num_epochs4=5,
        param_inertia1=0.1, param_inertia2=0.9, param_inertia3=0.99, param_inertia4=0.999,
        verbose=true, logiter=5, patience=20, warmup=100)
end