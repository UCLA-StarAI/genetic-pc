export data_summary, evaluate_pc, report_ll

function data_summary(name, datas...)
    println("Dataset $(name) summary: ")
    num_cats = maximum(datas[1]) + 1
    println(" - Number of variables: $(size(datas[1], 2))")
    for data in datas
        if !isnothing(data)
            println(" - Number of examples: $(size(data, 1))")
        end
    end
    println(" - Number of categories: $(num_cats)")
end

function evaluate_pc(pc, train_x, valid_x, test_x; latents=nothing, batch_size=512)
    bpc = CuBitsProbCircuit(pc)
    println("# Latents: $latents")
    println("# parameters: $(num_parameters(pc))")
    sum_p = (sum(n -> num_parameters_node(n, true), sumnodes(pc)))
    println("# sum parameters: $sum_p")

    ll1, ll2, ll3 = nothing, nothing, nothing
    ll1 = loglikelihood(bpc, train_x; batch_size)
    println("train ll: $ll1")
    if valid_x!== nothing
        ll2 = loglikelihood(bpc, valid_x; batch_size)
        println("valid ll: $ll2")
    end
    if test_x !== nothing
        ll3 = loglikelihood(bpc, test_x; batch_size)
        println("test ll: $ll3")
    end

    return ll1, ll2, ll3
end

function report_ll(bpc, train_x, valid_x, test_x; batch_size) 
    for (str, data) in zip(["train", "valid", "test"], [train_x, valid_x, test_x])
        if !isnothing(data)
            ll = loglikelihood(bpc, data; batch_size)
            println("  $str LL: $(ll), $(bits_per_dim(ll, data))")
        else
            println("  $str LL: nothing")
        end
    end
end

bits_per_dim(ll, data) = begin
    -(ll  / size(data, 2)) / log(2)
end
