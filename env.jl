using Pkg
Pkg.activate("$(@__DIR__)")
Pkg.add(PackageSpec(name="ProbabilisticCircuits", version = "0.4.1"))
Pkg.update()
Pkg.build()
Pkg.precompile()