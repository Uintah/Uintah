add_library( CCA_Components_Solvers_AMR_HypreSolvers
        HypreSolverBase.cc
        HypreSolverPFMG.cc
        HypreSolverSMG.cc
        HypreSolverSparseMSG.cc
        HypreSolverCG.cc
        HypreSolverHybrid.cc
        HypreSolverGMRES.cc
        HypreSolverAMG.cc
        HypreSolverFAC.cc
    )

target_link_libraries( CCA_Components_Solvers_AMR_HypreSolvers
    PUBLIC
        HYPRE::HYPRE
    )