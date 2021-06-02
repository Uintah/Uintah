add_library( CCA_Components_Solvers_AMR_HyperPreconds
        HyprePrecondBase.cc
        HyprePrecondSMG.cc
        HyprePrecondPFMG.cc
        HyprePrecondSparseMSG.cc
        HyprePrecondJacobi.cc
        HyprePrecondDiagonal.cc
    )

target_link_libraries( CCA_Components_Solvers_AMR_HyperPreconds
    PUBLIC
        HYPRE::HYPRE
    )
