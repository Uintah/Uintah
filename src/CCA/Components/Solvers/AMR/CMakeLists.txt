if( HAVE_HYPRE )

    add_library( CCA_Components_Solvers_AMR
        AMRSolver.cc
        HypreDriver.cc
        HypreDriverStruct.cc
        HypreDriverSStruct.cc
    )

    target_link_libraries( CCA_Components_Solvers_AMR
        PUBLIC
            UintahCore
            HYPRE::HYPRE
        )

    add_subdirectory( HyprePreconds )
    add_subdirectory( HypreSolvers )

endif()
