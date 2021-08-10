if( HAVE_HYPRE )

    set( HYPRE_SRC
        AMRSolver.cc
        HypreDriver.cc
        HypreDriverStruct.cc
        HypreDriverSStruct.cc
        )

    # collect more files into HYPRE_SRC list
    add_subdirectory( HyprePreconds )
    add_subdirectory( HypreSolvers )

    add_library( CCA_Components_Solvers_AMR ${HYPRE_SRC} )

    target_link_libraries( CCA_Components_Solvers_AMR
        PUBLIC
            UintahCore
            HYPRE::HYPRE
        )

endif()
