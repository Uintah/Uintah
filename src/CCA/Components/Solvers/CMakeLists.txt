unset( OTHER_SRC )
if( HAVE_HYPRE )
    set( HYPRE_SRC HypreSolver.cc )
    set( HYPRE_LIBS HYPRE::HYPRE )
    add_subdirectory( AMR )  # collect more files into the HYPRE_SRC list for inclusion here
endif()

add_library( CCA_Components_Solvers
        CGSolver.cc
        SolverCommon.cc
        SolverFactory.cc
        ${HYPRE_SRC}
    )

target_link_libraries( CCA_Components_Solvers
    PUBLIC
        CCA_Ports
        Core_Exceptions
        Core_Grid
        Core_Math
        Core_Parallel
        Core_Util
        ${HYPRE_LIBS}
        ${LAPACK_LIBRARIES}
        ${BLAS_LIBRARIES}
    )
