unset( OTHER_SRC )
if( HAVE_HYPRE )
    set( OTHER_SRC HypreSolver.cc )
    set( OTHER_LIBS CCA_Components_Solvers_AMR )
    add_subdirectory( AMR )
endif()

add_library( CCA_Components_Solvers
        CGSolver.cc
        SolverCommon.cc
        SolverFactory.cc
        ${OTHER_SRC}
    )

target_link_libraries( CCA_Components_Solvers
    PUBLIC
        CCA_Ports
        Core_Exceptions
        Core_Grid
        Core_Math
        Core_Parallel
        Core_Util
        ${OTHER_LIBS}
        ${LAPACK_LIBRARIES}
        ${BLAS_LIBRARIES}
    )
