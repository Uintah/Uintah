add_executable(
        mydriver
        Box.cc
        DebugStream.cc
        Error.cc
        Hierarchy.cc
        IntMatrix.cc
        Level.cc
        mydriver.cc
        Param.cc
        Patch.cc
        Side.cc
        Solver.cc
        SolverAMG.cc
        SolverFAC.cc
        TestLinear.cc
        util.cc
    )
target_link_libraries( mydriver ${HYPRE_LIB} UintahCore MPI::MPI_CXX )
target_include_directories( mydriver PRIVATE ${HYPRE_INC} ${CMAKE_CURRENT_SOURCE_DIR} )

add_executable( mpitest mpitest.cc )
target_link_libraries( mpitest Core_Parallel MPI::MPI_CXX )
target_include_directories( mpitest PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} )

add_executable( vectortest
        Box.cc
        DebugStream.cc
        Error.cc
        IntMatrix.cc
        Side.cc
        util.cc
        vectortest.cc
    )