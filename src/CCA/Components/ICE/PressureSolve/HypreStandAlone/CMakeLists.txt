set( ICE_SRC ${ICE_SRC}
        ${CMAKE_CURRENT_SOURCE_DIR}/Box.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/DebugStream.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/Error.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/Hierarchy.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/IntMatrix.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/Level.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/Param.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/Patch.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/Side.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/Solver.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/SolverAMG.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/SolverFAC.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/TestLinear.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/util.cc
        PARENT_SCOPE
    )
set( ICE_INC ${ICE_INC} ${CMAKE_CURRENT_SOURCE_DIR} PARENT_SCOPE )

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
target_link_libraries( mydriver ${HYPRE_LIB} MPI::MPI_CXX )
target_include_directories( mydriver PRIVATE ${HYPRE_INC} )

add_executable( mpitest mpitest.cc )
target_link_libraries( mpitest MPI::MPI_CXX )

add_executable( vectortest
        Box.cc
        DebugStream.cc
        Error.cc
        IntMatrix.cc
        Side.cc
        util.cc
        vectortest.cc
    )