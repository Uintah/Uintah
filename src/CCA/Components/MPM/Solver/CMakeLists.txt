set( MPM_SRC ${MPM_SRC}
        ${CMAKE_CURRENT_SOURCE_DIR}/FakePetscSolver.cc
#        ${CMAKE_CURRENT_SOURCE_DIR}/PetscSolver.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/SimpleSolver.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/Solver.cc
        PARENT_SCOPE
    )
