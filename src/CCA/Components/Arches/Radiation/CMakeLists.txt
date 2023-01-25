set( SRC
    ${CMAKE_CURRENT_SOURCE_DIR}/RadPropertyCalculator.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/RadiationSolver.cc
    )

# jcs sort out PETSC
if( ENABLE_PETSC )
  list( APPEND SRC ${CMAKE_CURRENT_SOURCE_DIR}/RadPetscSolver.cc )
else()
  list( APPEND SRC ${CMAKE_CURRENT_SOURCE_DIR}/FakeRadPetscSolver.cc )
endif()

if( HAVE_HYPRE )
  list( APPEND SRC
      ${CMAKE_CURRENT_SOURCE_DIR}/DORadiationModel.cc
      ${CMAKE_CURRENT_SOURCE_DIR}/RadHypreSolver.cc
      )
endif()

# this populates F_SRC - the list of fortran files to advertise upward
add_subdirectory( fortran )

set( ARCHES_SRC ${ARCHES_SRC} ${SRC} ${F_SRC} PARENT_SCOPE )
