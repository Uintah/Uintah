if( HAVE_HYPRE )
    # collect more files into HYPRE_SRC list
    add_subdirectory( HyprePreconds )
    add_subdirectory( HypreSolvers )

    set( HYPRE_SRC ${HYPRE_SRC}
        ${CMAKE_CURRENT_SOURCE_DIR}/AMRSolver.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/HypreDriver.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/HypreDriverStruct.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/HypreDriverSStruct.cc
        PARENT_SCOPE
        )
endif()
