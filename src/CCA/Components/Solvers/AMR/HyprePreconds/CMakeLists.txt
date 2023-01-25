set( HYPRE_SRC ${HYPRE_SRC}
    ${CMAKE_CURRENT_SOURCE_DIR}/HyprePrecondBase.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/HyprePrecondSMG.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/HyprePrecondPFMG.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/HyprePrecondSparseMSG.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/HyprePrecondJacobi.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/HyprePrecondDiagonal.cc
    PARENT_SCOPE
    )
