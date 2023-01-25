add_subdirectory( Models )

set( MPM_SRC ${MPM_SRC}
        ${CMAKE_CURRENT_SOURCE_DIR}/ArenaPartiallySaturated.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/WeibParameters.cc
        PARENT_SCOPE
    )
