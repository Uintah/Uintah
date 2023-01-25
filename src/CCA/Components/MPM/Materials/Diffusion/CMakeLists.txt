add_subdirectory( ConductivityModels )
add_subdirectory( DiffusionInterfaces )
add_subdirectory( DiffusionModels )

set( MPM_SRC ${MPM_SRC}
        ${CMAKE_CURRENT_SOURCE_DIR}/ScalarDiffusionModelFactory.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/SDInterfaceModelFactory.cc
        PARENT_SCOPE
    )
