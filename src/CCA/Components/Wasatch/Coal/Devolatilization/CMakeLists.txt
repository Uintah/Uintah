add_subdirectory( CPD )
add_subdirectory( KobayashiSarofim )
add_subdirectory( SingleRate )

set( WASATCH_SRC ${WASATCH_SRC}
        ${CMAKE_CURRENT_SOURCE_DIR}/DevolatilizationBase.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/DevolatilizationInterface.cc
        PARENT_SCOPE
    )
