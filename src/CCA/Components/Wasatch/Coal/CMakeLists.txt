add_subdirectory( CharOxidation )
add_subdirectory( Devolatilization )
add_subdirectory( VaporizationBoiling )

set( WASATCH_SRC ${WASATCH_SRC}
        ${CMAKE_CURRENT_SOURCE_DIR}/CoalData.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/CoalEquation.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/CoalInterface.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/SpeciesData.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/StringNames.cc
        PARENT_SCOPE
    )
