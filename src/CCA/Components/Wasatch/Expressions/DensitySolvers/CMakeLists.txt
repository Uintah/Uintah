add_subdirectory( SpeciesAndEnthalpyExpressions )

set( SRC
        ${CMAKE_CURRENT_SOURCE_DIR}/DensityFromMixFrac.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/DensityFromMixFracAndHeatLoss.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/TwoStreamMixingDensity.cc
    )
if( WASATCH_ENABLE_POKITT )
    list( APPEND SRC
            ${CMAKE_CURRENT_SOURCE_DIR}/DensityFromSpeciesAndEnthalpy.cc
        )
endif()

set( WASATCH_SRC ${WASATCH_SRC} ${SRC} PARENT_SCOPE )