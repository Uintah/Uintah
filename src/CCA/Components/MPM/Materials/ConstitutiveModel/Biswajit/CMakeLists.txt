add_subdirectory( Models )

set( MPM_SRC ${MPM_SRC}
        ${CMAKE_CURRENT_SOURCE_DIR}/Arena.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/CamClay.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/HyperelasticPlastic.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/SmallStrainPlastic.cc
        PARENT_SCOPE
    )
