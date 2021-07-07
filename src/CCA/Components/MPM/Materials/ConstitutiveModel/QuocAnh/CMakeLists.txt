set( MPM_SRC ${MPM_SRC}
        ${CMAKE_CURRENT_SOURCE_DIR}/BBMMatrix.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/BBMPoint.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/ClassicMohrCoulomb.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/HypoplasticB.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/MohrCoulomb.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/QADamage.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/ShengMohrCoulomb.cc
        PARENT_SCOPE
    )
