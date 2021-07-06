set( ICE_SRC ${ICE_SRC}
        ${CMAKE_CURRENT_SOURCE_DIR}/Component.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/Debye.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/NASAPolynomial.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/Polynomial.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/SpecificHeat.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/SpecificHeatFactory.cc
        PARENT_SCOPE
    )
set( ICE_INC ${ICE_INC} ${CMAKE_CURRENT_SOURCE_DIR} PARENT_SCOPE )
