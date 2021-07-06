set( ICE_SRC ${ICE_SRC}
        ${CMAKE_CURRENT_SOURCE_DIR}/BirchMurnaghan.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/EquationOfState.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/EquationOfStateFactory.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/Gruneisen.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/HardSphereGas.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/IdealGas.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/JWL.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/JWLC.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/KnaussSeaWater.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/KumariDass.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/Murnaghan.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/Thomsen_Hartka_water.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/Tillotson.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/TST.cc
        PARENT_SCOPE
    )

set( ICE_INC ${ICE_INC} ${CMAKE_CURRENT_SOURCE_DIR} PARENT_SCOPE )
