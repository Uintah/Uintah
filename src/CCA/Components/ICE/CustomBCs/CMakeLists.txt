set( ICE_SRC ${ICE_SRC}
        ${CMAKE_CURRENT_SOURCE_DIR}/BoundaryCond.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/C_BC_driver.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/inletVelocity.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/LODI.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/LODI2.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/microSlipBCs.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/MMS_BCs.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/sine.cc
        PARENT_SCOPE
    )
set( ICE_INC ${ICE_INC}
        ${CMAKE_CURRENT_SOURCE_DIR}
        PARENT_SCOPE
    )