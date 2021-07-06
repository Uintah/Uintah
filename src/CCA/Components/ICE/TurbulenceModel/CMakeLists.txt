set( ICE_SRC ${ICE_SRC}
        ${CMAKE_CURRENT_SOURCE_DIR}/DynamicModel.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/SmagorinskyModel.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/Turbulence.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/TurbulenceFactory.cc
        PARENT_SCOPE
    )
set( ICE_INC ${ICE_INC} ${CMAKE_CURRENT_SOURCE_DIR} PARENT_SCOPE )
