set( ICE_SRC ${ICE_SRC}
        ${CMAKE_CURRENT_SOURCE_DIR}/logLawModel.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/smoothWall.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/WallShearStress.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/WallShearStressFactory.cc
        PARENT_SCOPE
    )
set( ICE_INC ${ICE_INC} ${CMAKE_CURRENT_SOURCE_DIR} PARENT_SCOPE )
