set( MPM_SRC ${MPM_SRC}
        ${CMAKE_CURRENT_SOURCE_DIR}/ElasticModuli_Arena.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/ElasticModuliModel.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/ElasticModuliModelFactory.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/ModelState_Arena.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/ModelStateBase.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/Pressure_Air.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/Pressure_Granite.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/Pressure_Water.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/PressureModel.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/YieldCond_Arena.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/YieldCondition.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/YieldConditionFactory.cc
        PARENT_SCOPE
    )
