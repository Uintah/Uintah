
set( WASATCH_SRC ${WASATCH_SRC}
        ${CMAKE_CURRENT_SOURCE_DIR}/BoundaryConditionsConstant.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/BoundaryConditionsCopier.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/BoundaryConditionsGaussian.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/BoundaryConditionsLinear.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/BoundaryConditionsOneSided.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/BoundaryConditionsParabolic.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/BoundaryConditionsPowerLaw.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/BoundaryConditionsPrimvar.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/OpenBC.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/OutflowBC.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/ParticleWallBC.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/VardenMMSBCs.cc
        PARENT_SCOPE
    )
