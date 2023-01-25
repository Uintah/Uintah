set( WASATCH_SRC ${WASATCH_SRC}
        ${CMAKE_CURRENT_SOURCE_DIR}/DynamicSmagorinskyCoefficient.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/StrainTensorBase.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/StrainTensorMagnitude.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/TurbulenceParameters.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/TurbulentDiffusivity.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/TurbulentViscosity.cc
        PARENT_SCOPE
    )
