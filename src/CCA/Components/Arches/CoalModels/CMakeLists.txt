add_subdirectory( fortran )

set( ARCHES_SRC ${ARCHES_SRC}
    ${F_FILES}
    ${CMAKE_CURRENT_SOURCE_DIR}/BirthDeath.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/CharOxidation.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/CharOxidationShaddix.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/CharOxidationSmith.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/CharOxidationSmithConstLv0.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/CoalModelFactory.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/ConstantModel.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/Deposition.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/Devolatilization.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/DragModel.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/EnthalpyShaddix.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/FOWYDevol.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/HeatTransfer.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/KobayashiSarofimDevol.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/LinearSwelling.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/MaximumTemperature.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/ModelBase.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/PartVel.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/RichardsFletcherDevol.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/ShrinkageRate.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/Thermophoresis.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/YamamotoDevol.cc
    PARENT_SCOPE
    )
