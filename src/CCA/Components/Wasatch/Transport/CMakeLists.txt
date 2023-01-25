set( SRC
        ${CMAKE_CURRENT_SOURCE_DIR}/CompressibleMomentumTransportEquation.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/EnthalpyTransportEquation.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/EquationBase.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/LowMachMomentumTransportEquation.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/MomentTransportEquation.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/MomentumTransportEquationBase.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/ParseEquation.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/ParseEquationHelper.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/ParseParticleEquations.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/ParticleEquationBase.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/ParticleMassEquation.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/ParticleMomentumEquation.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/ParticlePositionEquation.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/ParticleSizeEquation.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/ParticleTemperatureEquation.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/PersistentParticleICs.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/PreconditioningParser.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/ScalabilityTestTransportEquation.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/ScalarTransportEquation.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/TotalInternalEnergyTransportEquation.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/TransportEquation.cc
    )

if( WASATCH_ENABLE_POKITT )
    list( APPEND SRC
            ${CMAKE_CURRENT_SOURCE_DIR}/SetupCoalModels.cc
            ${CMAKE_CURRENT_SOURCE_DIR}/PseudospeciesTransportEquation.cc
            ${CMAKE_CURRENT_SOURCE_DIR}/SootParticleTransportEquation.cc
            ${CMAKE_CURRENT_SOURCE_DIR}/SootTransportEquation.cc
            ${CMAKE_CURRENT_SOURCE_DIR}/SpeciesTransportEquation.cc
            ${CMAKE_CURRENT_SOURCE_DIR}/SpeciesTransportEquation_diffusion.cc
            ${CMAKE_CURRENT_SOURCE_DIR}/SpeciesTransportEquation_reaction.cc
            ${CMAKE_CURRENT_SOURCE_DIR}/TarAndSootInfo.cc
            ${CMAKE_CURRENT_SOURCE_DIR}/TarTransportEquation.cc
            ${CMAKE_CURRENT_SOURCE_DIR}/ThermodynamicPressureTransportEquation.cc
        )
endif()

set( WASATCH_SRC ${WASATCH_SRC} ${SRC} PARENT_SCOPE )