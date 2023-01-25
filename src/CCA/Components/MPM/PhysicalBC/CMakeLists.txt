add_library( CCA_Components_MPM_PhysicalBC
        ArchesHeatFluxBC.cc
        AutoCycleFluxBC.cc
        FluxBCModel.cc
        FluxBCModelFactory.cc
        ForceBC.cc
        HeatFluxBC.cc
        MPMPhysicalBCFactory.cc
        NormalForceBC.cc
        PolynomialData.cc
        PressureBC.cc
        ScalarFluxBC.cc
    )

target_link_libraries( CCA_Components_MPM_PhysicalBC
        CCA_Components_MPM_Core
    )

target_include_directories( CCA_Components_MPM_PhysicalBC
    PUBLIC
        ${CMAKE_BINARY_DIR}/include
        ${CMAKE_SOURCE_DIR}
    )
