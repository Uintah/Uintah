set ( OTF_SRC
        )
add_library( CCA_Components_OnTheFlyAnalysis
        AnalysisModule.cc
        AnalysisModuleFactory.cc
        FileInfoVar.cc
        MinMax.cc
        controlVolFluxes.cc
        controlVolume.cc
        lineExtract.cc
        meanTurbFluxes.cc
        momentumAnalysis.cc
        planeAverage.cc
        planeExtract.cc
        statistics.cc
    )

if( ENABLE_ICE )
    set( OTF_SRC ${OTF_SRC}
            vorticity.cc
        )
endif()
if( ENABLE_MPM )
    set( OTF_SRC ${OTF_SRC}
            flatPlate_heatFlux.cc
            particleExtract.cc
        )
endif()
if( ENABLE_MPM_ICE )
    set( OTF_SRC ${OTF_SRC}
            1stLawThermo.cc
        )
endif()


target_link_libraries( CCA_Components_OnTheFlyAnalysis
        CCA_Ports
        Core_Grid
        Core_Exceptions
        Core_Math
        Core_OS
        Core_Parallel
        Core_Util
    )
