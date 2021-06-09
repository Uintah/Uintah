set ( OTF_SRC
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
    list( APPEND OTF_SRC vorticity.cc )
endif()
if( ENABLE_MPM )
    list( APPEND OTF_SRC flatPlate_heatFlux.cc particleExtract.cc )
endif()
if( ENABLE_MPM_ICE )
    list( APPEND OTF_SRC 1stLawThermo.cc )
endif()

add_library( CCA_Components_OnTheFlyAnalysis ${OTF_SRC} )

target_link_libraries( CCA_Components_OnTheFlyAnalysis
    PUBLIC
        CCA_Ports
        Core_Grid
        Core_Exceptions
        Core_Math
        Core_OS
        Core_Parallel
        Core_Util
    )
