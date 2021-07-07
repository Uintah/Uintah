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

# Collect list of library dependencies
set( LIB_DEPS
        CCA_Ports
        Core_Grid
        Core_Exceptions
        Core_Math
        Core_OS
        Core_Parallel
        Core_Util
    )

if( ENABLE_ICE )
    list( APPEND OTF_SRC vorticity.cc )
    list( APPEND LIB_DEPS CCA_Components_ICE_Core )
endif()
if( ENABLE_MPM )
    list( APPEND OTF_SRC flatPlate_heatFlux.cc particleExtract.cc )
    list( APPEND LIB_DEPS CCA_Components_MPM_Core CCA_Components_MPM_Materials )
endif()
if( ENABLE_MPM_ICE )
    list( APPEND OTF_SRC 1stLawThermo.cc )
endif()

add_library( CCA_Components_OnTheFlyAnalysis ${OTF_SRC} )

target_link_libraries( CCA_Components_OnTheFlyAnalysis PUBLIC ${LIB_DEPS} )

target_include_directories( CCA_Components_OnTheFlyAnalysis
    PUBLIC
        ${CMAKE_SOURCE_DIR}
        ${CMAKE_BINARY_DIR}/include
    )
