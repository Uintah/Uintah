add_library( CCA_Components_PostProcessUda
        PostProcessUda.cc
        Module.cc
        ModuleFactory.cc
        statistics.cc
        spatioTemporalAvg.cc
    )

target_link_libraries( CCA_Components_PostProcessUda
        CCA_Components_Application
        CCA_Ports
        Core_DataArchive
        Core_Exceptions
        Core_Grid
        Core_OS
        Core_ProblemSpec
        Core_Util
    )

target_include_directories( CCA_Components_PostProcessUda
    PUBLIC
        ${CMAKE_SOURCE_DIR}
    )