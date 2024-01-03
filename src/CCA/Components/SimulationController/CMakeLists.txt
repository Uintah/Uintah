add_library( CCA_Components_SimulationController
        SimulationController.cc
        AMRSimulationController.cc
    )

if( HAVE_VISIT )
    set( VISIT_INTERFACE VisIt_libsim )
endif()

target_link_libraries( CCA_Components_SimulationController
    PUBLIC
        CCA_Components_DataArchiver
        CCA_Components_PostProcessUda
        CCA_Components_Regridder
        CCA_Components_Schedulers
        CCA_Ports
        Core_DataArchive
        Core_Exceptions
        Core_Grid
        Core_OS
        Core_Parallel
        Core_Util
        ${GPERFTOOLS_LIBRARIES} # for gperftools if active
        ${VISIT_INTERFACE}
    )

target_include_directories( CCA_Components_SimulationController
    PUBLIC
        ${CMAKE_SOURCE_DIR}}
        ${GPERFTOOLS_INCLUDE_DIRS}  # for gperftools if active
    )
