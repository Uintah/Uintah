if( HAVE_PIDX )
    set( PIDX_SRC PIDXOutputContext.cc )
endif()

add_library( CCA_Ports
        ApplicationInterface.cc
        DataWarehouse.cc
        LoadBalancer.cc
        ModelInterface.cc
        Output.cc
        ProblemSpecInterface.cc
        Regridder.cc
        SFC.cc
        Scheduler.cc
        SolverInterface.cc
        SwitchingCriteria.cc
        ${PIDX_SRC}
    )

target_link_libraries( CCA_Ports
        Core_Geometry
        Core_Grid
        Core_ProblemSpec
        Core_Util
    )

target_include_directories( CCA_Ports
    PUBLIC
        ${CMAKE_SOURCE_DIR}
    )
