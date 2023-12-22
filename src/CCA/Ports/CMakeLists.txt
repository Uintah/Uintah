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
        "$<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}>"
        "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>"
    )

install( FILES ApplicationInterface.h
    DataWarehouse.h
    DataWarehouseP.h
    InputContext.h
    LoadBalancer.h
    ModelInterface.h
    Output.h
    OutputContext.h
    PIDXOutputContext.h
    ProblemSpecInterface.h
    Regridder.h
    Scheduler.h
    SchedulerP.h
    SFC.h
    SolverInterface.h
    SwitchingCriteria.h
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/CCA/Ports
)

install( TARGETS CCA_Ports
    EXPORT uintah_lib_targets
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_LIBDIR}
)
