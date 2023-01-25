add_library( CCA_Components_LoadBalancers
        CostModelForecaster.cc
        CostProfiler.cc
        DynamicLoadBalancer.cc
        LoadBalancerCommon.cc
        LoadBalancerFactory.cc
        ParticleLoadBalancer.cc
        ProfileDriver.cc
        RoundRobinLoadBalancer.cc
        SimpleLoadBalancer.cc
    )

target_link_libraries( CCA_Components_LoadBalancers
        CCA_Components_ProblemSpecification
        CCA_Components_Schedulers
        CCA_Ports

        Core_DataArchive
        Core_Exceptions
        Core_Grid
        Core_Parallel
        Core_Util
    )

target_include_directories( CCA_Components_LoadBalancers
    PUBLIC
        ${CMAKE_SOURCE_DIR}
    )
