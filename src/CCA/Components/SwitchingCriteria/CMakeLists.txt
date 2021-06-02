# jcs add additional components here...

add_library( CCA_Components_SwitchingCriteria
        None.cc
        SwitchingCriteriaFactory.cc
        SteadyState.cc
        TimestepNumber.cc
    )

target_link_libraries( CCA_Components_SwitchingCriteria
    PUBLIC
        CCA_Ports
        Core_Grid
        Core_Parallel
        Core_ProblemSpec
    )
