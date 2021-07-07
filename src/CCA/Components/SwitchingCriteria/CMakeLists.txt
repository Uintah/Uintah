# jcs add additional components here...

set( src
        None.cc
        SwitchingCriteriaFactory.cc
        SteadyState.cc
        TimestepNumber.cc
    )

if( ENABLE_ICE AND ENABLE_MPM )
    list( APPEND src
            SimpleBurn.cc
            SteadyState.cc
            DDT1.cc
        )
endif()

add_library( CCA_Components_SwitchingCriteria ${src} )

target_link_libraries( CCA_Components_SwitchingCriteria
    PUBLIC
        CCA_Ports
        Core_Grid
        Core_Parallel
        Core_ProblemSpec
    )

target_include_directories( CCA_Components_SwitchingCriteria
    PUBLIC
        ${CMAKE_SOURCE_DIR}
        ${CMAKE_BINARY_DIR}/include
    )