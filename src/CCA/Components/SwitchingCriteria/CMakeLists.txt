set( src
        None.cc
        SwitchingCriteriaFactory.cc
        SteadyState.cc
        TimestepNumber.cc
    )
set( LIB_DEPS
        CCA_Ports
        Core_Grid
        Core_Parallel
        Core_ProblemSpec
    )

if( ENABLE_MPM_ICE )
    list( APPEND src
            SimpleBurn.cc
            SteadyBurn.cc
            DDT1.cc
        )
    list( APPEND LIB_DEPS
            CCA_Components_ICE_Core
            CCA_Components_MPM_Core
            CCA_Components_MPMICE_Core
        )
endif()

add_library( CCA_Components_SwitchingCriteria ${src} )

target_link_libraries( CCA_Components_SwitchingCriteria ${LIB_DEPS} )

target_include_directories( CCA_Components_SwitchingCriteria
    PUBLIC
        ${CMAKE_SOURCE_DIR}
        ${CMAKE_BINARY_DIR}/include
    )