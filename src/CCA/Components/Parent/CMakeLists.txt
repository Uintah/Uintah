add_library( CCA_Components_Parent
        Switcher.cc
        ApplicationFactory.cc
    )

# jcs tons of extra stuff if components are active...???

target_link_libraries( CCA_Components_Parent
        Component_Libs
        CCA_Components_PostProcessUda
        CCA_Components_Solvers
        CCA_Components_SwitchingCriteria
        Core_Exceptions
        Core_GeometryPiece
        Core_Parallel
    )

target_include_directories( CCA_Components_Parent
    PUBLIC
        ${CMAKE_SOURCE_DIR}
    )
