add_library( CCA_Components_Parent
        Switcher.cc
        ApplicationFactory.cc
    )

target_link_libraries( CCA_Components_Parent
        Component_Libs
        Core_Exceptions
        Core_GeometryPiece
        Core_Parallel
    )

target_include_directories( CCA_Components_Parent
    PUBLIC
        ${CMAKE_SOURCE_DIR}
    )
