add_library( CCA_Components_ICE_Core
        Diffusion.cc
        ICELabel.cc
    )

target_link_libraries( CCA_Components_ICE_Core
    PUBLIC
        Core_Exceptions
        Core_Grid
    )

target_include_directories( CCA_Components_ICE_Core
    PUBLIC
        ${CMAKE_SOURCE_DIR}
    )
