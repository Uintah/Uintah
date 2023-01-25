add_library( CCA_Components_Regridder
        RegridderCommon.cc
        RegridderFactory.cc
        TiledRegridder.cc
        SingleLevelRegridder.cc
    )

target_link_libraries( CCA_Components_Regridder
    PUBLIC
        CCA_Ports
        Core_Exceptions
        Core_Geometry
        Core_Grid
        Core_Parallel
        Core_Util
    )
target_include_directories( CCA_Components_Regridder
    PUBLIC
        ${CMAKE_SOURCE_DIR}
    )