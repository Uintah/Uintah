add_subdirectory( Core )

add_library( CCA_Components_MPMICE MPMICE.cc )
target_link_libraries( CCA_Components_MPMICE
        CCA_Components_Application
        CCA_Components_MPM
        CCA_Components_MPM_Core
        CCA_Components_MPM_Materials
        CCA_Components_ICE
        CCA_Components_ICE_EOS
        CCA_Components_ICE_Materials
        CCA_Components_OnTheFlyAnalysis
        CCA_Ports
        Core_Exceptions
        Core_Geometry
        Core_Grid
        Core_Math
        Core_ProblemSpec
        Core_Util
    )
target_include_directories( CCA_Components_MPMICE
    PUBLIC
        ${CMAKE_SOURCE_DIR}
        ${CMAKE_BINARY_DIR}/include
    )