add_library( CCA_Components_ICE_Materials ICEMaterial.cc )

target_link_libraries( CCA_Components_ICE_Materials
        CCA_Components_ICE_SpecificHeatModel
        CCA_Components_ICE_EOS
        Core_Exceptions
        Core_Geometry
        Core_GeometryPiece
        Core_Grid
        Core_Util
    )

target_include_directories( CCA_Components_ICE_Materials
    PUBLIC
        ${CMAKE_SOURCE_DIR}
        ${CMAKE_BINARY_DIR}/include
    )