add_library( CCA_Components_ICE_CustomBCs
        BoundaryCond.cc
        C_BC_driver.cc
        inletVelocity.cc
        LODI2.cc
        microSlipBCs.cc
        MMS_BCs.cc
        sine.cc
    )

target_link_libraries( CCA_Components_ICE_CustomBCs
        CCA_Components_ICE_Core
        CCA_Components_ICE_EOS
        CCA_Components_ICE_Materials
        Core_Exceptions
        Core_Geometry
        Core_GeometryPiece
        Core_ProblemSpec
    )

target_include_directories( CCA_Components_ICE_CustomBCs
    PUBLIC
        ${CMAKE_SOURCE_DIR}
        ${CMAKE_BINARY_DIR}/include
    )