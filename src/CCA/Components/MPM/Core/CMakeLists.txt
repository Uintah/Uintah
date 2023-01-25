add_library( CCA_Components_MPM_Core
        ImpMPMFlags.cc
        MPMBoundCond.cc
        MPMDiffusionLabel.cc
        MPMFlags.cc
        MPMLabel.cc
    )

target_link_libraries( CCA_Components_MPM_Core
        Core_Exceptions
        Core_Geometry
        Core_GeometryPiece
        Core_Grid
        Core_ProblemSpec
        Core_Util
    )

target_include_directories( CCA_Components_MPM_Core
    PUBLIC
        ${CMAKE_BINARY_DIR}/include
        ${CMAKE_SOURCE_DIR}
    )
