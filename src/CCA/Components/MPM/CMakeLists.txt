# each of these subdirectories should append to this list of source files:
set( MPM_SRC
        AMRMPM.cc
        ImpMPM.cc
        MPMCommon.cc
        RigidMPM.cc
        SerialMPM.cc
        ShellMPM.cc
        SingleFieldMPM.cc
    )

add_subdirectory( CohesiveZone )
add_subdirectory( Core )
add_subdirectory( HeatConduction )
add_subdirectory( Materials )
add_subdirectory( MMS )
add_subdirectory( PhysicalBC )
add_subdirectory( Solver )
add_subdirectory( ThermalContact )

add_library( CCA_Components_MPM ${MPM_SRC} )

target_link_libraries( CCA_Components_MPM
        Core_Exceptions
        Core_Geometry
        Core_GeometryPiece
        Core_Grid
        Core_Math
        Core_Parallel
        Core_ProblemSpec
        Core_Util
        CCA_Components_Application
        CCA_Components_OnTheFlyAnalysis
        CCA_Ports
    )

target_include_directories( CCA_Components_MPM
    PUBLIC
        ${CMAKE_SOURCE_DIR}
        ${CMAKE_BINARY_DIR}/include
    )