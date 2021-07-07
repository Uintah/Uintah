# each of these subdirectories should append to this list of source files (unless they build their own libs):
set( ICE_SRC
        AMRICE.cc
        ICE.cc
        impAMRICE.cc
        impICE.cc
        customInitialize.cc
    )

add_subdirectory( Advection )
add_subdirectory( Core )
add_subdirectory( CustomBCs )
##add_subdirectory( Docs )
add_subdirectory( EOS )
add_subdirectory( Materials )
add_subdirectory( PressureSolve )
add_subdirectory( SpecificHeatModel )
add_subdirectory( TurbulenceModel )
add_subdirectory( WallShearStressModel )

add_library( CCA_Components_ICE ${ICE_SRC} )
target_link_libraries( CCA_Components_ICE
    PUBLIC
        Core_GeometryPiece
        Core_Grid
        Core_Math
        Core_OS
        Core_ProblemSpec
        Core_Util
        CCA_Components_Application
        CCA_Components_Models
        CCA_Components_OnTheFlyAnalysis
        CCA_Components_Solvers
        ${HYPRE_LIBRARY}
        MPI::MPI_CXX
    )
target_include_directories( CCA_Components_ICE
    PRIVATE
        ${CMAKE_SOURCE_DIR}
        ${CMAKE_BINARY_DIR}/include
        ${HYPRE_INCLUDE}
    )
