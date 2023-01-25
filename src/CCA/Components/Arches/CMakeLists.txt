# Arches requires Fortran-C interface, so ensure it works.
include( FortranCInterface )
FortranCInterface_VERIFY( CXX )

# require HYPRE
if( NOT HYPRE_FOUND )
  message( SEND_ERROR "HYPRE was not found.  Try setting one or more of the following:"
      "\n\tHYPRE_DIR - path to hyper installation"
      "\n\tHYPRE_INC - path to hypre header files"
      "\n\tHYPRE_LIB - path to hypre library files" )
endif()


#-------------------------------------------
# Collect all source files for the Arches component over this directory and subdirectories
set( ARCHES_SRC
    Arches.cc
    ArchesConstVariables.cc
    ArchesLabel.cc
    ArchesMaterial.cc
    ArchesParticlesHelper.cc
    ArchesVariables.cc
    BoundaryCond_new.cc
    BoundaryCondition.cc
    CQMOM.cc
    CellInformation.cc
    CellInformationP.cc
    CompDynamicProcedure.cc
    DQMOM.cc
    Discretization.cc
    ExplicitSolver.cc
    ExplicitTimeInt.cc
    GridTools.cc
    HandoffHelper.cc
    IntrusionBC.cc
    KokkosSolver.cc
    linSolver.cc
    LU.cc
    MomentumSolver.cc
    NonlinearSolver.cc
    PhysicalConstants.cc
    PressureSolverV2.cc
    Properties.cc
    RHSSolver.cc
    ScaleSimilarityModel.cc
    SmagorinskyModel.cc
    Source.cc
    TurbulenceModel.cc
    TurbulenceModelPlaceholder.cc
    UPSHelper.cc
    WBCHelper.cc
    )

if( ENABLE_PETSC )
  list( APPEND ARCHES_SRC Filter.cc PetscCommon.cc )
endif()

if( NOT ENABLE_WASATCH )
  list( APPEND ARCHES_SRC ${CMAKE_SOURCE_DIR}/CCA/Components/Wasatch/ParticlesHelper.cc )
endif()

add_subdirectory( BoundaryConditions )
add_subdirectory( ChemMix )
add_subdirectory( ChemMixV2 )
add_subdirectory( CoalModels )
add_subdirectory( DigitalFilter )
add_subdirectory( fortran )
add_subdirectory( LagrangianParticles )
add_subdirectory( ParticleModels )
add_subdirectory( PropertyModels )
add_subdirectory( PropertyModelsV2 )
add_subdirectory( Radiation )
add_subdirectory( SourceTerms )
add_subdirectory( SourceTermsV2 )
add_subdirectory( Task )
add_subdirectory( Transport )
add_subdirectory( TransportEqns)
add_subdirectory( TurbulenceModels )
add_subdirectory( Utility )
add_subdirectory( WallHTModels )

add_library( CCA_Components_Arches ${ARCHES_SRC} )

target_link_libraries( CCA_Components_Arches
    CCA_Components_Application
    CCA_Components_Models
    CCA_Ports
    Core_Exceptions
    Core_Geometry
    Core_GeometryPiece
    Core_Grid
    Core_Parallel
    Core_ProblemSpec
    Core_Util
    HYPRE::HYPRE
    )

target_include_directories( CCA_Components_Arches
    PUBLIC
      ${CMAKE_SOURCE_DIR}
      ${CMAKE_BINARY_DIR}
      ${CMAKE_BINARY_DIR}/include
    )
