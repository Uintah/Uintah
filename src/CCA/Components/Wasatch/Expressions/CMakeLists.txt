add_subdirectory( BoundaryConditions )
add_subdirectory( DensitySolvers )
add_subdirectory( EmbeddedGeometry )
add_subdirectory( MMS )
#add_subdirectory( Particles )
#add_subdirectory( PBE )
add_subdirectory( PostProcessing )
add_subdirectory( TarAndSoot )
add_subdirectory( Turbulence )

set( WASATCH_SRC ${WASATCH_SRC}
        ${CMAKE_CURRENT_SOURCE_DIR}/ActuatorDisk.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/BasicExprBuilder.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/CellType.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/ConvectiveFlux.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/Coordinate.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/DiffusiveFlux.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/DiffusiveVelocity.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/Dilatation.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/DORadSolver.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/ExprAlgebra.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/FanModel.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/MomentumPartialRHS.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/MomentumRHS.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/MonolithicRHS.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/PoissonExpression.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/Pressure.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/PressureSource.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/PrimVar.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/RadiationSource.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/RadPropsEvaluator.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/Reduction.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/ReductionBase.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/ScalabilityTestSrc.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/ScalarEOSCoupling.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/ScalarEOSHelper.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/ScalarRHS.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/SimpleEmission.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/SolnVarEst.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/StableTimestep.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/Strain.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/TabPropsHeatLossEvaluator.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/TargetValueSource.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/TestNestedExpression.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/TimeAdvance.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/CourantNumber.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/CellReynoldsNumber.cc                
        ${CMAKE_CURRENT_SOURCE_DIR}/StableTimestepForEq.cc                        
        PARENT_SCOPE
    )
