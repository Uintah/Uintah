include_directories( ${CMAKE_BINARY_DIR}/tmp ) # for expression/ExprLib.h

add_executable( graph graph.cpp )
target_link_libraries( graph exprlib )
add_test( graph_test graph )

if( NOT ENABLE_UINTAH AND NOT ENABLE_LBMS )

  add_subdirectory( FieldMgr )
  add_subdirectory( SrcTest )
  add_subdirectory( VecIntegrate )
  add_subdirectory( bcs )
  add_subdirectory( function )
  add_subdirectory( heatFluxTest )
  add_subdirectory( integrator )
  add_subdirectory( ModifierExpr )
  add_subdirectory( multFieldExpr )
  add_subdirectory( rxnmech )
  add_subdirectory( simpleHeatEqn )
  add_subdirectory( treeCleave )
  add_subdirectory( Poller )
  if( ENABLE_OUTPUT )
    add_subdirectory( FieldWriter )
  endif( ENABLE_OUTPUT )

  add_subdirectory( threadIntegrator )
  add_subdirectory( threadTest )

endif()

if( ENABLE_UINTAH )
  add_subdirectory( UintahTest )
endif()
