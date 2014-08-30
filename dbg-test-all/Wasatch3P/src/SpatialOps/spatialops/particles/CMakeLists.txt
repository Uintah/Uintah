if( ENABLE_TESTS )
  add_subdirectory( test )
endif( ENABLE_TESTS )

install( FILES
  ParticleFieldTypes.h
  ParticleOperators.h
  DESTINATION include/spatialops/particles
  PERMISSIONS OWNER_READ GROUP_READ WORLD_READ
  )
