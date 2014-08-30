if( ENABLE_TESTS )
  add_subdirectory( test )
endif( ENABLE_TESTS )

install( FILES
  PointFieldTypes.h
  PointOperators.h
  DESTINATION include/spatialops/pointfield
  PERMISSIONS OWNER_READ GROUP_READ WORLD_READ
  )