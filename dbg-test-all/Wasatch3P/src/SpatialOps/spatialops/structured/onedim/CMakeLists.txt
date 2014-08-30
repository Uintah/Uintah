add_library( spatialops-onedim STATIC LagrangePoly.cpp )

if( ENABLE_TESTS )
  add_subdirectory( test )
endif( ENABLE_TESTS )

install( FILES LagrangePoly.h
         DESTINATION include/spatialops
         PERMISSIONS OWNER_READ GROUP_READ WORLD_READ
        )

install( TARGETS spatialops-onedim
         EXPORT SpatialOps
         DESTINATION lib/spatialops/
         PERMISSIONS OWNER_READ GROUP_READ WORLD_READ
        )
