nebo_cuda_prep_dir()

set( src
     StencilBuilder.cpp
    )

nebo_add_library( spatialops-stencil ${src} )

if( ENABLE_TESTS )
  add_subdirectory( test )
endif( ENABLE_TESTS )

install( FILES
  FVStaggeredBCOp.h
  StencilBuilder.h
  FVStaggeredOperatorTypes.h
  DESTINATION include/spatialops/structured/stencil/
  PERMISSIONS OWNER_READ GROUP_READ WORLD_READ
  )

install( TARGETS spatialops-stencil
         EXPORT SpatialOps
         DESTINATION lib/spatialops/
         PERMISSIONS OWNER_READ GROUP_READ WORLD_READ
        )
