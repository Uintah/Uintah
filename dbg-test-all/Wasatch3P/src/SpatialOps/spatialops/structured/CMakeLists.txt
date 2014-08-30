set( src
     Grid.cpp
     MemoryPool.cpp
     MemoryWindow.cpp
    )

if( ENABLE_CUDA )
    set( src ${src}
         CudaMemoryAllocator.cu
        )
endif( ENABLE_CUDA )

nebo_add_library( spatialops-structured ${src} )

if( ENABLE_TESTS )
    add_subdirectory( test )
endif( ENABLE_TESTS )

add_subdirectory( onedim )
target_link_libraries( spatialops-structured spatialops-onedim )

add_subdirectory( stencil )
target_link_libraries( spatialops-structured spatialops-stencil )

install( FILES
  BitField.h
  BoundaryCellInfo.h
  CudaMemoryAllocator.h
  DoubleVec.h
  ExternalAllocators.h
  FieldComparisons.h
  FieldHelper.h
  FieldInfo.h
  FVStaggeredBCTools.h
  FVStaggered.h
  FVStaggeredLocationTypes.h
  FVStaggeredFieldTypes.h
  Grid.h
  GhostData.h
  IndexTriplet.h
  IntVec.h
  Numeric3Vec.h
  MemoryPool.h
  MemoryTypes.h
  MemoryWindow.h
  SpatialField.h
  SpatialFieldStore.h
  SpatialMask.h
  DESTINATION include/spatialops/structured
  PERMISSIONS OWNER_READ GROUP_READ WORLD_READ
  )

install( TARGETS spatialops-structured
         EXPORT SpatialOps
         DESTINATION lib/spatialops/
         PERMISSIONS OWNER_READ GROUP_READ WORLD_READ
        )
