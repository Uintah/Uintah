add_subdirectory( structured )
add_subdirectory( particles  )
add_subdirectory( pointfield )

set( SO_HEADERS
  Nebo.h
  NeboBasic.h
  NeboRhs.h
  NeboMask.h
  NeboOperators.h
  NeboCond.h
  NeboStencils.h
  NeboStencilBuilder.h
  NeboLhs.h
  NeboAssignment.h
  NeboReductions.h
  FieldFunctions.h
  OperatorDatabase.h
  SpatialOpsDefs.h
  SpatialOpsTools.h
  Semaphore.h
  )

# create dummy/empty file to circumvent restriction on add_library
file( WRITE ${CMAKE_CURRENT_BINARY_DIR}/empty.cpp "" )

# set dummy/empty file to SO_Sources
set( SO_Sources ${CMAKE_CURRENT_BINARY_DIR}/empty.cpp )

if( ENABLE_THREADS )
    set( SO_Sources ${SO_Sources} ThreadPool.cpp )
    set( SO_HEADERS ${SO_HEADERS} ThreadPool.h )
    add_subdirectory( threadpool )
endif( ENABLE_THREADS )

nebo_add_library( spatialops ${SO_Sources} )

target_link_libraries( spatialops ${TPL_LIBRARIES} )
target_link_libraries( spatialops spatialops-structured )

install( FILES
  ${SO_HEADERS}
  DESTINATION include/spatialops
  PERMISSIONS OWNER_READ GROUP_READ WORLD_READ
  )

install( TARGETS
  spatialops
  EXPORT SpatialOps
  DESTINATION lib/spatialops/
  PERMISSIONS OWNER_READ GROUP_READ WORLD_READ
  )
install( EXPORT SpatialOps DESTINATION lib/spatialops )
