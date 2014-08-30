set( tabprops_SRC
  LagrangeInterpolant.cpp
  PropertyStash.cpp
  StateTable.cpp
  )
set( tabprops_H
  Archive.h
  LagrangeInterpolant.h
  PropertyStash.h
  StateTable.h
  TabProps.h
  )
  
add_library( interp STATIC LagrangeInterpolant.cpp )
target_link_libraries( interp ${LAPACK_LIBRARIES} ${Boost_LIBRARIES} )
install( TARGETS interp EXPORT Interpolant DESTINATION lib/tabprops )
install( EXPORT Interpolant DESTINATION lib/tabprops )
configure_file( ${PROJECT_SOURCE_DIR}/InterpConfig.cmake.in
  ${PROJECT_BINARY_DIR}/config/InterpConfig.cmake
  @ONLY
  )
install( FILES
  ${PROJECT_BINARY_DIR}/config/InterpConfig.cmake
  DESTINATION ${TabProps_CONFIG_INSTALL}
  )

  
add_library( tabprops STATIC ${tabprops_SRC} )
target_link_libraries( tabprops ${TPL_LIBRARIES} )

if( TabProps_PREPROCESSOR )
  message( STATUS "Building TabProps preprocessing tools" )
  add_subdirectory( prepro )
endif( TabProps_PREPROCESSOR )

if( TabProps_ENABLE_TESTING )
  add_subdirectory( test )
endif()
add_subdirectory( util )

# Installation
install( TARGETS tabprops ARCHIVE DESTINATION lib/tabprops )
install( TARGETS tabprops EXPORT TabProps DESTINATION lib/tabprops )
install( EXPORT TabProps DESTINATION lib/tabprops )

install( FILES
  ${tabprops_H}
  DESTINATION include/tabprops
  PERMISSIONS OWNER_READ GROUP_READ WORLD_READ
  )
