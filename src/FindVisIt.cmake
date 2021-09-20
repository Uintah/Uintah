include( FindPackageHandleStandardArgs )

set( VISIT_FOUND false )

find_path(
    VISIT_INCLUDE_DIR
    NAMES VisItControlInterface_V2.h
    PATHS ${VISIT_DIR}
    PATH_SUFFIXES
      libsim/V2/include
      src/sim/V2/lib
    DOC
      "VisIt include path"
    )

find_library(
    VISIT_LIBRARY
    NAMES simV2
    PATHS ${VISIT_DIR}
    PATH_SUFFIXES
      lib
      libsim/V2/lib
    )

find_package_handle_standard_args( VISIT
    FOUND_VAR VISIT_FOUND
    REQUIRED_VARS
      VISIT_LIBRARY
      VISIT_INCLUDE_DIR
    )

if( VISIT_FOUND )
  set( VISIT_LIBRARIES ${VISIT_LIBRARY} )
  set( VISIT_INCLUDE_DIRS ${VISIT_INCLUDE_DIR} )

  if( NOT TARGET VISIT::VISIT )
    add_library( VISIT::VISIT UNKNOWN IMPORTED )
  endif()
  set_target_properties( VISIT::VISIT PROPERTIES
      INTERFACE_INCLUDE_DIRECTORIES "${VISIT_INCLUDE_DIR}"
      IMPORTED_LOCATION ${VISIT_LIBRARY}
      )

  mark_as_advanced(
      VISIT_INCLUDE_DIR
      VISIT_LIBRARY
  )

endif()
