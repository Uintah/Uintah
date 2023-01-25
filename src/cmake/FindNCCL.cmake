if( NOT DEFINED NCCL_ROOT_DIR )
  find_package(PkgConfig)
  pkg_check_modules( PC_NCCL QUIET nccl )
endif()

find_path( NCCL_INCLUDE_DIR nccl.h
           HINTS ${PC_NCCL_INCLUDEDIR}
                 ${PC_NCCL_INCLUDE_DIRS}
           PATHS ${NCCL_ROOT_DIR}
           PATH_SUFFIXES include
)

find_library( NCCL_LIBRARY NAMES nccl
              HINTS ${PC_NCCL_LIBDIR}
                    ${PC_NCCL_LIBRARY_DIRS}
              PATHS ${NCCL_ROOT_DIR}
              PATH_SUFFIXES lib lib64 lib32
)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args( 
  NCCL DEFAULT_MSG
  NCCL_LIBRARY
  NCCL_INCLUDE_DIR
)

if( NCCL_FOUND AND NOT TARGET NCCL::nccl )

  set( NCCL_INCLUDE_DIRS ${NCCL_INCLUDE_DIR} )
  set( NCCL_LIBRARIES    ${NCCL_LIBRARY}     )

  add_library( NCCL::nccl INTERFACE IMPORTED )
  set_target_properties( NCCL::nccl PROPERTIES
    INTERFACE_INCLUDE_DIRECTORIES "${NCCL_INCLUDE_DIRS}"
    INTERFACE_LINK_LIBRARIES      "${NCCL_LIBRARIES}"
  )

endif()