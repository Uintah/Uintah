set( F_FILE_PREFIXES
    rqpart
    )

foreach( name ${F_FILE_PREFIXES} )
  list( APPEND F_FILES ${CMAKE_CURRENT_SOURCE_DIR}/${name}.F )
endforeach()

# This will generate a header with appropriate name-mangling macros for C-Fortran compatibility
FortranCInterface_HEADER( FortranNameMangle.h SYMBOLS ${F_FILE_PREFIXES} )

set( F_SRC ${F_FILES} PARENT_SCOPE )
