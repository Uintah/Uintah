set( F_FILE_PREFIXES
    hottel
    radarray
    radcal
    radcoef
    radwsgg
    rdombmcalc
    rdomsolve
    rordr
    rordrss
    rordrtn
    )

# This will generate a header with appropriate name-mangling macros for C-Fortran compatibility
FortranCInterface_HEADER( FortranNameMangle.h
    SYMBOL_NAMESPACE "F_"
    SYMBOLS ${F_FILE_PREFIXES}
    )

# These fortran functions don't have C++ linkage, so we don't need name mangling on them:
list( APPEND F_FILE_PREFIXES
    eco
    eco2
    efuel
    eh2o
    eico2
    eih2o
    find
    fixradval
    soot
    )

foreach( name ${F_FILE_PREFIXES} )
  list( APPEND F_FILES ${CMAKE_CURRENT_SOURCE_DIR}/${name}.F )
  #  list( APPEND INC_FILES ${name}_fort.h )
endforeach()

set( F_SRC ${F_SRC} ${F_FILES} PARENT_SCOPE )
