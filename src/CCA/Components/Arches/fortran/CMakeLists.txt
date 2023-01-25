set( F_FILE_PREFIXES
    arrass
    cellg
    mascal_scalar
    mm_computevel
    mm_explicit_vel
    mmbcvelocity
    prescoef_var
    pressrcpred
    pressrcpred_var
    smagmodel
    uvelcoef
    uvelcoef_central
    uvelcoef_hybrid
    uvelcoef_mixed
    uvelcoef_upwind
    uvelsrc
    vvelcoef
    vvelcoef_central
    vvelcoef_hybrid
    vvelcoef_mixed
    vvelcoef_upwind
    vvelsrc
    wallbc
    wvelcoef
    wvelcoef_central
    wvelcoef_hybrid
    wvelcoef_mixed
    wvelcoef_upwind
    wvelsrc
    )

# This will generate a header with appropriate name-mangling macros for C-Fortran compatibility
FortranCInterface_HEADER( FortranNameMangle.h
    SYMBOL_NAMESPACE F_
    SYMBOLS ${F_FILE_PREFIXES}
    )

# These fortran functions don't have C++ linkage, so we don't need name mangling on them:
list( APPEND F_FILE_PREFIXES
    fixval
    fixval_trans
    )

foreach( name ${F_FILE_PREFIXES} )
  list( APPEND F_FILES ${CMAKE_CURRENT_SOURCE_DIR}/${name}.F )
#  list( APPEND INC_FILES ${name}_fort.h )
endforeach()

# advertise these files upward to be included in the ARCHES_SRC list of files to compile
set( ARCHES_SRC ${ARCHES_SRC} ${F_FILES} PARENT_SCOPE )
