include( CheckIncludeFile )
include( CheckTypeSize )
include( CheckCXXSourceCompiles )

configure_file( bits_testdefs.h.in   ${CMAKE_CURRENT_BINARY_DIR}/bits_defs.h @ONLY )

configure_file( blas_testdefs.h.in   ${CMAKE_CURRENT_BINARY_DIR}/blas_defs.h )

configure_file( boost_testdefs.h.in ${CMAKE_CURRENT_BINARY_DIR}/boost_defs.h )

configure_file( environment_testdefs.h.in ${CMAKE_CURRENT_BINARY_DIR}/environment_defs.h )

#--------------- error_testdefs.h.in ---------------
check_cxx_source_compiles( "#include <stdio.h>
    int main() {
      try {
        throw \"hello world\";
      }
      catch( const char* e ) {
        printf(\"error is %s\n\", e);
      }
    }"
    HAS_EXCEPTIONS_CRASH
    )
if( HAS_EXCEPTIONS_CRASH )
    set( EXCEPTIONS_CRASH_DEF )
endif()
configure_file( error_testdefs.h.in ${CMAKE_CURRENT_BINARY_DIR}/error_defs.h )
#--------------- error_testdefs.h.in ---------------

#--------------- hypre_testdefs.h.in ---------------
configure_file( hypre_testdefs.h.in ${CMAKE_CURRENT_BINARY_DIR}/hypre_defs.h )
#--------------- hypre_testdefs.h.in ---------------

if( HAVE_GPERFTOOLS )
    set( USE_GPERFTOOLS )
endif()
configure_file( gperftools_testdefs.h.in ${CMAKE_CURRENT_BINARY_DIR}/gperftools_defs.h )


configure_file( kokkos_testdefs.h.in ${CMAKE_CURRENT_BINARY_DIR}/kokkos_defs.h )

configure_file( compile_testdefs.h.in ${CMAKE_CURRENT_BINARY_DIR}/compile_defs.h )

#--------------- config_testdefs.h.in ---------------
check_include_file_cxx( ext/algorithm   HAVE_EXT_ALGORITHM )
check_include_file_cxx( inttypes.h      HAVE_INTTYPES_H )
check_include_file_cxx( limits.h        HAVE_LIMITS_H )
check_include_file_cxx( stdint.h        HAVE_STDINT_H )
check_include_file_cxx( strings.h       HAVE_STRINGS_H )
check_include_file_cxx( memory.h        HAVE_MEMORY_H )
check_include_file_cxx( string.h        HAVE_STRING_H )
check_include_file_cxx( sys/select.h    HAVE_SYS_SELECT_H )
check_include_file_cxx( sys/time.h      HAVE_SYS_TIME_H )
check_include_file_cxx( sys/types.h     HAVE_SYS_TYPES_H )
check_include_file_cxx( unistd.h        HAVE_UNISTD_H )

check_type_size( "long long" LONG_LONG LANGUAGE CXX )
check_type_size( void*       VOIDP     LANGUAGE CXX )

configure_file( config_testdefs.h.in ${CMAKE_CURRENT_BINARY_DIR}/config_defs.h )
#--------------- config_testdefs.h.in ---------------

if( ENABLE_CUDA )
    set( HAVE_CUDA true )
endif()
configure_file( cuda_testdefs.h.in ${CMAKE_CURRENT_BINARY_DIR}/cuda_defs.h )

if( LAPACK_FOUND )
    set( HAVE_LAPACK true )
endif()
configure_file( lapack_testdefs.h.in ${CMAKE_CURRENT_BINARY_DIR}/lapack_defs.h )

#--------------- malloc_testdefs.h.in ---------------
if( ENABLE_SCI_MALLOC )
    if( APPLE )
        message( WARNING "SCI_MALLOC probably does not work on OSX" )
    endif()
else()
    set( DISABLE_SCI_MALLOC true )
endif()
if( ENABLE_SCINEW_LINE_NUMBERS )
    set( USE_TAG_LINENUM true )
endif()
unset( MALLOC_TRACE )
if( ENABLE_MALLOC_TRACE )
    # jcs not sure what this malloc_trace stuff is.  See configure.ac line 1461
    set( MALLOC_TRACE true )
endif()
configure_file( malloc_testdefs.h.in ${CMAKE_CURRENT_BINARY_DIR}/malloc_defs.h )
#--------------- malloc_testdefs.h.in ---------------

configure_file( mpi_testdefs.h.in ${CMAKE_CURRENT_BINARY_DIR}/mpi_defs.h )


if( CMAKE_HOST_APPLE )
    set( IS_OSX true )
    if( CMAKE_HOST_SYSTEM_VERSION GREATER_EQUAL 10 )
        set( OSX_SNOW_LEOPARD_OR_LATER true )
    endif()
endif()
configure_file( osx_testdefs.h.in ${CMAKE_CURRENT_BINARY_DIR}/osx_defs.h )

# jcs petsc...
unset( HAVE_PETSC )
configure_file( petsc_testdefs.h.in ${CMAKE_CURRENT_BINARY_DIR}/petsc_defs.h )

configure_file( pidx_testdefs.h.in ${CMAKE_CURRENT_BINARY_DIR}/pidx_defs.h )

function( set_active_flags COMP )
#    unset( NO_${COMP} PARENT_SCOPE )
    if( NOT ENABLE_${COMP} )
        set( NO_${COMP} TRUE PARENT_SCOPE )
    endif()
endfunction()

set_active_flags( ARCHES )
set_active_flags( EXAMPLES )
set_active_flags( FVM )
set_active_flags( ICE )
set_active_flags( MPM )
set_active_flags( MODELS_RADIATION )
set_active_flags( PHASEFIELD )
set_active_flags( WASATCH )
set_active_flags( RAY_SCATTER )
set_active_flags( FORTRAN )

set( HAVE_CXX11 true )  # we are requiring this as the minimum c++ standard

# jcs what about zoltan, radprops?
configure_file( uintah_testdefs.h.in ${CMAKE_CURRENT_BINARY_DIR}/uintah_defs.h )

configure_file( visit_testdefs.h.in ${CMAKE_CURRENT_BINARY_DIR}/visit_defs.h )

# jcs visus

if( WASATCH_ENABLE_POKITT )
    set( HAVE_POKITT TRUE )
endif()
configure_file( wasatch_testdefs.h.in ${CMAKE_CURRENT_BINARY_DIR}/wasatch_defs.h )
