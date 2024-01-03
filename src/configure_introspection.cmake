# Other introspection to consider:
# Compiler flags: https://cmake.org/cmake/help/latest/module/CheckCompilerFlag.html
# Check for library: https://cmake.org/cmake/help/latest/module/CheckLibraryExists.html
# Check linker flags: https://cmake.org/cmake/help/latest/module/CheckLinkerFlag.html
# Processor count: https://cmake.org/cmake/help/latest/module/ProcessorCount.html

include( CheckLibraryExists )
include( CheckIncludeFileCXX )
include( CheckCXXSourceCompiles )
include( CheckTypeSize )  # CMake module for type introspection - https://cmake.org/cmake/help/latest/module/CheckTypeSize.html
include( ProcessorCount )

ProcessorCount( NUMBER_OF_PROCESSORS )

set( CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR} )

check_library_exists( m fabs "" HAVE_LIB_M )
if( NOT HAVE_LIB_M )
    message( SEND_ERROR "Math library not found" )
endif()

#----------------------------------------------------------
# jcs rather than caching some of these, we can generate the appropriate header files here and avoid caching...
function( validate_type_size the_type type_name allowable_sizes )
    check_type_size( ${the_type} ${type_name} )
    if( NOT HAVE_${type_name} )
        message( SEND_ERROR "Required type: ${the_type} not found." )
    endif()
    list( FIND ${allowable_sizes} $CACHE{${type_name}} loc )
    if( loc LESS_EQUAL 0 )
        message( SEND_ERROR " Unexpected sizeof(${the_type}). Found a size of $CACHE{${type_name}}" )
    endif()
    #message( STATUS "${the_type} -> $CACHE{${type_name}} " )
endfunction()

if( ${CMAKE_SIZEOF_VOID_P} EQUAL 8 )
    set( NBITS 64 CACHE INTERNAL "bit size (32/64)" )
else()
    set( NBITS 32 CACHE INTERNAL "bit size (32/64)" )
endif()
# Check long long size
set( sizes 4 8 )
validate_type_size( "long long" "LONG_LONG" sizes )
if( ${LONG_LONG} EQUAL 8 )
    set( LONG_LONG_SWAP "SWAP_8" )
elseif( ${LONG_LONG} EQUAL 4 )
    set( LONG_LONG_SWAP "SWAP_4" )
else()
    message( SEND_ERROR "Invalid long long size: ${LONG_LONG}" )
endif()
#----------------------------------------------------------

# On Intel compilers, ICE needs a flag set:
if( CMAKE_CXX_COMPILER_ID STREQUAL "Intel" )
    add_compile_options( "$<$<COMPILE_LANGUAGE:CXX>:SHELL:-fp-model precise>" )
    add_compile_options( "$<$<COMPILE_LANGUAGE:Fortran>:SHELL:-fp-model precise>" )
endif()

#----------------------------------------------------------
# Check for ENDIAN-ness of the platform -- https://cmake.org/cmake/help/latest/module/TestBigEndian.html
include( TestBigEndian )
test_big_endian( ENDIAN )
if( ${ENDIAN} )
    set( PLATFORM_IS_BIG_ENDIAN TRUE )
else()
    set( PLATFORM_IS_LITTLE_ENDIAN TRUE )
endif()
#----------------------------------------------------------

#----------------------------------------------------------
#
# Find some required libraries
#
#--------------------------------------------------------------------

#--------------------------------------------------------------------
# Boost: https://cmake.org/cmake/help/latest/module/FindBoost.html
if( DEFINED ENV{BOOST_DIR} )
    message( STATUS "searching for boost in environment path: $ENV{BOOST_DIR}" )
    set( BOOST_ROOT $ENV{BOOST_DIR} )
endif()
find_package( Boost
        1.65 # minimum version
        COMPONENTS filesystem system
    ) # Boost_FOUND, Boost_INCLUDE_DIRS, Boost_LIBRARY_DIRS, Boost_LIBRARIES
if( NOT Boost_FOUND )
    message( WARNING "Boost wasn't found. Try defining 'BOOST_ROOT' or 'BOOSTROOT' in your configure line" )
    unset( HAVE_BOOST )
else()
    set( HAVE_BOOST true )
    message( STATUS "Boost installation information:"
        "\n\theader : ${Boost_INCLUDE_DIR}"
        "\n\tlibrary: ${Boost_LIBRARY_DIRS}"
        "\n\tversion: ${Boost_MAJOR_VERSION}.${Boost_MINOR_VERSION}.${Boost_SUBMINOR_VERSION}"
        "\n\tlibs   : ${Boost_LIBRARIES}" )
endif()
#--------------------------------------------------------------------

#--------------------------------------------------------------------
# see https://cmake.org/cmake/help/latest/module/FindMPI.html
find_package( MPI REQUIRED COMPONENTS CXX ) # MPI::MPI_CXX
if( NOT MPI_FOUND )
    message( SEND_ERROR "MPI was not found. Try setting some of the following:"
        "\n\tMPIEXEC_EXECUTABLE - location of mpirun/mpiexec"
        "\n\tMPI_HOME - base directory of the MPI installation"
        "\n\tMPI_C_COMPILER - location of the MPI C compiler"
        "\n\tMPI_CXX_COMPILER - location of the MPI CXX compiler"
        "\nAlso see https://cmake.org/cmake/help/latest/module/FindMPI.html for more information"
        )
endif()
set( CMAKE_REQUIRED_INCLUDES ${MPI_CXX_INCLUDE_DIRS} )
set( CMAKE_REQUIRED_FLAGS ${MPI_CXX_LINK_FLAGS} )
set( CMAKE_REQUIRED_LIBRARIES ${MPI_CXX_LIBRARIES} )
check_cxx_source_compiles(
        "#include <mpi.h>
inline int Add_error_string( int errorcode , const char *string ){
return MPI_Add_error_string( errorcode , string );
}
int main(){ return 0; }
"
        MPI_CONST_WORKS
)
if( MPI_CONST_WORKS )
    set( MPICONST "const" )
endif()
message( STATUS "MPI INFORMATION:"
    "\n\tInclude dir: ${MPI_CXX_INCLUDE_DIRS}"
    "\n\tCXX Compiler: ${MPI_CXX_COMPILER}"
    )
#--------------------------------------------------------------------

#--------------------------------------------------------------------
## Python3: https://cmake.org/cmake/help/latest/module/FindPython3.html
find_package( Python3 COMPONENTS Interpreter )  # ${Python3_FOUND} ${Python3_EXECUTABLE}
if( NOT Python3_FOUND )
    message( WARNING "Python3 was not found. Regression testing won't work" )
endif()
#--------------------------------------------------------------------

#--------------------------------------------------------------------
# Zlib: https://cmake.org/cmake/help/latest/module/FindZLIB.html
find_package( ZLIB REQUIRED ) # imports target ZLIB::ZLIB, defines ZLIB_INCLUDE_DIRS, ZLIB_LIBRARIES
if( NOT ZLIB_FOUND )
    message( SEND_ERROR "ZLib was not found. Consider defining 'ZLIB_ROOT' to the path where zlib is installed" )
endif()
#--------------------------------------------------------------------

#--------------------------------------------------------------------
# https://cmake.org/cmake/help/latest/module/FindLibXml2.html
find_package( LibXml2 REQUIRED )  # LIBXML2_LIBRARIES LIBXML2_INCLUDE_DIRS  -- can switch to LibXml2::LibXml2 once CMake 3.12 is required
#--------------------------------------------------------------------

#--------------------------------------------------------------------
# BLAS: https://cmake.org/cmake/help/latest/module/FindBLAS.html
find_package( BLAS )  # BLAS_LIBRARIES
if( APPLE )
    set( BLA_VENDOR Apple )
    set( HAVE_ACCELERATE true )
    set( CBLAS_H "Accelerate/Accelerate.h" )
    set( HAVE_BLAS true )
    set( HAVE_CBLAS true )
else()
    function ( set_blas_header headers )
        set( CMAKE_REQUIRED_LIBRARIES ${BLAS_LIBRARIES} )
        foreach( header ${headers} )
            find_path( CBLAS_H_PATH ${header} )
            set( CMAKE_REQUIRED_INCLUDES ${CBLAS_H_PATH} )
            check_include_file_cxx( ${header} HAVE_BLAS_H )
            unset( CMAKE_REQUIRED_INCLUDES )
            if( ${HAVE_BLAS_H} )
                set( CBLAS_H ${header} PARENT_SCOPE )
                message( STATUS "Found BLAS header: ${header}" )
                set( HAVE_BLAS true )
                set( HAVE_CBLAS true )
                break()
            endif()
        endforeach()
        unset( CMAKE_REQUIRED_LIBRARIES )
        if( NOT HAVE_BLAS_H )
            message( "BLAS header wasn't found in: ${headers}" )
        endif()
    endfunction()
    set_blas_header( "cblas.h;vecLib/cblas.h;vecLib.h" )
endif()

# LAPACK: https://cmake.org/cmake/help/latest/module/FindLAPACK.html
find_package( LAPACK ) # LAPACK_LIBRARIES
#--------------------------------------------------------------------


#--------------------------------------------------------------------
if( ENABLE_CUDA )
    # https://cmake.org/cmake/help/latest/module/FindCUDAToolkit.html
    find_package( CUDAToolkit REQUIRED )
endif()
#--------------------------------------------------------------------

#--------------------------------------------------------------------
# Git information  https://cmake.org/cmake/help/latest/module/FindGit.html
find_package( Git )
if( GIT_FOUND )
    execute_process(
            COMMAND ${GIT_EXECUTABLE} rev-parse HEAD
            WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
            RESULT_VARIABLE RESULT
            OUTPUT_VARIABLE GIT_HASH
            OUTPUT_STRIP_TRAILING_WHITESPACE
        )
    if( NOT ${RESULT} EQUAL 0 )
        set( GIT_HASH "HASH NOT FOUND" )
    endif()
    execute_process(
            COMMAND ${GIT_EXECUTABLE} log -1 "--pretty=format:\"%cd\""
            WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
            OUTPUT_VARIABLE GIT_DATE
            OUTPUT_STRIP_TRAILING_WHITESPACE
        )
    if( NOT ${RESULT} EQUAL 0 )
        set( GIT_DATE "\"DATE NOT FOUND\"" )
    endif()
    execute_process(
            COMMAND ${GIT_EXECUTABLE} rev-parse --abbrev-ref HEAD
            WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
            OUTPUT_VARIABLE GIT_BRANCH
            OUTPUT_STRIP_TRAILING_WHITESPACE
        )
    if( NOT ${RESULT} EQUAL 0 )
        set( GIT_BRANCH "BRANCH NOT FOUND" )
    endif()
else()
    set( GIT_HASH "HASH NOT FOUND" )
    set( GIT_DATE "\"DATE NOT FOUND\"" )
    set( GIT_BRANCH "BRANCH NOT FOUND" )
endif()
##--------------------------------------------------------------------

##----------------------------- HYPRE --------------------------------
unset( HAVE_HYPRE )
find_package( HYPRE )
if( HYPRE_FOUND )
    message( STATUS "HYPRE:"
        "\n\tVERSION: ${HYPRE_VERSION}"
        "\n\tInclude dir: ${HYPRE_INCLUDE_DIR}"
        )
    set( HAVE_HYPRE true )
endif()
##----------------------------- HYPRE --------------------------------

##----------------------------- VISIT --------------------------------
find_package( VISIT )
if( VISIT_FOUND )
    message( STATUS "VisIt:"
        "\n\tInclude dir: ${VISIT_INCLUDE_DIRS}"
        "\n\tLibrary: ${VISIT_LIBRARIES}"
        )
    set( HAVE_VISIT TRUE )
else()
    message( STATUS "VisIt was not found. "
        "If you want to build VisIt in-situ extensions, specify"
        "\n\t\tVISIT_DIR=[path to visit installation]"
        "\n\twith the location of the VisIt installation"
        )
    unset( HAVE_VISIT )
endif()
##----------------------------- VISIT --------------------------------

find_package( Threads REQUIRED )

##----------------------------- gperf --------------------------------
unset( HAVE_GPERFTOOLS )
if( ENABLE_GPERFTOOLS )
    find_package( GPERFTOOLS )
    if( GPERFTOOLS_FOUND )
        set( HAVE_GPERFTOOLS true )
    else()
        message( WARNING "GPERFTOOLS not found. Try setting `GPERFTOOLS_ROOT_DIR` to the path where it is installed" )
    endif()
endif()
##----------------------------- gperf --------------------------------

##--------------------------------------------------------------------
# Some tools to find libraries:
#
# Doxygen: https://cmake.org/cmake/help/latest/module/FindDoxygen.html
# OpenMP: https://cmake.org/cmake/help/latest/module/FindOpenMP.html
# CUDA: https://cmake.org/cmake/help/latest/module/FindCUDAToolkit.html

# PIDX...?
