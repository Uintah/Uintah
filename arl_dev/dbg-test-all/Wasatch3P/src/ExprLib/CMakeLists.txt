cmake_minimum_required(VERSION 2.8.4)
project(exprlib)

set( MAJOR_VERSION "1" )
set( MINOR_VERSION "2" )
set( PATCH_VERSION "0" )

# Information for creating packages
set( CPACK_PACKAGE_VENDOR "James C. Sutherland (james.sutherland@utah.edu" )
set( CPACK_PACKAGE_VERSION_MAJOR ${MAJOR_VERSION} )
set( CPACK_PACKAGE_VERSION_MINOR ${MINOR_VERSION} )
set( CPACK_PACKAGE_VERSION_PATCH ${PATCH_VERSION} )
include( CPack )


option( ENABLE_UINTAH	"Build the Uintah interface"                        OFF )
option( ENABLE_LBMS	    "Build the LBMS interface"                          OFF )
option( ENABLE_OUTPUT	"Enable field output database support (incompatible with LBMS, Uintah)" OFF )
option( ENABLE_TESTS	"Build the regression tests (recommended)"          ON  )

# default to release builds
if( NOT CMAKE_BUILD_TYPE )
  set( CMAKE_BUILD_TYPE "Release" CACHE STRING
    "Choose the type of build, options are: Debug Release RelWithDebInfo MinSizeRel."
    FORCE )
endif( NOT CMAKE_BUILD_TYPE )

# Set the location for the installation.  Comment this out to install
# to /usr/local.  Or modify it to install to a different location.
if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
  set( CMAKE_INSTALL_PREFIX
    ${CMAKE_CURRENT_BINARY_DIR}
    CACHE PATH "Installation directory" FORCE
    )
endif()

set( EXPR_CONFIG_INSTALL
  ${CMAKE_INSTALL_PREFIX}/share
  CACHE PATH
  "Location to install configuration files"
  )

#--------------------------------------
# resolve required libraries

#-- SpatialOps
find_package( SpatialOps REQUIRED )
set( TPL_INCLUDE_DIRS ${TPL_INCLUDE_DIRS} ${SpatialOps_INCLUDE_DIR} )
set( TPL_LIBRARIES ${TPL_LIBRARIES} spatialops )
if( NOT NTHREADS )
  set( NTHREADS 1 )
endif()

#-- SpatialOps/Cuda
if( ENABLE_CUDA )
  message( STATUS "Exprlib built with CUDA support" )
endif( ENABLE_CUDA )


#-- Boost
# SpatialOps will set information about where it found the Boost libraries so that
# we can pick that up here and look for additional boost libraries to build against.
set(Boost_ADDITIONAL_VERSIONS ${Boost_ADDITIONAL_VERSIONS}
 "1.50" "1.50.0" "1.51" "1.51.0" "1.52" "1.52.0" "1.53" "1.53.0" "1.54" "1.54.0"
 "1.55" "1.55.0" )
set(Boost_USE_MULTITHREAD ON)
if( NOT ENABLE_LBMS AND NOT ENABLE_UINTAH AND ENABLE_OUTPUT )
  set( boost_comps ${boost_comps} "filesystem;system" )
endif()
set( boost_comps ${boost_comps} "program_options;system" )
if( DEFINED BOOST_ROOT )
  message( STATUS "Setting boost installation location as: " ${BOOST_ROOT} )
  set( Boost_NO_SYSTEM_PATHS ON )  # only look in the user-specified location - nowhere else!
endif( DEFINED BOOST_ROOT )
if( DEFINED BOOST_INCLUDEDIR AND DEFINED BOOST_LIBRARYDIR )
  message( STATUS "Setting boost include location as: " ${BOOST_INCLUDEDIR} )
  message( STATUS "Setting boost library location as: " ${BOOST_LIBRARYDIR} )
  set( Boost_NO_SYSTEM_PATHS ON )  # only look in the user-specified location - nowhere else!
endif()
message( STATUS "Looking for boost libs: '${boost_comps}'" )
set( ENV{BOOST_INCLUDEDIR} ${Boost_INCLUDE} )
set( ENV{BOOST_LIBRARYDIR} ${Boost_LIB} )
set(BOOST_MIN_VERSION "1.49.0" )
find_package( Boost ${BOOST_MIN_VERSION} REQUIRED COMPONENTS ${boost_comps} )
message( STATUS "Boost header installation found at: " ${Boost_INCLUDE_DIR} )
message( STATUS "      library installation found at: " ${Boost_LIBRARY_DIRS} )
message( STATUS "      version: " ${Boost_MAJOR_VERSION}.${Boost_MINOR_VERSION}.${Boost_SUBMINOR_VERSION} )
message( STATUS " found boost libs: '${Boost_LIBRARIES}'" )

set( TPL_INCLUDE_DIRS ${TPL_INCLUDE_DIRS} ${Boost_INCLUDE_DIR} )
set( TPL_LIBRARIES    ${TPL_LIBRARIES}    ${Boost_LIBRARIES} )
#--------------------------------------

if( NOT ENABLE_UINTAH AND NOT ENABLE_LBMS )
  set( DefaultFieldManager ON )
else()
  set( DefaultFieldManager OFF )
endif()

# set include directories for external library dependencies:
include_directories( ${TPL_INCLUDE_DIRS} ${PROJECT_SOURCE_DIR} )

if( ENABLE_OUTPUT )
  set( ExprCompilerDefs ${ExprCompilerDefs} -DHAVE_EXPR_DB_OUTPUT )
endif()
if( ENABLE_UINTAH )
  set( ExprCompilerDefs ${ExprCompilerDefs} -DENABLE_UINTAH )
endif()

foreach( prop ${ExprCompilerDefs} )
  message( STATUS "Setting compiler defintion: ${prop}" )
  add_definitions( ${prop} )
endforeach()

include( CreateExprHeaderFile.cmake )
add_subdirectory( expression )

include_directories( ${CMAKE_BINARY_DIR}/config )

nebo_add_library( exprlib STATIC ${expr_sources} )
target_link_libraries( exprlib ${TPL_LIBRARIES} )
  
if( ENABLE_TESTS )
  enable_testing()
  add_subdirectory( test )
endif()
add_subdirectory( util )

message( STATUS "Files will be installed to: " ${CMAKE_INSTALL_PREFIX} )


#------------------------------------------------------------------
# configuration for projects using ExprLib

# lump SpatialOps includes here as well
set( TPL_INCLUDE_DIRS ${TPL_INCLUDE_DIRS}
  ${SpatialOps_INCLUDE_DIR}
  ${SpatialOps_TPL_INCLUDE_DIRS}
  )
set( ExprLib_INCLUDE ${ExprLib_INCLUDE} ${CMAKE_INSTALL_PREFIX}/include )

configure_file(
  ${PROJECT_SOURCE_DIR}/ExprLibConfig.cmake.in
  ${CMAKE_BINARY_DIR}/config/ExprLibConfig.cmake
  @ONLY
  )

install( TARGETS exprlib EXPORT ExprLib DESTINATION lib/expression )
install( EXPORT ExprLib DESTINATION lib/expression )
install( FILES
  ${CMAKE_BINARY_DIR}/config/ExprLibConfig.cmake
  DESTINATION ${EXPR_CONFIG_INSTALL}
  )
#-----------------------------------------------------------------
