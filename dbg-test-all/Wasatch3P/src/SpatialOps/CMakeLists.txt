cmake_minimum_required(VERSION 2.8.4)
project( SpatialOps CXX C )

set( MAJOR_VERSION "1" )
set( MINOR_VERSION "3" )
set( PATCH_VERSION "0" )


option( ENABLE_TESTS "Enable building of regression tests (recommended)" ON )
option( ENABLE_EXAMPLES "Enable building of examples tests" ON )
option( ENABLE_THREADS "Enable multi-threaded support for stencil operations" OFF )
option( ENABLE_CUDA   "Build Spatial Ops with CUDA support" OFF )
option( NEBO_REPORT_BACKEND "Require Nebo to report what backend it is using" OFF )

set( NTHREADS 1 CACHE STRING "Number of threads to use if ENABLE_THREADS is ON" )

# default to release builds
if( NOT CMAKE_BUILD_TYPE )
  set( CMAKE_BUILD_TYPE "Release" CACHE STRING
    "Choose the type of build, options are: Debug Release RelWithDebInfo MinSizeRel."
    FORCE )
endif( NOT CMAKE_BUILD_TYPE )

# Set the location for the installation.
if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
  set (CMAKE_INSTALL_PREFIX
    ${CMAKE_CURRENT_BINARY_DIR}
    CACHE PATH "Installation directory" FORCE
    )
endif()


set( SpatialOps_CONFIG_INSTALL
  ${CMAKE_INSTALL_PREFIX}/share
  CACHE PATH
  "Location to install configuration files"
  )

include( NeboBuildTools.cmake )

#--------------------------------------
# resolve required libraries

# setup the boost library
set( boost_comps "program_options" )

if( ENABLE_THREADS )
  find_package( Threads REQUIRED )
  set( TPL_LIBRARIES ${TPL_LIBRARIES} ${CMAKE_THREAD_LIBS_INIT} )
  set( boost_comps ${boost_comps} "thread;system" )
  set(Boost_USE_MULTITHREADED ON)
  message( STATUS "Thread libs: " ${CMAKE_THREAD_LIBS_INIT} )
else()
  set( NTHREADS 1 )
endif( ENABLE_THREADS )

set(Boost_USE_MULTITHREAD ON)

if( DEFINED BOOST_ROOT )
  message( STATUS "Setting boost installation location as: " ${BOOST_ROOT} )
  set( Boost_NO_SYSTEM_PATHS ON )  # only look in the user-specified location - nowhere else!
endif()
if( DEFINED BOOST_INCLUDEDIR AND DEFINED BOOST_LIBRARYDIR )
  message( STATUS "Setting boost include location as: " ${BOOST_INCLUDEDIR} )
  message( STATUS "Setting boost library location as: " ${BOOST_LIBRARYDIR} )
  set( Boost_NO_SYSTEM_PATHS ON )  # only look in the user-specified location - nowhere else!
endif()

set(BOOST_MIN_VERSION "1.49.0" )
set(Boost_ADDITIONAL_VERSIONS
 ${Boost_ADDITIONAL_VERSIONS}
 "1.49" "1.49.0" "1.50" "1.50.0" "1.51" "1.51.0" "1.52" "1.52.0"
 "1.53" "1.53.0" "1.54" "1.54.0" "1.55" "1.55.0" )
find_package( Boost ${BOOST_MIN_VERSION} REQUIRED COMPONENTS ${boost_comps} )

if( NOT Boost_FOUND )
  message( FATAL_ERROR "Boost was not found.  Try setting BOOST_INCLUDEDIR and BOOST_LIBRARYDIR to the path where boost is installed and try again" )
else()
  message( STATUS "Boost header installation found at: " ${Boost_INCLUDE_DIR} )
  message( STATUS "      library installation found at: " ${Boost_LIBRARY_DIRS} )
  message( STATUS "      version: " ${Boost_MAJOR_VERSION}.${Boost_MINOR_VERSION}.${Boost_SUBMINOR_VERSION} )
endif()
set( TPL_INCLUDE_DIRS ${TPL_INCLUDE_DIRS} ${Boost_INCLUDE_DIR} )
if( boost_comps )
  set( TPL_LIBRARIES ${TPL_LIBRARIES} ${Boost_LIBRARIES} )
  message( STATUS ${Boost_LIBRARIES} )
endif( boost_comps )

#-- CUDA
if( ENABLE_CUDA )
  message( STATUS "SpatialOps will be compiled with CUDA Support" )
  find_package( CUDA REQUIRED )
  set( CUDA_ARCHITECTURE "3.0" CACHE STRING "CUDA compute capability" )
  string( REGEX REPLACE "([1-9])\\.([0-9])" "\\1\\2" CUDA_ARCH_STR ${CUDA_ARCHITECTURE} )
  set( CUDA_NVCC_FLAGS "-arch=sm_${CUDA_ARCH_STR}" CACHE STRING "Compiler flags passed to NVCC" FORCE )
  set( CUDA_NVCC_FLAGS_RELEASE "--optimize=03" CACHE STRING "Release build compiler flags" FORCE )
  option( NEBO_GPU_TEST "Require Nebo to handle data transfer between CPU and GPU" OFF )
  set( TPL_LIBRARIES ${TPL_LIBRARIES} ${CUDA_LIBRARIES} )
  set( TPL_INCLUDE_DIRS ${TPL_INCLUDE_DIRS} ${CUDA_INCLUDE_DIRS} )
else( ENABLE_CUDA )
  # avoids confusion if someone tries to build with cuda support enabled and then
  # reverts without clearing the build directory
  SET( CUDA_CUDA_LIBRARY OFF )
endif( ENABLE_CUDA )

#--------------------------------------

include_directories(
  ${PROJECT_SOURCE_DIR}
  ${TPL_INCLUDE_DIRS}
  )

# look for git.  This is used to configure version information into the
# executable and also to build upstream dependencies if necessary
set( SOPS_REPO_DATE "\"DATE NOT FOUND\"" )
set( SOPS_REPO_HASH "\"HASH NOT FOUND\"" )
find_package( Git )
if( GIT_FOUND )
  execute_process(
    COMMAND ${GIT_EXECUTABLE} log -1 "--pretty=format:\"%H\""
    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
    RESULT_VARIABLE RESULT
    OUTPUT_VARIABLE SOPS_REPO_HASH
    OUTPUT_STRIP_TRAILING_WHITESPACE
    )
  if( NOT ${RESULT} EQUAL 0 )
    set( SOPS_REPO_HASH "\"HASH NOT FOUND\"" )
  endif()
  execute_process(
    COMMAND ${GIT_EXECUTABLE} log -1 "--pretty=format:\"%cd\""
    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
    OUTPUT_VARIABLE SOPS_REPO_DATE
    OUTPUT_STRIP_TRAILING_WHITESPACE
    )
   if( NOT ${RESULT} EQUAL 0 )
     set( SOPS_REPO_DATE "\"DATE NOT FOUND\"" )
   endif()
endif( GIT_FOUND )


# build the configure file using the information defined here.
configure_file(
  ${PROJECT_SOURCE_DIR}/SpatialOpsConfigure.h.in
  ${PROJECT_BINARY_DIR}/config/spatialops/SpatialOpsConfigure.h
  )
include_directories( ${PROJECT_BINARY_DIR}/config )

if( ENABLE_TESTS )
  enable_testing()
endif( ENABLE_TESTS )

add_subdirectory( spatialops )


if( ENABLE_EXAMPLES )
  add_subdirectory( examples )
endif( ENABLE_EXAMPLES )

# this will compile all of the test cases.  Use "make test" to execute
# them and see a summary.
if( ENABLE_TESTS )
  add_subdirectory( test )
endif()



#--- INSTALLATION SECTION ---
# "make install" performs these commands, along with any install commands inside child CMakeLists.txt files.
message( STATUS "Files will be installed to: " ${CMAKE_INSTALL_PREFIX} )

# install spatialops headers
install( FILES
  ${PROJECT_BINARY_DIR}/config/spatialops/SpatialOpsConfigure.h
  DESTINATION include/spatialops
  PERMISSIONS OWNER_READ GROUP_READ WORLD_READ
  )


# ------------------------------------------------------------------
# configuration for projects using SpatialOps this section defines
# variables that are used for building the configuration file for the
# SpatialOps installation.  This can be used by other CMake build
# systems that use the SpatialOps library.

set( SpatialOps_INCLUDE ${CMAKE_INSTALL_PREFIX}/include )

# at this point, all of the required variables are defined to generate
# the configure file
configure_file(
  ${PROJECT_SOURCE_DIR}/SpatialOpsConfig.cmake.in
  ${PROJECT_BINARY_DIR}/config/SpatialOpsConfig.cmake
  @ONLY
  )

# install all of these files into the spatial ops lib dir
install( FILES
  ${PROJECT_BINARY_DIR}/config/SpatialOpsConfig.cmake
  ${PROJECT_SOURCE_DIR}/NeboBuildTools.cmake
   DESTINATION ${SpatialOps_CONFIG_INSTALL}
   )
#-----------------------------------------------------------------


# Information for creating packages
#set( CPACK_PACKAGE_VENDOR "James C. Sutherland (james.sutherland@utah.edu" )
#set( CPACK_PACKAGE_VERSION_MAJOR ${MAJOR_VERSION} )
#set( CPACK_PACKAGE_VERSION_MINOR ${MINOR_VERSION} )
#set( CPACK_PACKAGE_VERSION_PATCH ${PATCH_VERSION} )
#include( CPack )
