# SpatialOpsConfig.cmake - defines variables to assist in applications
# that link with SpatialOps
#
# The following variables are defined:
#
# SpatialOps_FOUND - true (1)
#
# SpatialOps_INCLUDE_DIR - include directories for SpatialOps
#
# Typical useage:
#    find_package( SpatialOps )
#    include_directories( ${SpatialOps_INCLUDE_DIR} )
#    target_link_libraries( ... spatialops )
#

set( SpatialOps_INCLUDE_DIR
  /home/alan/uintah-md/md-dev/dbg-test-all/Wasatch3P/install/SpatialOps/include
  /usr/local/boost/include
  )
set( Boost_INCLUDE /usr/local/boost/include )
set( Boost_LIB /usr/local/boost/lib/libboost_program_options.so )

set( ENABLE_THREADS OFF )
set( NTHREADS 1 )
set( ENABLE_CUDA OFF )

# CUDA Settings
if( ENABLE_CUDA )
  find_package( CUDA REQUIRED )
  set( CUDA_ARCHITECTURE              CACHE STRING "CUDA compute capability" FORCE )
  set( CUDA_NVCC_FLAGS                  CACHE STRING "Compiler flags passed to NVCC" FORCE)
  set( CUDA_NVCC_FLAGS_RELEASE  CACHE STRING "Release build compiler flags" FORCE )
endif( ENABLE_CUDA )

include( /home/alan/uintah-md/md-dev/dbg-test-all/Wasatch3P/install/SpatialOps/lib/spatialops/SpatialOps.cmake )
include( /home/alan/uintah-md/md-dev/dbg-test-all/Wasatch3P/install/SpatialOps/share/NeboBuildTools.cmake )
