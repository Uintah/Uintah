# Install script for directory: /home/alan/uintah-md/md-dev/dbg-test-all/Wasatch3P/src/SpatialOps/spatialops/structured

# Set the install prefix
IF(NOT DEFINED CMAKE_INSTALL_PREFIX)
  SET(CMAKE_INSTALL_PREFIX "/home/alan/uintah-md/md-dev/dbg-test-all/Wasatch3P/install/SpatialOps")
ENDIF(NOT DEFINED CMAKE_INSTALL_PREFIX)
STRING(REGEX REPLACE "/$" "" CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")

# Set the install configuration name.
IF(NOT DEFINED CMAKE_INSTALL_CONFIG_NAME)
  IF(BUILD_TYPE)
    STRING(REGEX REPLACE "^[^A-Za-z0-9_]+" ""
           CMAKE_INSTALL_CONFIG_NAME "${BUILD_TYPE}")
  ELSE(BUILD_TYPE)
    SET(CMAKE_INSTALL_CONFIG_NAME "Release")
  ENDIF(BUILD_TYPE)
  MESSAGE(STATUS "Install configuration: \"${CMAKE_INSTALL_CONFIG_NAME}\"")
ENDIF(NOT DEFINED CMAKE_INSTALL_CONFIG_NAME)

# Set the component getting installed.
IF(NOT CMAKE_INSTALL_COMPONENT)
  IF(COMPONENT)
    MESSAGE(STATUS "Install component: \"${COMPONENT}\"")
    SET(CMAKE_INSTALL_COMPONENT "${COMPONENT}")
  ELSE(COMPONENT)
    SET(CMAKE_INSTALL_COMPONENT)
  ENDIF(COMPONENT)
ENDIF(NOT CMAKE_INSTALL_COMPONENT)

# Install shared libraries without execute permission?
IF(NOT DEFINED CMAKE_INSTALL_SO_NO_EXE)
  SET(CMAKE_INSTALL_SO_NO_EXE "1")
ENDIF(NOT DEFINED CMAKE_INSTALL_SO_NO_EXE)

IF(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "Unspecified")
  FILE(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/spatialops/structured" TYPE FILE PERMISSIONS OWNER_READ GROUP_READ WORLD_READ FILES
    "/home/alan/uintah-md/md-dev/dbg-test-all/Wasatch3P/src/SpatialOps/spatialops/structured/BitField.h"
    "/home/alan/uintah-md/md-dev/dbg-test-all/Wasatch3P/src/SpatialOps/spatialops/structured/BoundaryCellInfo.h"
    "/home/alan/uintah-md/md-dev/dbg-test-all/Wasatch3P/src/SpatialOps/spatialops/structured/CudaMemoryAllocator.h"
    "/home/alan/uintah-md/md-dev/dbg-test-all/Wasatch3P/src/SpatialOps/spatialops/structured/DoubleVec.h"
    "/home/alan/uintah-md/md-dev/dbg-test-all/Wasatch3P/src/SpatialOps/spatialops/structured/ExternalAllocators.h"
    "/home/alan/uintah-md/md-dev/dbg-test-all/Wasatch3P/src/SpatialOps/spatialops/structured/FieldComparisons.h"
    "/home/alan/uintah-md/md-dev/dbg-test-all/Wasatch3P/src/SpatialOps/spatialops/structured/FieldHelper.h"
    "/home/alan/uintah-md/md-dev/dbg-test-all/Wasatch3P/src/SpatialOps/spatialops/structured/FieldInfo.h"
    "/home/alan/uintah-md/md-dev/dbg-test-all/Wasatch3P/src/SpatialOps/spatialops/structured/FVStaggeredBCTools.h"
    "/home/alan/uintah-md/md-dev/dbg-test-all/Wasatch3P/src/SpatialOps/spatialops/structured/FVStaggered.h"
    "/home/alan/uintah-md/md-dev/dbg-test-all/Wasatch3P/src/SpatialOps/spatialops/structured/FVStaggeredLocationTypes.h"
    "/home/alan/uintah-md/md-dev/dbg-test-all/Wasatch3P/src/SpatialOps/spatialops/structured/FVStaggeredFieldTypes.h"
    "/home/alan/uintah-md/md-dev/dbg-test-all/Wasatch3P/src/SpatialOps/spatialops/structured/Grid.h"
    "/home/alan/uintah-md/md-dev/dbg-test-all/Wasatch3P/src/SpatialOps/spatialops/structured/GhostData.h"
    "/home/alan/uintah-md/md-dev/dbg-test-all/Wasatch3P/src/SpatialOps/spatialops/structured/IndexTriplet.h"
    "/home/alan/uintah-md/md-dev/dbg-test-all/Wasatch3P/src/SpatialOps/spatialops/structured/IntVec.h"
    "/home/alan/uintah-md/md-dev/dbg-test-all/Wasatch3P/src/SpatialOps/spatialops/structured/Numeric3Vec.h"
    "/home/alan/uintah-md/md-dev/dbg-test-all/Wasatch3P/src/SpatialOps/spatialops/structured/MemoryPool.h"
    "/home/alan/uintah-md/md-dev/dbg-test-all/Wasatch3P/src/SpatialOps/spatialops/structured/MemoryTypes.h"
    "/home/alan/uintah-md/md-dev/dbg-test-all/Wasatch3P/src/SpatialOps/spatialops/structured/MemoryWindow.h"
    "/home/alan/uintah-md/md-dev/dbg-test-all/Wasatch3P/src/SpatialOps/spatialops/structured/SpatialField.h"
    "/home/alan/uintah-md/md-dev/dbg-test-all/Wasatch3P/src/SpatialOps/spatialops/structured/SpatialFieldStore.h"
    "/home/alan/uintah-md/md-dev/dbg-test-all/Wasatch3P/src/SpatialOps/spatialops/structured/SpatialMask.h"
    )
ENDIF(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "Unspecified")

IF(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "Unspecified")
  FILE(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/spatialops" TYPE STATIC_LIBRARY PERMISSIONS OWNER_READ GROUP_READ WORLD_READ FILES "/home/alan/uintah-md/md-dev/dbg-test-all/Wasatch3P/src/SpatialOps/build/spatialops/structured/libspatialops-structured.a")
ENDIF(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "Unspecified")

IF(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for each subdirectory.
  INCLUDE("/home/alan/uintah-md/md-dev/dbg-test-all/Wasatch3P/src/SpatialOps/build/spatialops/structured/onedim/cmake_install.cmake")
  INCLUDE("/home/alan/uintah-md/md-dev/dbg-test-all/Wasatch3P/src/SpatialOps/build/spatialops/structured/stencil/cmake_install.cmake")

ENDIF(NOT CMAKE_INSTALL_LOCAL_ONLY)

