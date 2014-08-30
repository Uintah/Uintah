# Install script for directory: /home/alan/uintah-md/md-dev/dbg-test-all/Wasatch3P/src/SpatialOps/spatialops

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
  FILE(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/spatialops" TYPE FILE PERMISSIONS OWNER_READ GROUP_READ WORLD_READ FILES
    "/home/alan/uintah-md/md-dev/dbg-test-all/Wasatch3P/src/SpatialOps/spatialops/Nebo.h"
    "/home/alan/uintah-md/md-dev/dbg-test-all/Wasatch3P/src/SpatialOps/spatialops/NeboBasic.h"
    "/home/alan/uintah-md/md-dev/dbg-test-all/Wasatch3P/src/SpatialOps/spatialops/NeboRhs.h"
    "/home/alan/uintah-md/md-dev/dbg-test-all/Wasatch3P/src/SpatialOps/spatialops/NeboMask.h"
    "/home/alan/uintah-md/md-dev/dbg-test-all/Wasatch3P/src/SpatialOps/spatialops/NeboOperators.h"
    "/home/alan/uintah-md/md-dev/dbg-test-all/Wasatch3P/src/SpatialOps/spatialops/NeboCond.h"
    "/home/alan/uintah-md/md-dev/dbg-test-all/Wasatch3P/src/SpatialOps/spatialops/NeboStencils.h"
    "/home/alan/uintah-md/md-dev/dbg-test-all/Wasatch3P/src/SpatialOps/spatialops/NeboStencilBuilder.h"
    "/home/alan/uintah-md/md-dev/dbg-test-all/Wasatch3P/src/SpatialOps/spatialops/NeboLhs.h"
    "/home/alan/uintah-md/md-dev/dbg-test-all/Wasatch3P/src/SpatialOps/spatialops/NeboAssignment.h"
    "/home/alan/uintah-md/md-dev/dbg-test-all/Wasatch3P/src/SpatialOps/spatialops/NeboReductions.h"
    "/home/alan/uintah-md/md-dev/dbg-test-all/Wasatch3P/src/SpatialOps/spatialops/FieldFunctions.h"
    "/home/alan/uintah-md/md-dev/dbg-test-all/Wasatch3P/src/SpatialOps/spatialops/OperatorDatabase.h"
    "/home/alan/uintah-md/md-dev/dbg-test-all/Wasatch3P/src/SpatialOps/spatialops/SpatialOpsDefs.h"
    "/home/alan/uintah-md/md-dev/dbg-test-all/Wasatch3P/src/SpatialOps/spatialops/SpatialOpsTools.h"
    "/home/alan/uintah-md/md-dev/dbg-test-all/Wasatch3P/src/SpatialOps/spatialops/Semaphore.h"
    )
ENDIF(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "Unspecified")

IF(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "Unspecified")
  FILE(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/spatialops" TYPE STATIC_LIBRARY PERMISSIONS OWNER_READ GROUP_READ WORLD_READ FILES "/home/alan/uintah-md/md-dev/dbg-test-all/Wasatch3P/src/SpatialOps/build/spatialops/libspatialops.a")
ENDIF(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "Unspecified")

IF(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "Unspecified")
  IF(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/spatialops/SpatialOps.cmake")
    FILE(DIFFERENT EXPORT_FILE_CHANGED FILES
         "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/spatialops/SpatialOps.cmake"
         "/home/alan/uintah-md/md-dev/dbg-test-all/Wasatch3P/src/SpatialOps/build/spatialops/CMakeFiles/Export/lib/spatialops/SpatialOps.cmake")
    IF(EXPORT_FILE_CHANGED)
      FILE(GLOB OLD_CONFIG_FILES "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/spatialops/SpatialOps-*.cmake")
      IF(OLD_CONFIG_FILES)
        MESSAGE(STATUS "Old export file \"$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/spatialops/SpatialOps.cmake\" will be replaced.  Removing files [${OLD_CONFIG_FILES}].")
        FILE(REMOVE ${OLD_CONFIG_FILES})
      ENDIF(OLD_CONFIG_FILES)
    ENDIF(EXPORT_FILE_CHANGED)
  ENDIF()
  FILE(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/spatialops" TYPE FILE FILES "/home/alan/uintah-md/md-dev/dbg-test-all/Wasatch3P/src/SpatialOps/build/spatialops/CMakeFiles/Export/lib/spatialops/SpatialOps.cmake")
  IF("${CMAKE_INSTALL_CONFIG_NAME}" MATCHES "^([Rr][Ee][Ll][Ee][Aa][Ss][Ee])$")
    FILE(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/spatialops" TYPE FILE FILES "/home/alan/uintah-md/md-dev/dbg-test-all/Wasatch3P/src/SpatialOps/build/spatialops/CMakeFiles/Export/lib/spatialops/SpatialOps-release.cmake")
  ENDIF("${CMAKE_INSTALL_CONFIG_NAME}" MATCHES "^([Rr][Ee][Ll][Ee][Aa][Ss][Ee])$")
ENDIF(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "Unspecified")

IF(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for each subdirectory.
  INCLUDE("/home/alan/uintah-md/md-dev/dbg-test-all/Wasatch3P/src/SpatialOps/build/spatialops/structured/cmake_install.cmake")
  INCLUDE("/home/alan/uintah-md/md-dev/dbg-test-all/Wasatch3P/src/SpatialOps/build/spatialops/particles/cmake_install.cmake")
  INCLUDE("/home/alan/uintah-md/md-dev/dbg-test-all/Wasatch3P/src/SpatialOps/build/spatialops/pointfield/cmake_install.cmake")

ENDIF(NOT CMAKE_INSTALL_LOCAL_ONLY)

