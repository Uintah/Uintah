# Install script for directory: /home/alan/uintah-md/md-dev/dbg-test-all/Wasatch3P/src/ExprLib/expression

# Set the install prefix
IF(NOT DEFINED CMAKE_INSTALL_PREFIX)
  SET(CMAKE_INSTALL_PREFIX "/home/alan/uintah-md/md-dev/dbg-test-all/Wasatch3P/install/ExprLib")
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
  FILE(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/expression" TYPE FILE PERMISSIONS OWNER_READ GROUP_READ WORLD_READ FILES
    "/home/alan/uintah-md/md-dev/dbg-test-all/Wasatch3P/src/ExprLib/expression/ExprFwd.h"
    "/home/alan/uintah-md/md-dev/dbg-test-all/Wasatch3P/src/ExprLib/expression/ExprDeps.h"
    "/home/alan/uintah-md/md-dev/dbg-test-all/Wasatch3P/src/ExprLib/expression/Context.h"
    "/home/alan/uintah-md/md-dev/dbg-test-all/Wasatch3P/src/ExprLib/expression/Expression.h"
    "/home/alan/uintah-md/md-dev/dbg-test-all/Wasatch3P/src/ExprLib/expression/ExpressionID.h"
    "/home/alan/uintah-md/md-dev/dbg-test-all/Wasatch3P/src/ExprLib/expression/ExpressionBase.h"
    "/home/alan/uintah-md/md-dev/dbg-test-all/Wasatch3P/src/ExprLib/expression/ExpressionFactory.h"
    "/home/alan/uintah-md/md-dev/dbg-test-all/Wasatch3P/src/ExprLib/expression/ExpressionTree.h"
    "/home/alan/uintah-md/md-dev/dbg-test-all/Wasatch3P/src/ExprLib/expression/FieldDeps.h"
    "/home/alan/uintah-md/md-dev/dbg-test-all/Wasatch3P/src/ExprLib/expression/FieldManagerBase.h"
    "/home/alan/uintah-md/md-dev/dbg-test-all/Wasatch3P/src/ExprLib/expression/FieldManagerList.h"
    "/home/alan/uintah-md/md-dev/dbg-test-all/Wasatch3P/src/ExprLib/expression/Functions.h"
    "/home/alan/uintah-md/md-dev/dbg-test-all/Wasatch3P/src/ExprLib/expression/GraphType.h"
    "/home/alan/uintah-md/md-dev/dbg-test-all/Wasatch3P/src/ExprLib/expression/HybridScheduler.h"
    "/home/alan/uintah-md/md-dev/dbg-test-all/Wasatch3P/src/ExprLib/expression/ManagerTypes.h"
    "/home/alan/uintah-md/md-dev/dbg-test-all/Wasatch3P/src/ExprLib/expression/Poller.h"
    "/home/alan/uintah-md/md-dev/dbg-test-all/Wasatch3P/src/ExprLib/expression/PriorityScheduler.h"
    "/home/alan/uintah-md/md-dev/dbg-test-all/Wasatch3P/src/ExprLib/expression/PropertyStash.h"
    "/home/alan/uintah-md/md-dev/dbg-test-all/Wasatch3P/src/ExprLib/expression/Schedulers.h"
    "/home/alan/uintah-md/md-dev/dbg-test-all/Wasatch3P/src/ExprLib/expression/SchedulerBase.h"
    "/home/alan/uintah-md/md-dev/dbg-test-all/Wasatch3P/src/ExprLib/expression/SourceExprOp.h"
    "/home/alan/uintah-md/md-dev/dbg-test-all/Wasatch3P/src/ExprLib/expression/Tag.h"
    "/home/alan/uintah-md/md-dev/dbg-test-all/Wasatch3P/src/ExprLib/expression/TransportEquation.h"
    "/home/alan/uintah-md/md-dev/dbg-test-all/Wasatch3P/src/ExprLib/expression/VertexProperty.h"
    )
ENDIF(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "Unspecified")

IF(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for each subdirectory.
  INCLUDE("/home/alan/uintah-md/md-dev/dbg-test-all/Wasatch3P/src/ExprLib/build/expression/uintah/cmake_install.cmake")

ENDIF(NOT CMAKE_INSTALL_LOCAL_ONLY)

