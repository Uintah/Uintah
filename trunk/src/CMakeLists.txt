# This is the root SCIRun CMakeLists file.
IF(WIN32)
  CMAKE_MINIMUM_REQUIRED(VERSION 2.4.4)

  # for the moment, force the user to select the build type here.  Maybe 
  # we can enable all types again if dynamic compilation goes away.
  # This needs to be before PROJECT, or changing the value will append to CMAKE_CONFIGURATION_TYPES, instead
  # of replacing it. - BW
  SET(BUILD_TYPE Release CACHE STRING "Choose the type of build, options are: None Debug Release RelWithDebInfo MinSizeRel.")
  SET(CMAKE_CONFIGURATION_TYPES ${BUILD_TYPE} CACHE STRING "Choose the type of build, options are: Debug Release RelWithDebInfo MinSizeRel." FORCE)
ELSE(WIN32)
  CMAKE_MINIMUM_REQUIRED(VERSION 2.2.3)
ENDIF(WIN32)

PROJECT(UINTAH)

#-----------------------------------------------------------------------------
# Test for some required system information.
INCLUDE (CMakeBackwardCompatibilityC)

#-----------------------------------------------------------------------------

SET(UINTAH_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR})

#----------------------------------------------------------------------------
# shared/static
OPTION(BUILD_SHARED_LIBS "Build shared libraries" ON)
MARK_AS_ADVANCED(BUILD_SHARED_LIBS)

#----------------------------------------------------------------------------
# Output directories.
SET (LIBRARY_OUTPUT_PATH ${UINTAH_BINARY_DIR}/lib CACHE INTERNAL "Single output directory for building all libraries.")
SET (EXECUTABLE_OUTPUT_PATH ${UINTAH_BINARY_DIR} CACHE INTERNAL "Single output directory for building all executables.")

MARK_AS_ADVANCED(LIBRARY_OUTPUT_PATH EXECUTABLE_OUTPUT_PATH)
#MARK_AS_ADVANCED(LIBRARY_OUTPUT_PATH EXECUTABLE_OUTPUT_PATH CXX_TEST_PATH)

SET(BUILD_FROM_TOP_LEVEL 0)

IF(WIN32)
  SET(DEF_ARCHES OFF)
  SET(DEF_RADIATION OFF)

  IF(NOT BUILD_SHARED_LIBS)
    ADD_DEFINITIONS(-DBUILD_CORE_STATIC)
    ADD_DEFINITIONS(-DGLEW_STATIC)
  ENDIF(NOT BUILD_SHARED_LIBS)
  #  This defines M_PI for windows
  ADD_DEFINITIONS(-D_USE_MATH_DEFINES)
  # defines Windows 2000 features like mouse leaving window
  ADD_DEFINITIONS(-D_WIN32_WINNT=0x0500)
  # turns off VS 2005 warnings about strcpy et.al. being unsafe
  ADD_DEFINITIONS(-D_CRT_SECURE_NO_DEPRECATE)
  
  # win32 l3 warnings are a little ridiculous
  SET(CMAKE_CXX_WARNING_LEVEL 1)
  IF(CMAKE_CXX_FLAGS MATCHES "/W[0-4]")
    STRING(REGEX REPLACE "/W[0-4]" "/W1"
            CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
  ELSE(CMAKE_CXX_FLAGS MATCHES "/W[0-4]")
    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /W1")
  ENDIF(CMAKE_CXX_FLAGS MATCHES "/W[0-4]")
ELSE(WIN32)
  SET(DEF_ARCHES ON)
  SET(DEF_RADIATION ON)
ENDIF(WIN32)

IF(NOT WIN32 AND NOT CMAKE_BUILD_TYPE)
  SET(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build, options are: None Debug Release RelWithDebInfo MinSizeRel." FORCE)
ENDIF(NOT WIN32 AND NOT CMAKE_BUILD_TYPE)

# Turn off Arches, ICE, MPM, or Radiation

OPTION(ARCHES "Build Arches and MPMArches" ${DEF_ARCHES})
OPTION(RADIATION "Build Radiation" ${DEF_RADIATION})
OPTION(ICE "Build ICE and MPMICE" ON)
OPTION(MPM "Build MPM, MPMArches and MPMICE" ON)

IF(NOT ARCHES)
  ADD_DEFINITIONS(-DNO_ARCHES)
ENDIF(NOT ARCHES)
IF(NOT ICE)
  ADD_DEFINITIONS(-DNO_ICE)
ENDIF(NOT ICE)
IF(NOT MPM)
  ADD_DEFINITIONS(-DNO_MPM)
ENDIF(NOT MPM)
IF(NOT RADIATION)
  ADD_DEFINITIONS(-DNO_RADIATION)
ENDIF(NOT RADIATION)


#-----------------------------------------------------------------------------
# SCIRun Packages to enable (must enable packages when building core for now)
# DV Fix need error checking to make sure SCIRun gets included
SET(LOAD_PACKAGE "SCIRun,Teem,Uintah" CACHE STRING "Building SCIRun, Teem, and Uintah packages by default.")


INCLUDE_DIRECTORIES(
  ${UINTAH_SOURCE_DIR}
  ${UINTAH_SOURCE_DIR}/SCIRun
  ${UINTAH_SOURCE_DIR}/SCIRun/externals
  ${UINTAH_SOURCE_DIR}/SCIRun/externals/glew/include
  ${UINTAH_SOURCE_DIR}/SCIRun/include
  ${CMAKE_CURRENT_BINARY_DIR}
  ${CMAKE_CURRENT_BINARY_DIR}/SCIRun/include
  ${XML2_INCLUDE}
)

IF(TEEM_INCLUDE)
  INCLUDE_DIRECTORIES(${TEEM_INCLUDE})
ENDIF(TEEM_INCLUDE)


IF(MPI_INCLUDE)
  INCLUDE_DIRECTORIES(${MPI_INCLUDE})
ENDIF(MPI_INCLUDE)

IF(ARCHES OR RADIATION)
  # fortran files
  ENABLE_LANGUAGE(Fortran)
  FIND_PROGRAM(PERL perl)
  SET(FORTRAN_UNDERSCORE end CACHE STRING "What type of fortran underscore style - end,none")
# TODO - if AIX, set UNDERSCORE to none
  SET(FSPEC ${CMAKE_CURRENT_BINARY_DIR}/genfspec.pl CACHE FILE "Path to fspec")
  CONFIGURE_FILE(${UINTAH_SOURCE_DIR}/tools/fspec.pl.in ${FSPEC})

  # Adds fortran file to list of srcs and creates the .h file for it
  MACRO(FORTRAN_DEPENDENCY fortran_file lib)
    # should pass in a fortran_file and package it belongs to.  I.e.,
    # FORTRAN_DEPENDENCY(fortran/bcscalar_fort.h CCA_Components_Arches)
    STRING(REPLACE .F _fort.h header ${CMAKE_CURRENT_BINARY_DIR}/${fortran_file})
    STRING(REPLACE .F .fspec fspec_file ${fortran_file})
    FILE(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/fortran)
    SET(${lib}_FORTRAN_SRCS ${${lib}_FORTRAN_SRCS} ${fortran_file})
    EXECUTE_PROCESS(COMMAND perl ${FSPEC} ${CMAKE_CURRENT_SOURCE_DIR}/${fspec_file} ${header})
  ENDMACRO(FORTRAN_DEPENDENCY)
ELSE(ARCHES OR RADIATION)
  MACRO(FORTRAN_DEPENDENCY fortran_file lib)
  ENDMACRO(FORTRAN_DEPENDENCY)
ENDIF(ARCHES OR RADIATION)


ADD_SUBDIRECTORY(SCIRun)

# grab the BUILD_DATAFLOW prop so we can use it to build Core_Datatypes
GET_DIRECTORY_PROPERTY( BUILD_DATAFLOW DIRECTORY SCIRun DEFINITION BUILD_DATAFLOW)

ADD_SUBDIRECTORY(Core)
ADD_SUBDIRECTORY(CCA)
ADD_SUBDIRECTORY(StandAlone)

#options

# scirun on
# sus on
# standalone utils on

# look for petsc
# look for hypre
# look for tau

