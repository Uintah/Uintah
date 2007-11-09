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

# compiler flags
IF(CXX_FLAGS)
  SET(CMAKE_CXX_FLAGS ${CXX_FLAGS} CACHE STRING "Default compiler flags" FORCE)
ENDIF(CXX_FLAGS)

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
  SET(DEFAULT_ARCHES ON)
  SET(DEFAULT_RADIATION ON)
ENDIF(WIN32)

IF(NOT WIN32 AND NOT CMAKE_BUILD_TYPE)
  SET(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build, options are: None Debug Release RelWithDebInfo MinSizeRel." FORCE)
ENDIF(NOT WIN32 AND NOT CMAKE_BUILD_TYPE)

# Turn off Arches, ICE, MPM, or Radiation

SET(DEF_ARCHES "")
SET(DEF_ICE "")
SET(DEF_MPM "")
SET(DEF_RADIATION "")

OPTION(ARCHES "Build Arches and MPMArches" ${DEFAULT_ARCHES})
OPTION(RADIATION "Build Radiation" ${DEFAULT_RADIATION})
OPTION(ICE "Build ICE and MPMICE" ON)
OPTION(MPM "Build MPM, MPMArches and MPMICE" ON)

IF(NOT ARCHES)
  SET(DEF_ARCHES "#define NO_ARCHES")
ENDIF(NOT ARCHES)
IF(NOT ICE)
  SET(DEF_ICE "#define NO_ICE")
ENDIF(NOT ICE)
IF(NOT MPM)
  SET(DEF_MPM "#define NO_MPM")
ENDIF(NOT MPM)
IF(NOT RADIATION)
  SET(DEF_RADIATION "#define NO_RADIATION")
ENDIF(NOT RADIATION)


IF(ARCHES OR RADIATION)
  # fortran files
  ENABLE_LANGUAGE(Fortran)

  STRING(REGEX MATCH gfortran HAVE_GFORTRAN ${CMAKE_Fortran_COMPILER})
  STRING(REGEX MATCH xlf HAVE_XLF ${CMAKE_Fortran_COMPILER})
  STRING(REGEX MATCH pg77 HAVE_PG77 ${CMAKE_Fortran_COMPILER})
  STRING(REGEX MATCH g77 HAVE_G77 ${CMAKE_Fortran_COMPILER})
  STRING(REGEX MATCH ifort HAVE_IFORT ${CMAKE_Fortran_COMPILER})
  STRING(REGEX MATCH f77 HAVE_F77 ${CMAKE_Fortran_COMPILER})

  # tried to do 'elseif' but cmake wouldn't let me --bjw
  IF(HAVE_GFORTRAN)
    FIND_LIBRARY(F_LIBRARY NAMES gfortran)
    SET(FORTRAN_UNDERSCORE end CACHE STRING "What type of fortran underscore style - linux,end,none")
  ENDIF(HAVE_GFORTRAN)
  IF(HAVE_XLF)
    FIND_LIBRARY(F_LIBRARY NAMES xlf90)
    SET(FORTRAN_UNDERSCORE none CACHE STRING "What type of fortran underscore style - linux,end,none")
  ENDIF(HAVE_XLF)
  IF(HAVE_PGF77)
    FIND_LIBRARY(F_LIBRARY NAMES pgftnrtl)
    SET(FORTRAN_UNDERSCORE end CACHE STRING "What type of fortran underscore style - linux,end,none")
  ENDIF(HAVE_PGF77)
  IF(HAVE_G77)
    FIND_LIBRARY(F_LIBRARY NAMES g2c)
    SET(FORTRAN_UNDERSCORE linux CACHE STRING "What type of fortran underscore style - linux,end,none")
  ENDIF(HAVE_G77)
  IF(HAVE_ifort)
    FIND_LIBRARY(F_LIBRARY NAMES ifcore)
    SET(FORTRAN_UNDERSCORE end CACHE STRING "What type of fortran underscore style - linux,end,none")
  ENDIF(HAVE_ifort)
  IF(HAVE_F77)
    # sgi needs ftn fortran
    #IF(CMAKE_SYSTEM MATCHES "IRIX.*")
    #SET(FORTRAN_UNDERSCORE end CACHE STRING "What type of fortran underscore style - linux,end,none")
    # dec needs for Ufor
    #IF(CMAKE_SYSTEM MATCHES "OSF.*")
    # FIND_LIBRARY(F_LIBRARY NAMES ........)
    #SET(FORTRAN_UNDERSCORE end CACHE STRING "What type of fortran underscore style - linux,end,none")
  ENDIF(HAVE_F77)

  IF(NOT F_LIBRARY)
    MESSAGE(FATAL_ERROR "Cannot find fortran library")
  ENDIF(NOT F_LIBRARY)

  FIND_PROGRAM(PERL perl)
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
    # if everybody has cmake 2.4.5, we can do IS_NEWER_THAN, and not create these files on every cmake
    #IF(${CMAKE_CURRENT_SOURCE_DIR}/${fspec_file} IS_NEWER_THAN ${header})
    IF(NOT EXISTS ${header})
      EXECUTE_PROCESS(COMMAND perl ${FSPEC} ${CMAKE_CURRENT_SOURCE_DIR}/${fspec_file} ${header})
    ENDIF(NOT EXISTS ${header})
    #ENDIF(${CMAKE_CURRENT_SOURCE_DIR}/${fspec_file} IS_NEWER_THAN ${header})

  ENDMACRO(FORTRAN_DEPENDENCY)
ELSE(ARCHES OR RADIATION)
  MACRO(FORTRAN_DEPENDENCY fortran_file lib)
  ENDMACRO(FORTRAN_DEPENDENCY)
ENDIF(ARCHES OR RADIATION)

SET(MPI_DIR "" CACHE PATH "Path to MPI")

#-----------------------------------------------------------------------------
# SCIRun Packages to enable (must enable packages when building core for now)
SET(LOAD_PACKAGE "SCIRun,Teem,Uintah" CACHE STRING "Building SCIRun, Teem, and Uintah packages by default.")
SET(HANDLED_PACKAGE_Uintah "YES")

ADD_SUBDIRECTORY(SCIRun)

INCLUDE_DIRECTORIES(
  ${PETSC_INCLUDES}
  ${HYPRE_INCLUDE}
  ${XML2_INCLUDE}
)

# grab the BUILD_DATAFLOW prop so we can use it to build Core_Datatypes
GET_DIRECTORY_PROPERTY( ITK_FOUND DIRECTORY SCIRun DEFINITION ITK_FOUND)
GET_DIRECTORY_PROPERTY( SCIRUN_CORE_SOURCE_DIR DIRECTORY SCIRun DEFINITION SCIRUN_CORE_SOURCE_DIR)
GET_DIRECTORY_PROPERTY( SCIRUN_DATAFLOW_BINARY_DIR DIRECTORY SCIRun DEFINITION SCIRUN_DATAFLOW_BINARY_DIR)
GET_DIRECTORY_PROPERTY( TCLSH DIRECTORY SCIRun/Dataflow DEFINITION TCLSH)
GET_DIRECTORY_PROPERTY( TCL_INIT_DIR DIRECTORY SCIRun DEFINITION TCL_INIT_DIR)
GET_DIRECTORY_PROPERTY( ITCL_INIT_DIR SCIRun DEFINITION ITCL_INIT_DIR)
GET_DIRECTORY_PROPERTY( CREATE_TCL_INDEX DIRECTORY SCIRun/Dataflow DEFINITION CREATE_TCL_INDEX)

IF(ITK_FOUND)
  GET_DIRECTORY_PROPERTY( ITK_USE_FILE DIRECTORY SCIRun DEFINITION ITK_USE_FILE)
  GET_DIRECTORY_PROPERTY( ITK_LIBRARY_DIRS DIRECTORY SCIRun DEFINITION ITK_LIBRARY_DIRS)
  # If SCIRun uses ITK, Uintah/Core/Datatypes will inadvertently want it as well
  INCLUDE(${ITK_USE_FILE})
ENDIF(ITK_FOUND)


# grab the BUILD_DATAFLOW prop so we can use it to build Core_Datatypes
GET_DIRECTORY_PROPERTY( BUILD_DATAFLOW DIRECTORY SCIRun DEFINITION BUILD_DATAFLOW)

IF(NOT MPI_LIBS)
  MESSAGE(FATAL_ERROR "MPI Required!  Specify MPI_DIR=<path-to-mpi>")
ENDIF(NOT MPI_LIBS)

IF(TEEM_INCLUDE)
  INCLUDE_DIRECTORIES(${TEEM_INCLUDE})
ENDIF(TEEM_INCLUDE)


IF(MPI_INCLUDE)
  INCLUDE_DIRECTORIES(${MPI_INCLUDE})
ENDIF(MPI_INCLUDE)

INCLUDE_DIRECTORIES(
  ${UINTAH_SOURCE_DIR}
  ${UINTAH_SOURCE_DIR}/SCIRun
  ${UINTAH_SOURCE_DIR}/SCIRun/externals
  ${UINTAH_SOURCE_DIR}/SCIRun/externals/glew/include
  ${UINTAH_SOURCE_DIR}/SCIRun/include
  ${CMAKE_CURRENT_BINARY_DIR}
  ${CMAKE_CURRENT_BINARY_DIR}/SCIRun/include
)

ADD_SUBDIRECTORY(Core)
ADD_SUBDIRECTORY(CCA)
ADD_SUBDIRECTORY(StandAlone)

IF(BUILD_DATAFLOW)
  ADD_SUBDIRECTORY(Dataflow)

ENDIF(BUILD_DATAFLOW)

# look for tau

