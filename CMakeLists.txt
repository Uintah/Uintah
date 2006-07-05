# This is the root ITK CMakeLists file.
CMAKE_MINIMUM_REQUIRED(VERSION 2.0)

PROJECT(SCIRUN)

#-----------------------------------------------------------------------------
# SCIRUN version number.  Must reflect the value of SCIRUN_RCFILE_VERSION 
# in src/scirunrc
SET(SCIRUN_VERSION_MAJOR "1")
SET(SCIRUN_VERSION_MINOR "25")
SET(SCIRUN_VERSION_PATCH "3")

# Version string should not include patch level.  The major.minor is
# enough to distinguish available features of the toolkit.
SET(SCIRUN_VERSION_STRING "${SCIRUN_VERSION_MAJOR}.${SCIRUN_VERSION_MINOR}.${SCIRUN_VERSION_PATCH}")

#-----------------------------------------------------------------------------
# Configure Dart testing support.  This should be done before any
# MESSAGE(FATAL_ERROR ...) commands are invoked.
INCLUDE(${CMAKE_ROOT}/Modules/Dart.cmake)
MARK_AS_ADVANCED(TCL_TCLSH DART_ROOT)
IF(BUILD_TESTING)
  ENABLE_TESTING()
  SET(BUILDNAME "${BUILDNAME}" CACHE STRING "Name of build on the dashboard")
  MARK_AS_ADVANCED(BUILDNAME)
ENDIF(BUILD_TESTING)

#----------------------------------------------------------------------------
# Output directories.
SET (LIBRARY_OUTPUT_PATH ${CMAKE_BINARY_DIR}/bin CACHE INTERNAL "Single output directory for building all libraries.")
SET (EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR}/bin CACHE INTERNAL "Single output directory for building all executables.")
#output directory for installing all include files.")
MARK_AS_ADVANCED(LIBRARY_OUTPUT_PATH EXECUTABLE_OUTPUT_PATH)

# ----------------------------------------------------------------------------
# SCIRun build configuration options
OPTION(BUILD_SHARED_LIBS "Build SCIRun with shared libraries." OFF)
SET(SCIRUN_BUILD_SHARED_LIBS ${BUILD_SHARED_LIBS})

IF(WIN32)
  IF(NOT BUILD_SHARED_LIBS)
    ADD_DEFINITIONS(-DBUILD_STATIC)
  ENDIF(NOT BUILD_SHARED_LIBS)
ENDIF(WIN32)

#-----------------------------------------------------------------------------
# SCIRUN build configuration options.
# OPTION(ENABLE_SCI_MALLOC "Use SCI memory allocation/deallocation checks." OFF)
# SET(SCIRUN_ENABLE_SCI_MALLOC ${ENABLE_SCI_MALLOC})

# sci_defs/bits_defs.h
# DV clean up (if user tries to configure 64 bit machine to 32, flag error?)
# DV also maybe need to add binary flags here
# Set NBITS (64 or 32) based on the size of a void pointer.
IF(CMAKE_SIZEOF_VOID_P MATCHES  8)
  SET(NBITS 64 CACHE INTERNAL "64 or 32 bit build")
ELSE(CMAKE_SIZEOF_VOID_P MATCHES 8)
  SET(NBITS 32 CACHE INTERNAL "64 or 32 bit build")
ENDIF(CMAKE_SIZEOF_VOID_P MATCHES 8)
SET (SCIRUN_ENABLE_64BIT 0 CACHE BOOL "Enable 64 bit build.")
IF(${SCIRUN_ENABLE_64BIT})
  SET(NBITS 64 CACHE INTERNAL "64 bit build")
ENDIF(${SCIRUN_ENABLE_64BIT})
MARK_AS_ADVANCED(SCIRUN_ENABLE_64BIT)

#-----------------------------------------------------------------------------
# SCIRun Thirdparty location
INCLUDE(${PROJECT_SOURCE_DIR}/CMake/sciThirdparty.cmake)


#DV FIX HARD CODED VERSION
SET(COMPILER_VERS unknown)

IF(CMAKE_COMPILER_IS_GNUCXX)
  EXEC_PROGRAM(${CMAKE_CXX_COMPILER} ARGS --version OUTPUT_VARIABLE CMAKE_CXX_COMPILER_VERSION)
  IF(CMAKE_CXX_COMPILER_VERSION)
    SET(COMPILER_VERS ${CMAKE_CXX_COMPILER_VERSION})
  ENDIF(CMAKE_CXX_COMPILER_VERSION)
ENDIF(CMAKE_COMPILER_IS_GNUCXX) 

SCIRUN_THIRDPARTY_OPTION( 
  "${SCIRUN_VERSION_MAJOR}.${SCIRUN_VERSION_MINOR}.${SCIRUN_VERSION_PATCH}"
  ${COMPILER_VERS}
  ${NBITS}
)

IF(SCIRUN_THIRDPARTY_FOUND)
  SET(SCIRUN_INCLUDE_THIRDPARTY ${SCIRUN_THIRDPARTY_PATH}/include)	
ELSE(SCIRUN_THIRDPARTY_FOUND)
  MESSAGE(FATAL_ERROR
    "Cannot build SCIRun without Thirdparty. Please set SCIRUN_THIRDPARTY_PATH.")
ENDIF(SCIRUN_THIRDPARTY_FOUND)


# sci_defs/compile_defs.h
# DV Not needed eventually?
IF(WIN32 AND NOT CYGWIN)
  SET (PATH_TO_VC "c:/Program Files/Microsoft Visual Studio .NET 2003/Vc7" CACHE PATH "Path to Visual Studio")
  SET (PATH_TO_PSDK "c:/Program Files/Microsoft Visual Studio .NET 2003/Vc7/PlatformSDK" CACHE PATH "Path to PSDK")
  SET (PATH_TO_MSYS_BIN "c:/msys/1.0/bin" CACHE PATH "Path to Msys bin directory")
ENDIF(WIN32 AND NOT CYGWIN)

#-----------------------------------------------------------------------------
# SCIRun Packages to enable
# DV fix enabling more packages
SET (PACKAGES "" CACHE STRING "SCIRun packages to enable (separated by white space).")

# sci_defs/environment_defs.h
SET(scirun_src_dir "${CMAKE_SOURCE_DIR}/src" CACHE INTERNAL "Path to SCIRun's src directory")
SET(scirun_obj_dir ${CMAKE_BINARY_DIR} CACHE INTERNAL "Path to SCIRun's binary directory")
# DV Fix iterating over packages in case no packages selected, there 
# won't be dangling comma
IF(PACKAGES)
  SET(LOAD_PACKAGE "SCIRun," CACHE STRING "Packages to load")
  FOREACH(p ${PACKAGES})
    SET(LOAD_PACKAGE "${LOAD_PACKAGE},${p}" CACHE STRING "Packages to load")
  ENDFOREACH(p)
ELSE(PACKAGES)
  SET(LOAD_PACKAGE "SCIRun" CACHE STRING "Just loading SCIRun")
ENDIF(PACKAGES)
SET(ITCL_WIDGETS ${SCI_THIRDPARTY_DIR}/iwidgets/scripts CACHE INTERNAL "Path to iwidgets scripts")
# DV maybe this shouldn't be alterable?
MARK_AS_ADVANCED(LOAD_PACKAGE)
SET(LIB_DIR_ITCL ${SCI_THIRDPARTY_DIR})

# sci_defs/error_defs.h
SET(THROW_LEVEL 0 CACHE STRING "Set throw level N(0-3)")
SET(ASSERTION_LEVEL 3 CACHE STRING "Set assertion level N(0-3)")

# sci_defs/lapack_defs.h
# DV FIX hard coding of lapack library
SET(DEF_LAPACK "#define HAVE_LAPACK 0" CACHE INTERNAL "Have Lapack")

# sci_defs/hashmap_defs.h
SET(DEF_HASHMAP "None" CACHE STRING "Determining hash map")

# DV Fix to figure out value of DEF_HASHMAP
TRY_COMPILE(DEF_HASHMAP
  ${PROJECT_BINARY_DIR}/CMakeTmp
  ${PROJECT_SOURCE_DIR}/CMake/SCIRunTestHashMap.cc
  COMPILE_DEFINITIONS
  -DSCIRUN_HASHMAP_TYPE=std::hash_map<int, int>
  OUTPUT_VARIABLE OUTPUT)
IF("${DEF_HASHMAP}" STREQUAL "TRUE")
  SET(DEF_HASHMAP "#define HAVE_STD_HASHMAP 1" CACHE INTERNAL "Using STD hash map")
ELSE("${DEF_HASHMAP}" STREQUAL "TRUE")
  TRY_COMPILE(DEF_HASHMAP 
  ${PROJECT_BINARY_DIR}/CMakeTmp
  ${PROJECT_SOURCE_DIR}/CMake/SCIRunTestHashMap.cc
  COMPILE_DEFINITIONS
  -DSCIRUN_HASHMAP_TYPE=std::hash_map<int, int>
  OUTPUT_VARIABLE OUTPUT)
  # include <ext/hash_map> instead
ENDIF("${DEF_HASHMAP}" STREQUAL "TRUE")

IF("${DEF_HASHMAP}" STREQUAL "TRUE")
  SET(DEF_HASHMAP "#define HAVE_EXT_HASHMAP 1" CACHE INTERNAL "Using EXT hash map")
ELSE("${DEF_HASHMAP}" STREQUAL "TRUE")
  TRY_COMPILE(DEF_HASHMAP 
  ${PROJECT_BINARY_DIR}/CMakeTmp
  ${PROJECT_SOURCE_DIR}/CMake/SCIRunTestHashMap.cc
  COMPILE_DEFINITIONS
  -DSCIRUN_HASHMAP_TYPE=stdext::hash_map<int, int>
  OUTPUT_VARIABLE OUTPUT)
ENDIF("${DEF_HASHMAP}" STREQUAL "TRUE")

IF("${DEF_HASHMAP}" STREQUAL "TRUE")
  SET(DEF_HASHMAP "#define HAVE_STDEXT_HASHMAP 1" CACHE INTERNAL "Using STD EXT hash map")
ELSE("${DEF_HASHMAP}" STREQUAL "TRUE")
  TRY_COMPILE(DEF_HASHMAP 
  ${PROJECT_BINARY_DIR}/CMakeTmp
  ${PROJECT_SOURCE_DIR}/CMake/SCIRunTestHashMap.cc
  COMPILE_DEFINITIONS
  -DSCIRUN_HASHMAP_TYPE=__gnu_css::hash_map<int, int>
  OUTPUT_VARIABLE OUTPUT)
  # include <ext/hash_map> instead
ENDIF("${DEF_HASHMAP}" STREQUAL "TRUE")

IF("${DEF_HASHMAP}" STREQUAL "TRUE")
  SET(DEF_HASHMAP "#define HAVE_GNU_HASHMAP 1" CACHE INTERNAL "Using GNU hash map")
ELSE("${DEF_HASHMAP}" STREQUAL "TRUE")
  MESSAGE(STATUS "Cannot find hash_map, will use map instead") 
ENDIF("${DEF_HASHMAP}" STREQUAL "TRUE")


# DV Figure out how to make these not even configurable
# sci_defs/template_defs.h
SET(TEMPLATE_TAG " " CACHE STRING "NOT SURE")
# DV This should be an empty string on some platforms
SET(TEMPLATE_BOX "<>" CACHE STRING "NOT SURE")

MARK_AS_ADVANCED(THROW_LEVEL ASSERTION_LEVEL DEF_HASH TEMPLATE_TAG TEMPLATE_BOX)
# DV Add with_exc stuff from configure.ac

INCLUDE (${CMAKE_ROOT}/Modules/CheckIncludeFile.cmake)
CHECK_INCLUDE_FILE("limits.h"     HAVE_LIMITS_H)
CHECK_INCLUDE_FILE("sys/select.h"     HAVE_SYS_SELECT_H)
CHECK_INCLUDE_FILE("uinstd.h"     HAVE_UNISTD_H)
CHECK_INCLUDE_FILE("sys/time.h"     HAVE_SYS_TIME_H)
# DV Fix this one (ext/algorithm)
CHECK_INCLUDE_FILE("ext/algorithm"     HAVE_EXT_ALGORITHM_H)
CHECK_INCLUDE_FILE("stdint.h"     HAVE_STDINT_H)
CHECK_INCLUDE_FILE("inttypes.h"     HAVE_INTTYPES_H)
SET(HAVE_SYS_UUID_H 0)
SET(HAVE_UUID_UUID_H 0)


#-----------------------------------------------------------------------------
# Thread library
FIND_PACKAGE( Threads )
SET(THREAD_LIBRARY ${CMAKE_THREAD_LIBS_INIT} CACHE INTERNAL "Thread library.")
# DV Check for sempahore libs w/in thread library

#-----------------------------------------------------------------------------
# Search for SGI's or Win32's traceback library
# traceback_lib exc_lib_dir_flag exc_lib_flag
# DV Make required and look for it, also fix for SGI
INCLUDE (${CMAKE_ROOT}/Modules/CheckIncludeFile.cmake)
IF(WIN32)
  CHECK_INCLUDE_FILE("windows.h"       HAVE_WINDOWS_H)
  CHECK_INCLUDE_FILE("imagehlp"       HAVE_IMAGEHLP_H)
  IF(HAVE_WINDOWS_H)
      IF(HAVE_IMAGEHLP_H)
        SET(TRACEBACK_LIB imagehlp psapi CACHE INTERNAL "Trace back library.")
        SET(HAVE_EXC 1)
        SET(DEF_EXC "#define HAVE_EXC 1")
      ELSE(HAVE_IMAGEHELP_H)
        MESSAGE(SEND_ERROR "Trace back library requires imagehlp.h for Windows.")
      ENDIF(HAVE_IMAGEHLP_H)
  ELSE(HAVE_WINDOWS_H)
     MESSAGE(SEND_ERROR "Trace back library requires windows.h for Windows.")
  ENDIF(HAVE_WINDOWS_H)
ELSE(WIN32)
  CHECK_INCLUDE_FILE("libexc.h"       HAVE_EXC_H)
  IF(HAVE_EXC_H)
    SET(TRACEBACK_LIB exc CACHE INTERNAL "Trace back library.")
    SET(HAVE_EXC 1)
    SET(DEF_EXC "#define HAVE_EXC 1")
  ENDIF(HAVE_EXC_H)
ENDIF(WIN32)


#-----------------------------------------------------------------------------
# Z lib
IF(WIN32 AND NOT CYGWIN)
  SET(ZLIB_LIBRARY ${SCI_THIRDPARTY_DIR}/lib/z)	 
  SET(ZLIB_INCLUDE ${SCI_THIRDPARTY_DIR}/include)	 
ELSE(WIN32 AND NOT CYGWIN)
  FIND_PACKAGE( ZLIB )
ENDIF(WIN32 AND NOT CYGWIN)
SET(SCIRUN_INCLUDE_THIRDPARTY ${SCIRUN_INCLUDE_THIRDPARTY} ${ZLIB_INCLUDE_DIR})

#-----------------------------------------------------------------------------
# Teem
SET (CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/CMake)
FIND_PACKAGE( Teem )
IF (TEEM_LIBRARY)
  IF (TEEM_INCLUDE_DIR)
    SET(SCIRUN_INCLUDE_THIRDPARTY ${SCIRUN_INCLUDE_THIRDPARTY} ${TEEM_INCLUDE_DIR})
  ENDIF(TEEM_INCLUDE_DIR)
ENDIF(TEEM_LIBRARY)
SET(DEF_GDCM "#define HAVE_GDCM 0")


#-----------------------------------------------------------------------------
# The entire SCIRun tree should use the same include path

# This should be the only INCLUDE_DIRECTORIES command in the entire
# tree.  We need to do this in one place to make sure the order is correct.
#-----------------------------------------------------------------------------
# Include directories for other projects installed on the system.
SET(SCIRUN_INCLUDE_DIRS_SYSTEM 
  ${PROJECT_SOURCE_DIR}/src/
  ${PROJECT_SOURCE_DIR}/src/include/
)

# Set include directories for sci_defs
SET(SCIRUN_INCLUDE_DIRS_BUILD_TREE 
  ${PROJECT_BINARY_DIR}/include/
  ${PROJECT_BINARY_DIR}/include/sci_defs/
)

INCLUDE_DIRECTORIES(
  ${SCIRUN_INCLUDE_DIRS_SYSTEM}
  ${SCIRUN_INCLUDE_DIRS_BUILD_TREE}
  ${SCIRUN_INCLUDE_THIRDPARTY}
)

SUBDIRS(src)

