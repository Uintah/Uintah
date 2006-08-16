# This is the root ITK CMakeLists file.
CMAKE_MINIMUM_REQUIRED(VERSION 2.0)

PROJECT(CIBC)

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
# Test for some required system information.
INCLUDE (CMakeBackwardCompatibilityC)

#-----------------------------------------------------------------------------
# Configure testing support.  This should be done before any
# MESSAGE(FATAL_ERROR ...) commands are invoked.
INCLUDE( CTest )
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
  SET(SCIRUN_INCLUDE_THIRDPARTY 
    ${SCIRUN_THIRDPARTY_PATH}/include
  )	
ELSE(SCIRUN_THIRDPARTY_FOUND)
  MESSAGE(FATAL_ERROR
    "Cannot build SCIRun without Thirdparty. Please set SCIRUN_THIRDPARTY_PATH.")
ENDIF(SCIRUN_THIRDPARTY_FOUND)


#-----------------------------------------------------------------------------
# Thread library
FIND_PACKAGE( Threads )
SET(THREAD_LIBRARY ${CMAKE_THREAD_LIBS} CACHE INTERNAL "Thread library.")
# DV Define DEF_THREAD?

IF(CMAKE_USE_PTHREADS)
  SET(DEF_THREAD "#define SCI_PTHREAD 1")
  # Check for sempahore libs w/in pthread library
  # DV can I scale down the lib like configure.ac for loop
  TRY_COMPILE(HAS_SEMAPHORE_LIBS
    ${PROJECT_BINARY_DIR}/CMakeTmp
    ${PROJECT_SOURCE_DIR}/CMake/testSemaphore.cc
      COMPILE_DEFINITIONS
      -l${THREAD_LIBRARY}
      -I/usr/local/include -I/usr/include -I/include
    )
    IF(HAS_SEMAPHORE_LIBS)
      SET(SEMAPHORE_LIBRARY ${THREAD_LIBRARY} CACHE INTERNAL "Semaphore library in pthreads.")
    ENDIF(HAS_SEMAPHORE_LIBS)
  ELSE(CMAKE_USE_PTHREADS)
    SET(DEF_THREAD "") 
ENDIF(CMAKE_USE_PTHREADS)

#-----------------------------------------------------------------------------
# Socket library
# DV Not sure if socket library needs to be defined for cygwin
IF(WIN32 AND NOT CYGWIN)
  SET(SOCKET_LIBRARY ws2_32 CACHE INTERNAL "Socket library ws2_32")
ENDIF(WIN32 AND NOT CYGWIN)

#-----------------------------------------------------------------------------
# dlopen library
IF(CMAKE_SYSTEM_NAME MATCHES Darwin)
  FIND_LIBRARY(DL_LIBRARY dlopen
             ${SCIRUN_THIRDPARTY_PATH}/lib
             /usr/local/lib
             /usr/lib
             /lib)
ENDIF(CMAKE_SYSTEM_NAME MATCHES Darwin)

#-----------------------------------------------------------------------------
# math library
IF(UNIX)
  FIND_LIBRARY(M_LIBRARY m
             /usr/local/lib
             /usr/lib 
             /lib)
ENDIF(UNIX)

#-----------------------------------------------------------------------------
# lapack library
OPTION(WITH_LAPACK "Build with lapack library" OFF)
MARK_AS_ADVANCED(WITH_LAPACK)		   
IF(WITH_LAPACK)		   
  FIND_LIBRARY(LAPACK_LIBRARY lapack clapack
    /usr/local/lib
    /usr/lib
    /lib)

   IF(LAPACK_LIBRARY)
     SET(HAVE_LAPACK "YES")
   ELSE(LAPACK_LIBRARY)
     SET(HAVE_LAPACK "NO")
   ENDIF(LAPACK_LIBRARY)
ELSE(WITH_LAPACK)
  SET(HAVE_LAPACK "NO")
ENDIF(WITH_LAPACK)

#-----------------------------------------------------------------------------
# blas library
# DV Fix finding blas/atlas stuff
SET(DEF_BLAS "#define HAVE_BLAS 0")
SET(DEF_CBLAS "#define HAVE_CLBLAS 0")

# DV Fix these libraries!
# blas, f, 

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
        SET(HAVE_TRACEBACK 1)
        SET(DEF_TRACEBACK "#define HAVE_TRACEBACK 1")
      ELSE(HAVE_IMAGEHELP_H)
        MESSAGE(SEND_ERROR "Trace back library requires imagehlp.h for Windows.")
      ENDIF(HAVE_IMAGEHLP_H)
  ELSE(HAVE_WINDOWS_H)
     MESSAGE(SEND_ERROR "Trace back library requires windows.h for Windows.")
  ENDIF(HAVE_WINDOWS_H)
ELSE(WIN32)
  CHECK_INCLUDE_FILE("libexc.h"       HAVE_TRACEBACK_H)
  IF(HAVE_TRACEBACK_H)
    SET(TRACEBACK_LIB exc CACHE INTERNAL "Trace back library.")
    SET(HAVE_TRACEBACK 1)
    SET(DEF_TRACEBACK "#define HAVE_TRACEBACK 1")
  ENDIF(HAVE_TRACEBACK_H)
ENDIF(WIN32)


#-----------------------------------------------------------------------------
# Z lib
IF(WIN32 AND NOT CYGWIN)
  SET(ZLIB_LIBRARY ${SCI_THIRDPARTY_DIR}/lib/z)
  SET(ZLIB_INCLUDE ${SCI_THIRDPARTY_DIR}/include)
ELSE(WIN32 AND NOT CYGWIN)
  FIND_PACKAGE( ZLIB )
ENDIF(WIN32 AND NOT CYGWIN)
SET(SCIRUN_INCLUDE_ADDITIONAL ${SCIRUN_INCLUDE_ADDITIONAL} ${ZLIB_INCLUDE_DIR})

#-----------------------------------------------------------------------------
# png lib
OPTION(WITH_PNG "Build SCIRun with PNG subbport." OFF)
IF(WITH_PNG)
  FIND_PACKAGE( PNG )
  IF(PNG_LIBRARY)
    SET(HAVE_PNG "YES")
    SET(DEF_PNG "#define HAVE_PNG 1")
    SET(SCIRUN_INCLUDE_ADDITIONAL ${SCIRUN_INCLUDE_ADDITIONAL} ${PNG_INCLUDE_DIR})
  ENDIF(PNG_LIBRARY)
ENDIF(WITH_PNG)

#-----------------------------------------------------------------------------
# DV XML2 FROM 3P FOR NOW
FIND_LIBRARY(XML2_LIBRARY xml2
            ${SCI_THIRDPARTY_DIR}/lib/
            /usr/local/lib
            /usr/lib
            /lib)
FIND_PATH(XML2_INCLUDE libxml2/libxml/
           ${SCIRUN_THIRDPARTY_PATH}/include
           /usr/local/include
           /usr/include
           /include)
SET(SCIRUN_INCLUDE_ADDITIONAL ${SCIRUN_INCLUDE_ADDITIONAL} ${XML2_INCLUDE})


#-----------------------------------------------------------------------------
# Tcl/Tk
SET(TK_VERSION 8.3.4)

#-----------------------------------------------------------------------------
# ITK library
FIND_LIBRARY(ITK_LIBRARY itk
             ${SCI_THIRDPARTY_DIR}/lib/
             /usr/local/lib
             /usr/lib
             /lib)
FIND_PATH(ITK_INCLUDE itk.h
           ${SCIRUN_THIRDPARTY_PATH}/include
           /usr/local/include
           /usr/include
           /include)
SET(SCIRUN_INCLUDE_ADDITIONAL ${SCIRUN_INCLUDE_ADDITIONAL} ${ITK_INCLUDE})
#-----------------------------------------------------------------------------
# ITCL library
FIND_LIBRARY(ITCL_LIBRARY itcl
             ${SCI_THIRDPARTY_DIR}/lib/
             /usr/local/lib
             /usr/lib
             /lib)
FIND_PATH(ITCL_INCLUDE itcl.h
           ${SCIRUN_THIRDPARTY_PATH}/include
           /usr/local/include
           /usr/include
           /include)
SET(SCIRUN_INCLUDE_ADDITIONAL ${SCIRUN_INCLUDE_ADDITIONAL} ${ITCL_INCLUDE})

#-----------------------------------------------------------------------------
# TK library
FIND_LIBRARY(TK_LIBRARY tk
             ${SCI_THIRDPARTY_DIR}/lib/
             /usr/local/lib
             /usr/lib
             /lib)
SET(TK_SRC_PATH ${SCIRUN_THIRDPARTY_PATH}/src/tk CACHE PATH "Path to tk source")

SET(TK_INCLUDE1 ${TK_SRC_PATH}/generic)
SET(TK_INCLUDE2 ${TK_SRC_PATH}/unix)
#FIND_PATH(TK_INCLUDE tk.h
#           ${SCIRUN_THIRDPARTY_PATH}/include
#           /usr/local/include
#           /usr/include
#           /include)
SET(SCIRUN_INCLUDE_ADDITIONAL ${SCIRUN_INCLUDE_ADDITIONAL} ${TK_INCLUDE1} ${TK_INCLUDE2})
#-----------------------------------------------------------------------------
# TCL library
FIND_LIBRARY(TCL_LIBRARY tcl
             ${SCI_THIRDPARTY_DIR}/lib/
             /usr/local/lib
             /usr/lib
             /lib)
FIND_PATH(TCL_INCLUDE1 tcl.h
           ${SCIRUN_THIRDPARTY_PATH}/include
           /usr/local/include
           /usr/include
           /include)

FIND_PATH(TCL_INCLUDE2 tclMath.h
           ${SCIRUN_THIRDPARTY_PATH}/src/tk8.3.4/generic
           /usr/local/include
           /usr/include
           /include)

SET(SCIRUN_INCLUDE_ADDITIONAL ${SCIRUN_INCLUDE_ADDITIONAL} ${TCL_INCLUDE1} ${TCL_INCLUDE2})

#-----------------------------------------------------------------------------
# blt lib
FIND_LIBRARY(BLT_LIBRARY blt
             ${SCI_THIRDPARTY_DIR}/lib/
             /usr/local/lib
             /usr/lib
             /lib)
FIND_PATH(BLT_INCLUDE blt.h
           ${SCIRUN_THIRDPARTY_PATH}/include
           /usr/local/include
           /usr/include
           /include)
SET(SCIRUN_INCLUDE_ADDITIONAL ${SCIRUN_INCLUDE_ADDITIONAL} ${BLT_INCLUDE})

IF(WIN32)
  SET(BLT_LIBRARY ${BLT_LIBRARY} gdi32 winspool comdlg32 user32 advapi32)
ENDIF(WIN32)

#-----------------------------------------------------------------------------
# OpenGL library
FIND_PACKAGE( OpenGL )
IF(OPENGL_LIBRARIES AND OPENGL_INCLUDE_DIR)
  SET(GL_LIBRARY ${OPENGL_gl_LIBRARY})
  SET(GL_INCLUDE ${OPENGL_INCLUDE_DIR})
  SET(SCIRUN_INCLUDE_ADDITIONAL ${SCIRUN_INCLUDE_ADDITIONAL} ${GL_INCLUDE})
ENDIF(OPENGL_LIBRARIES AND OPENGL_INCLUDE_DIR)

#-----------------------------------------------------------------------------
# Freetype
FIND_LIBRARY(FREETYPE_LIBRARY freetype
             ${SCI_THIRDPARTY_DIR}/lib/
             /usr/local/lib
             /usr/lib
             /lib)
FIND_PATH(FREETYPE_INCLUDE freetype.h
           ${SCIRUN_THIRDPARTY_PATH}/include/freetype2
           /usr/local/include
           /usr/include
           /include)
SET(SCIRUN_INCLUDE_ADDITIONAL ${SCIRUN_INCLUDE_ADDITIONAL} ${FREETYPE_INCLUDE})


#----------------------------------------------------------------------------
# X11
FIND_PACKAGE( X11 )
IF(X11_FOUND)
  SET(X_LIBRARY ${X11_LIBRARIES})
ENDIF(X11_FOUND)

#-----------------------------------------------------------------------------
# Glew
FIND_LIBRARY(GLEW_LIBRARY GLEW
             ${SCI_THIRDPARTY_DIR}/lib/
             /usr/local/lib
             /usr/lib
             /lib)
FIND_PATH(GLEW_INCLUDE glew.h
           ${SCIRUN_THIRDPARTY_PATH}/include/GL
           /usr/local/include
           /usr/include
           /include)
SET(SCIRUN_INCLUDE_ADDITIONAL ${SCIRUN_INCLUDE_ADDITIONAL} ${GLEW_INCLUDE})

SET(DEF_GLEW "#define HAVE_GLEW 1")

#-----------------------------------------------------------------------------
# DV MPEG FROM 3P FOR NOW
FIND_LIBRARY(MPEG_LIBRARY mpeg
            ${SCI_THIRDPARTY_DIR}/lib/
            /usr/local/lib
            /usr/lib
            /lib)
FIND_PATH(MPEG_INCLUDE mpeg.h
           ${SCIRUN_THIRDPARTY_PATH}/include
           /usr/local/include
           /usr/include
           /include)
SET(DEF_MPEG "#define HAVE_MPEG 1")

SET(SCIRUN_INCLUDE_ADDITIONAL ${SCIRUN_INCLUDE_ADDITIONAL} ${MPEG_INCLUDE})

#-----------------------------------------------------------------------------
# Teem
SET (CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/CMake)
FIND_PACKAGE( Teem )
IF (TEEM_LIBRARY)
  IF (TEEM_INCLUDE_DIR)
    SET(SCIRUN_INCLUDE_ADDITIONAL ${SCIRUN_INCLUDE_ADDITIONAL} ${TEEM_INCLUDE_DIR})
  ENDIF(TEEM_INCLUDE_DIR)
ENDIF(TEEM_LIBRARY)

# DV Fix hard coding of HAVE_GDCM
SET(DEF_GDCM "#define HAVE_GDCM 0")


#-----------------------------------------------------------------------------
# SCIRUN build configuration options.
# OPTION(ENABLE_SCI_MALLOC "Use SCI memory allocation/deallocation checks." OFF)
# SET(SCIRUN_ENABLE_SCI_MALLOC ${ENABLE_SCI_MALLOC})
# DV Fix checking bit of thirdparty with size of pointer check
# sci_defs/bits_defs.h
# DV clean up (if user tries to configure 64 bit machine to 32, flag error?)
# DV also maybe need to add binary flags here
# Set NBITS (64 or 32) based on the size of a void pointer.
# Handle if machine defaults to one bit size but user is trying to do another
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
IF(HAVE_LAPACK)
  SET(DEF_LAPACK "#define HAVE_LAPACK 1" CACHE INTERNAL "Have Lapack")
ELSE(HAVE_LAPACK)
  SET(DEF_LAPACK "#define HAVE_LAPACK 0" CACHE INTERNAL "Have Lapack")
ENDIF(HAVE_LAPACK)

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

# DV Hack
SET(DEF_HASHMAP "#define HAVE_GNU_HASHMAP 1")

# DV Figure out how to make these not even configurable
# sci_defs/template_defs.h
SET(TEMPLATE_TAG " " CACHE STRING "NOT SURE")
# DV This should be an empty string on some platforms
SET(TEMPLATE_BOX "<>" CACHE STRING "NOT SURE")

MARK_AS_ADVANCED(THROW_LEVEL ASSERTION_LEVEL DEF_HASH TEMPLATE_TAG TEMPLATE_BOX)
# DV Add with_exc stuff from configure.ac

INCLUDE (${CMAKE_ROOT}/Modules/CheckIncludeFile.cmake)
CHECK_INCLUDE_FILE(limits.h HAVE_LIMITS)
IF(HAVE_LIMITS)
  SET(HAVE_LIMITS_H 1)
ELSE(HAVE_LIMITS)
  SET(HAVE_LIMITS_H 0)
ENDIF(HAVE_LIMITS)

CHECK_INCLUDE_FILE(sys/select.h HAVE_SYS_SELECT)
IF(HAVE_SYS_SELECT)
  SET(HAVE_SYS_SELECT_H 1)
ELSE(HAVE_SYS_SELECT)
  SET(HAVE_SYS_SELECT_H 0)
ENDIF(HAVE_SYS_SELECT)

CHECK_INCLUDE_FILE(uinstd.h HAVE_UNISTD)
IF(HAVE_UNISTD)
  SET(HAVE_UNISTD_H 1)
ELSE(HAVE_UNISTD)
  SET(HAVE_UNISTD_H 0)
ENDIF(HAVE_UNISTD)

CHECK_INCLUDE_FILE(sys/time.h HAVE_SYS_TIME)
IF(HAVE_SYS_TIME)
  SET(HAVE_SYS_TIME_H 1)
ELSE(HAVE_SYS_TIME)
  SET(HAVE_SYS_TIME_H 0)
ENDIF(HAVE_SYS_TIME)

# DV Fix this one (ext/algorithm)
CHECK_INCLUDE_FILE(ext/algorithm  HAVE_EXT_ALGORITHM)
IF(HAVE_EXT_ALGORITHM)
  SET(HAVE_EXT_ALGORITHM_H 1)
ELSE(HAVE_EXT_ALGORITHM)
  SET(HAVE_EXT_ALGORITHM_H 0)
ENDIF(HAVE_EXT_ALGORITHM)

CHECK_INCLUDE_FILE(stdint.h HAVE_STDINT)
IF(HAVE_STDINT)
  SET(HAVE_STDINT_H 1)
ELSE(HAVE_STDINT)
  SET(HAVE_STDINT_H 0)
ENDIF(HAVE_STDINT)

CHECK_INCLUDE_FILE(inttypes.h HAVE_INTTYPES)
IF(HAVE_INTTYPES)
  SET(HAVE_INTTYPES_H 1)
ELSE(HAVE_INTTYPES)
  SET(HAVE_INTTYPES_H 0)
ENDIF(HAVE_INTTYPES)

SET(HAVE_SYS_UUID_H 0)
SET(HAVE_UUID_UUID_H 0)

# sci_defs/oogl_defs.h
#INCLUDE (${CMAKE_ROOT}/Modules/CheckIncludeFile.cmake)
#INCLUDE (${CMAKE_ROOT}/Modules/CheckLibraryExists.cmake)
#CHECK_INCLUDE_FILE(oogl/basicTexture.h HAVE_BASIC_TEXTURE)
#CHECK_INCLUDE_FILE(oogl/shadedPrim.h HAVE_SHADED_PRIM)
#CHECK_LIBRARY_EXISTS(oogl )
SET(DEF_OOGL "")

#SET(DEF_FTGL)

# Freetype in 3p
SET(DEF_FREETYPE "#define HAVE_FREETYPE 1")

#SET(DEF_OGL_EXT_HDRS)

#SET(OGL_ERR_MACRO)

OPTION(ENABLE_SCI_MALLOC "Enable sci malloc system" OFF)
IF(ENABLE_SCI_MALLOC)
  SET(DEF_MALLOC "")
ELSE(ENABLE_SCI_MALLOC)
  SET(DEF_MALLOC "#define DISABLE_SCI_MALLOC 1")
ENDIF(ENABLE_SCI_MALLOC)
SET(DEF_MALLOC_LINENUMBERS "")

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
  ${SCIRUN_INCLUDE_THIRDPARTY}
  ${SCIRUN_INCLUDE_DIRS_BUILD_TREE}
  ${SCIRUN_INCLUDE_DIRS_SYSTEM}
  ${SCIRUN_INCLUDE_ADDITIONAL}
)

SUBDIRS(src Testing)


