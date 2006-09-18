# NOTE: This file should be the SAME as in src/StandAlone/Apps/Painter
# until a better solution is implemented


# - Find an SCIRUN installation or build tree.

# When SCIRUN is found, the SCIRUNConfig.cmake file is sourced to setup the
# location and configuration of SCIRUN.  Please read this file, or
# SCIRUNConfig.cmake.in from the SCIRUN source tree for the full list of
# definitions.  Of particular interest is SCIRUN_USE_FILE, a CMake source file
# that can be included to set the include directories, library directories,
# and preprocessor macros.  In addition to the variables read from
# SCIRUNConfig.cmake, this find module also defines
#  SCIRUN_DIR  - The directory containing SCIRUNConfig.cmake.  
#             This is either the root of the build tree, 
#             or the lib/InsightToolkit directory.  
#             This is the only cache entry.
#   
#  SCIRUN_FOUND - Whether SCIRUN was found.  If this is true, 
#              SCIRUN_DIR is okay.
#


SET(SCIRUN_DIR_STRING "directory containing SCIRUNConfig.cmake.  This is either the root of the build tree, or PREFIX/lib/SCIRun for an installation.")

# Search only if the location is not already known.
IF(NOT SCIRUN_DIR)
  # Get the system search path as a list.
  IF(UNIX)
    STRING(REGEX MATCHALL "[^:]+" SCIRUN_DIR_SEARCH1 "$ENV{PATH}")
  ELSE(UNIX)
    STRING(REGEX REPLACE "\\\\" "/" SCIRUN_DIR_SEARCH1 "$ENV{PATH}")
  ENDIF(UNIX)
  STRING(REGEX REPLACE "/;" ";" SCIRUN_DIR_SEARCH2 ${SCIRUN_DIR_SEARCH1})

  # Construct a set of paths relative to the system search path.
  SET(SCIRUN_DIR_SEARCH "")
  FOREACH(dir ${SCIRUN_DIR_SEARCH2})
    SET(SCIRUN_DIR_SEARCH ${SCIRUN_DIR_SEARCH} "${dir}/../lib/SCIRun")
  ENDFOREACH(dir)

  #
  # Look for an installation or build tree.
  #
  FIND_PATH(SCIRUN_DIR SCIRUNConfig.cmake
    # Look for an environment variable SCIRUN_DIR.
    $ENV{SCIRUN_DIR}

    # Look in places relative to the system executable search path.
    ${SCIRUN_DIR_SEARCH}

    # Look in standard UNIX install locations.
    /usr/local/lib/SCIRun
    /usr/lib/SCIRun

    # Read from the CMakeSetup registry entries.  It is likely that
    # SCIRUN will have been recently built.
    [HKEY_CURRENT_USER\\Software\\Kitware\\CMakeSetup\\Settings\\StartPath;WhereBuild1]
    [HKEY_CURRENT_USER\\Software\\Kitware\\CMakeSetup\\Settings\\StartPath;WhereBuild2]
    [HKEY_CURRENT_USER\\Software\\Kitware\\CMakeSetup\\Settings\\StartPath;WhereBuild3]
    [HKEY_CURRENT_USER\\Software\\Kitware\\CMakeSetup\\Settings\\StartPath;WhereBuild4]
    [HKEY_CURRENT_USER\\Software\\Kitware\\CMakeSetup\\Settings\\StartPath;WhereBuild5]
    [HKEY_CURRENT_USER\\Software\\Kitware\\CMakeSetup\\Settings\\StartPath;WhereBuild6]
    [HKEY_CURRENT_USER\\Software\\Kitware\\CMakeSetup\\Settings\\StartPath;WhereBuild7]
    [HKEY_CURRENT_USER\\Software\\Kitware\\CMakeSetup\\Settings\\StartPath;WhereBuild8]
    [HKEY_CURRENT_USER\\Software\\Kitware\\CMakeSetup\\Settings\\StartPath;WhereBuild9]
    [HKEY_CURRENT_USER\\Software\\Kitware\\CMakeSetup\\Settings\\StartPath;WhereBuild10]

    # Help the user find it if we cannot.
    DOC "The ${SCIRUN_DIR_STRING}"
  )
ENDIF(NOT SCIRUN_DIR)

# If SCIRUN was found, load the configuration file to get the rest of the
# settings.
IF(SCIRUN_DIR)
  SET(SCIRUN_FOUND 1)
  INCLUDE(${SCIRUN_DIR}/SCIRUNConfig.cmake)

ELSE(SCIRUN_DIR)
  SET(SCIRUN_FOUND 0)
  IF(SCIRUN_FIND_REQUIRED)
    MESSAGE(FATAL_ERROR "Please set SCIRUN_DIR to the ${SCIRUN_DIR_STRING}")
  ENDIF(SCIRUN_FIND_REQUIRED)
ENDIF(SCIRUN_DIR)
