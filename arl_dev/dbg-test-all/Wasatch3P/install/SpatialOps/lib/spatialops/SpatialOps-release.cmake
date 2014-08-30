#----------------------------------------------------------------
# Generated CMake target import file for configuration "Release".
#----------------------------------------------------------------

# Commands may need to know the format version.
SET(CMAKE_IMPORT_FILE_VERSION 1)

# Compute the installation prefix relative to this file.
GET_FILENAME_COMPONENT(_IMPORT_PREFIX "${CMAKE_CURRENT_LIST_FILE}" PATH)
GET_FILENAME_COMPONENT(_IMPORT_PREFIX "${_IMPORT_PREFIX}" PATH)
GET_FILENAME_COMPONENT(_IMPORT_PREFIX "${_IMPORT_PREFIX}" PATH)

# Import target "spatialops-onedim" for configuration "Release"
SET_PROPERTY(TARGET spatialops-onedim APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
SET_TARGET_PROPERTIES(spatialops-onedim PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_RELEASE "CXX"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/spatialops/libspatialops-onedim.a"
  )

LIST(APPEND _IMPORT_CHECK_TARGETS spatialops-onedim )
LIST(APPEND _IMPORT_CHECK_FILES_FOR_spatialops-onedim "${_IMPORT_PREFIX}/lib/spatialops/libspatialops-onedim.a" )

# Import target "spatialops-stencil" for configuration "Release"
SET_PROPERTY(TARGET spatialops-stencil APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
SET_TARGET_PROPERTIES(spatialops-stencil PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_RELEASE "CXX"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/spatialops/libspatialops-stencil.a"
  )

LIST(APPEND _IMPORT_CHECK_TARGETS spatialops-stencil )
LIST(APPEND _IMPORT_CHECK_FILES_FOR_spatialops-stencil "${_IMPORT_PREFIX}/lib/spatialops/libspatialops-stencil.a" )

# Import target "spatialops-structured" for configuration "Release"
SET_PROPERTY(TARGET spatialops-structured APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
SET_TARGET_PROPERTIES(spatialops-structured PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_RELEASE "CXX"
  IMPORTED_LINK_INTERFACE_LIBRARIES_RELEASE "spatialops-onedim;spatialops-stencil"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/spatialops/libspatialops-structured.a"
  )

LIST(APPEND _IMPORT_CHECK_TARGETS spatialops-structured )
LIST(APPEND _IMPORT_CHECK_FILES_FOR_spatialops-structured "${_IMPORT_PREFIX}/lib/spatialops/libspatialops-structured.a" )

# Import target "spatialops" for configuration "Release"
SET_PROPERTY(TARGET spatialops APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
SET_TARGET_PROPERTIES(spatialops PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_RELEASE "CXX"
  IMPORTED_LINK_INTERFACE_LIBRARIES_RELEASE "/usr/local/boost/lib/libboost_program_options.so;spatialops-structured"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/spatialops/libspatialops.a"
  )

LIST(APPEND _IMPORT_CHECK_TARGETS spatialops )
LIST(APPEND _IMPORT_CHECK_FILES_FOR_spatialops "${_IMPORT_PREFIX}/lib/spatialops/libspatialops.a" )

# Loop over all imported files and verify that they actually exist
FOREACH(target ${_IMPORT_CHECK_TARGETS} )
  FOREACH(file ${_IMPORT_CHECK_FILES_FOR_${target}} )
    IF(NOT EXISTS "${file}" )
      MESSAGE(FATAL_ERROR "The imported target \"${target}\" references the file
   \"${file}\"
but this file does not exist.  Possible reasons include:
* The file was deleted, renamed, or moved to another location.
* An install or uninstall procedure did not complete successfully.
* The installation package was faulty and contained
   \"${CMAKE_CURRENT_LIST_FILE}\"
but not all the files it references.
")
    ENDIF()
  ENDFOREACH()
  UNSET(_IMPORT_CHECK_FILES_FOR_${target})
ENDFOREACH()
UNSET(_IMPORT_CHECK_TARGETS)

# Cleanup temporary variables.
SET(_IMPORT_PREFIX)

# Commands beyond this point should not need to know the version.
SET(CMAKE_IMPORT_FILE_VERSION)
