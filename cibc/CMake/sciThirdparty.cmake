MACRO(SCIRUN_THIRDPARTY_OPTION version compiler_version bits)

# Search only if the location is not already known.
IF(NOT SCIRUN_THIRDPARTY_PATH)
  # Determine system
  IF(WIN32 AND NOT CYGWIN)
     SET(SCI_ARCH "Win32")
  ELSE(WIN32 AND NOT CYGWIN)
    IF(${CMAKE_SYSTEM_NAME} EQUAL "IRIX")
     SET(SCI_ARCH "IRIX64")
    ELSE(${CMAKE_SYSTEM_NAME} EQUAL "IRIX")
     SET(SCI_ARCH ${CMAKE_SYSTEM_NAME})
    ENDIF(${CMAKE_SYSTEM_NAME} EQUAL "IRIX") 
  ENDIF(WIN32 AND NOT CYGWIN)

  # First check if a thirdparty path exists on the sci system
  # based on the version, system architecture, compiler version, 
  # and number of bits
  SET (SCI_3P_DIR_ATTEMPT "/usr/sci/projects/SCIRun/Thirdparty/${version}/${SCI_ARCH}/${compiler_version}-${bits}bit")

  IF(EXISTS ${SCI_3P_DIR_ATTEMPT})
    SET(SCIRUN_THIRDPARTY_PATH ${SCI_3P_DIR_ATTEMPT} CACHE PATH "Path to SCIRun's Thirdparty")
  ELSE(EXISTS ${SCI_3P_DIR_ATTEMPT})
    SET(SCIRUN_THIRDPARTY_PATH NOTFOUND CACHE PATH "Path to SCIRun's Thirdparty")
  ENDIF(EXISTS ${SCI_3P_DIR_ATTEMPT})
ENDIF(NOT SCIRUN_THIRDPARTY_PATH)

IF(SCIRUN_THIRDPARTY_PATH)
  SET(SCIRUN_THIRDPARTY_FOUND 1)
ELSE(SCIRUN_THIRDPARTY_PATH)
  SET(SCIRUN_THIRDPARTY_FOUND 0)
  MESSAGE(FATAL_ERROR
    "Cannot build SCIRun without Thirdparty. Please set SCIRUN_THIRDPARTY_PATH.")
ENDIF(SCIRUN_THIRDPARTY_PATH)

ENDMACRO(SCIRUN_THIRDPARTY_OPTION)