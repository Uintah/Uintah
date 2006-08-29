# Generate the SCIRUNConfig.cmake file in the build tree.  Also configure
# one for installation.  The file tells external projects how to use
# SCIRun Core.

#-----------------------------------------------------------------------------
# Settings specific to the build tree.

# The "use" file.
SET(SCIRUN_USE_FILE ${PROJECT_BINARY_DIR}/UseSCIRUN.cmake)

# The library dependencies file.
SET(SCIRUN_LIBRARY_DEPENDS_FILE ${PROJECT_BINARY_DIR}/SCIRUNLibraryDepends.cmake)

# The build settings file.
SET(SCIRUN_BUILD_SETTINGS_FILE ${PROJECT_BINARY_DIR}/SCIRUNBuildSettings.cmake)

# Library directory.
SET(SCIRUN_LIBRARY_DIRS_CONFIG ${SCIRUN_LIBRARY_PATH})

# Determine the include directories needed.
SET(SCIRUN_INCLUDE_DIRS_CONFIG
  ${SCIRUN_INCLUDE_THIRDPARTY}
  ${SCIRUN_INCLUDE_DIRS_BUILD_TREE}
  ${SCIRUN_INCLUDE_DIRS_SYSTEM}
  ${SCIRUN_INCLUDE_ADDITIONAL}
)

#-----------------------------------------------------------------------------
# Configure SCIRUNConfig.cmake for the build tree.
CONFIGURE_FILE(${PROJECT_SOURCE_DIR}/SCIRUNConfig.cmake.in
               ${PROJECT_BINARY_DIR}/SCIRUNConfig.cmake @ONLY IMMEDIATE)

#-----------------------------------------------------------------------------
# Settings specific to the install tree.

# The "use" file.
SET(SCIRUN_USE_FILE ${CMAKE_INSTALL_PREFIX}/lib/SCIRun/UseSCIRUN.cmake)

# The library dependencies file.
SET(SCIRUN_LIBRARY_DEPENDS_FILE
    ${CMAKE_INSTALL_PREFIX}/lib/SCIRun/SCIRUNLibraryDepends.cmake)

# The build settings file.
SET(SCIRUN_BUILD_SETTINGS_FILE
    ${CMAKE_INSTALL_PREFIX}/lib/SCIRun/SCIRUNBuildSettings.cmake)

# Include directories.
SET(SCIRUN_INCLUDE_DIRS_CONFIG
  ${SCIRUN_INCLUDE_THIRDPARTY}
  ${SCIRUN_INCLUDE_DIRS_INSTALL_TREE}
  ${SCIRUN_INCLUDE_DIRS_SYSTEM}
  ${SCIRUN_INCLUDE_ADDITIONAL}
)

# Link directories.
SET(SCIRUN_LIBRARY_DIRS_CONFIG ${CMAKE_INSTALL_PREFIX}/lib/SCIRun)

#-----------------------------------------------------------------------------
# Configure SCIRUNConfig.cmake for the install tree.
#CONFIGURE_FILE(${PROJECT_SOURCE_DIR}/SCIRUNConfig.cmake.in
#               ${PROJECT_BINARY_DIR}/Utilities/SCIRUNConfig.cmake @ONLY IMMEDIATE)
