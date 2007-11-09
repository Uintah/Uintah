ADD_SUBDIRECTORY(Modules)
ADD_SUBDIRECTORY(Ports)


# Make Uintah's tclIndex
IF(WIN32) 
  # On Windows, we would just make a batch file to do the same thing as the createTCLIndex sh script,
  # but the << operator is not accepted.  So we must create a file per tcl index
  set (CREATE_TCL_PACKAGE ${UINTAH_SOURCE_DIR}/Dataflow/GUI)
  CONFIGURE_FILE (
    "${SCIRUN_CORE_SOURCE_DIR}/scripts/cmake_win32createTclIndex.in"
    "${CMAKE_CURRENT_BINARY_DIR}/createUintahTclIndex.tcl"
  )
  ADD_CUSTOM_TARGET(create_tclIndex_Uintah ALL
    COMMAND ${TCLSH} createUintahTclIndex.tcl
  )  
ELSE(WIN32)

  # Create the tclIndex file
  ADD_CUSTOM_TARGET(create_tclIndex_${pkg} ALL
    chmod a+x ${CREATETCLINDEX}
    COMMAND ${CREATETCLINDEX} ${UINTAH_SOURCE_DIR}/Dataflow/GUI 
  )
ENDIF(WIN32)
