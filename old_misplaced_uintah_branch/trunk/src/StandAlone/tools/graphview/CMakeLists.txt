IF(NOT WIN32)
  # doesn't build on windows....
  ADD_EXECUTABLE(graphview graphview.cc DaVinci.cc GV_TaskGraph.cc)

  TARGET_LINK_LIBRARIES(graphview
    Packages_Uintah_Core_DataArchive
    Packages_Uintah_Core_ProblemSpec
    Packages_Uintah_Core_Util
    Packages_Uintah_CCA_Components_ProblemSpecification
    Core_Exceptions)

ENDIF(NOT WIN32)