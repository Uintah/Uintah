IF(NOT WIN32)
  # doesn't build on windows....
  ADD_EXECUTABLE(graphview graphview.cc DaVinci.cc GV_TaskGraph.cc)
ENDIF(NOT WIN32)