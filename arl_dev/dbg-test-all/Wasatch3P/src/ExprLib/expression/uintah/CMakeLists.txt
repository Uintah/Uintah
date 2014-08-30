get_filename_component( dir ${CMAKE_CURRENT_LIST_FILE} PATH )

install( FILES
  UintahFieldManager.h
  DESTINATION include/expression/uintah
  PERMISSIONS OWNER_READ GROUP_READ WORLD_READ
  )