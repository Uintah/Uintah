get_filename_component( dir ${CMAKE_CURRENT_LIST_FILE} PATH )

install( FILES
  LBMSFieldManager.h
  DESTINATION include/expression/lbms
  PERMISSIONS OWNER_READ GROUP_READ WORLD_READ
  )
  