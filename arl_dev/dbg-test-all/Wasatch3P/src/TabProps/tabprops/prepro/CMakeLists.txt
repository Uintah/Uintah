add_subdirectory( rxnmdl )
add_subdirectory( mixmdl )

add_library( tablehelper STATIC 
  NonAdiabaticTableHelper.cpp
  TableBuilder.cpp
  )

add_library( parser STATIC ParseGroup.cpp )

add_executable( mixrxn PreproDriver.cpp )

target_link_libraries( mixrxn
  mixmdl
  rxnmdl
  tablehelper
  tabprops
  parser
  ${TPL_LIBRARIES}
  )

install( FILES 
  CanteraTblBuilder.h
  NonAdiabaticTableHelper.h
  ParseGroup.h
  PreproDriver.h
  TableBuilder.h
  DESTINATION include/tabprops/prepro
  PERMISSIONS OWNER_READ GROUP_READ WORLD_READ
  )

install( TARGETS parser ARCHIVE DESTINATION lib/tabprops )
install( TARGETS mixrxn RUNTIME DESTINATION bin )
install( TARGETS parser EXPORT TabProps DESTINATION lib/tabprops )
