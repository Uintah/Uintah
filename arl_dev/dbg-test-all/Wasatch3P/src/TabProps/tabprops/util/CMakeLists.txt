if( TabProps_UTILS )

  message( STATUS "Building tabprops query utilities" )

  add_executable( pointquery PointTableQuery.cpp )
  target_link_libraries( pointquery tabprops ${TPL_LIBRARIES} )

  add_executable( tablequery StateTableQuery.cpp )
  target_link_libraries( tablequery tabprops ${TPL_LIBRARIES} )
  install( TARGETS tablequery pointquery RUNTIME DESTINATION bin )

  add_executable( tablecompare TableCompare.cpp )
  target_link_libraries( tablecompare tabprops ${Boost_LIBRARIES} )
  install( TARGETS tablecompare RUNTIME DESTINATION bin )

  # FlameMaster import/export for SNL.  Because FlameMaster provides all auxiliary
  # quanitites of interest (e.g. viscosity), we don't shove it through the
  # TabProps prepro machinery.
  add_executable( fm_loader FlameMasterToTabProps.cpp FlameMaster.cpp )
  target_link_libraries( fm_loader tabprops ${Boost_LIBRARIES} )
  install( TARGETS fm_loader RUNTIME DESTINATION bin )

endif( TabProps_UTILS )