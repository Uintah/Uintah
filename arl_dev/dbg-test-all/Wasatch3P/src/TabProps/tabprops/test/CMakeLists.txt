if( TabProps_PREPROCESSOR )
  add_executable( rxnmdl_test RxnMdlTest.cpp )
  target_link_libraries( rxnmdl_test 
    rxnmdl
    tablehelper
    tabprops
    parser
    ${TPL_LIBRARIES}
    )

  add_executable( mixmdl_test MixMdlTest.cpp )
  target_link_libraries( mixmdl_test
    mixmdl
    tabprops
    parser
    ${TPL_LIBRARIES}
    )

  file( COPY ${PROJECT_SOURCE_DIR}/test/gri30.cti DESTINATION ${CMAKE_CURRENT_BINARY_DIR} )
  file( COPY ${PROJECT_SOURCE_DIR}/test/elements.xml DESTINATION ${CMAKE_CURRENT_BINARY_DIR} )

endif( TabProps_PREPROCESSOR )

if( TabProps_ENABLE_TESTING )

  if( TabProps_PREPROCESSOR )
    add_test( reaction_model rxnmdl_test )
    add_test( mixing_model mixmdl_test )
    add_executable( statetbl_test StateTableTest.cpp )
    target_link_libraries( statetbl_test tabprops ${TPL_LIBRARIES} )
    add_test( state_table statetbl_test )
  endif( TabProps_PREPROCESSOR )
  
  add_executable( testinterp1d testInterp1D.cpp )
  target_link_libraries( testinterp1d interp ${TPL_LIBRARIES} )
  add_test( interp_1d testinterp1d )
  
  add_executable( testinterp2d testInterp2D.cpp )
  target_link_libraries( testinterp2d interp ${TPL_LIBRARIES} )
  add_test( interp_2d testinterp2d )
  
  add_executable( testinterp3d testInterp3D.cpp )
  target_link_libraries( testinterp3d interp ${TPL_LIBRARIES} )
  add_test( interp_3d testinterp3d )
  
  add_executable( testinterp4d testInterp4D.cpp )
  target_link_libraries( testinterp4d interp ${TPL_LIBRARIES} )
  add_test( interp_4d testinterp4d )

  add_executable( testinterp5d testInterp5D.cpp )
  target_link_libraries( testinterp5d interp ${TPL_LIBRARIES} )
  add_test( interp_5d testinterp5d )

  add_executable( table_metadata testMetaData.cpp )
  target_link_libraries( table_metadata tabprops ${TPL_LIBRARIES} )
  add_test( table_metadata table_metadata )

  add_executable( table_loader StateTableLoader.cpp )
  target_link_libraries( table_loader tabprops ${TPL_LIBRARIES} )
endif()


