include_directories( . ${Boost_INCLUDE_DIR} ${PROJECT_SOURCE_DIR} )

add_executable( create_expr
  create_expr.cpp
  CreateExpr.cpp
  ${PROJECT_SOURCE_DIR}/expression/Tag.cpp
  ${PROJECT_SOURCE_DIR}/expression/Context.cpp
  )
target_link_libraries( create_expr ${Boost_LIBRARIES} )
install( TARGETS create_expr DESTINATION bin )

if( ENABLE_OUTPUT )
  add_executable( compare_database CompareDatabase.cpp )
  target_link_libraries( compare_database ${Boost_LIBRARIES} )
  install( TARGETS compare_database DESTINATION bin )
endif()

add_subdirectory( gui )

# ------------------------------------------------
# ------  Testing for create_expr executable -----
# ------------------------------------------------

#set( test_cmd ${CMAKE_BINARY_DIR}/util/create_expr )
#set( args "--expr-name=test1 --field-type=FieldT" )
#add_test( createExprTest_1
#  ${CMAKE_COMMAND}
#  -D test_cmd=${test_cmd}
#  -D test_args:string=${args}
#  -D output_blessed=${CMAKE_SOURCE_DIR}/util/test/test1.h
#  -D output_test=${CMAKE_BINARY_DIR}/util/test1.h
#  -P ${CMAKE_SOURCE_DIR}/util/run_test.cmake
#)
#
#set( args "--expr-name=test1 --file-name=test2 --field-type=FluxT -F --template-param=DivT" )
#add_test( createExprTest_2
#  ${CMAKE_COMMAND}
#  -Dtest_cmd=${test_cmd}
#  -D test_args:string=${args}
#  -Doutput_blessed=${CMAKE_SOURCE_DIR}/util/test/test2.h
#  -Doutput_test=${CMAKE_BINARY_DIR}/util/test2.h
#  -P ${CMAKE_SOURCE_DIR}/util/run_test.cmake
#)
#
#set( args "--expr-name=test1 --file-name=test3" )
#add_test( createExprTest_3
#  ${CMAKE_COMMAND}
#  -D test_cmd=${test_cmd}
#  -D test_args:string=${args}
#  -D output_blessed=${CMAKE_SOURCE_DIR}/util/test/test3.h
#  -D output_test=${CMAKE_BINARY_DIR}/util/test3.h
#  -P ${CMAKE_SOURCE_DIR}/util/run_test.cmake
#)
#
#set( args "--expr-name=test1 --file-name=test4 --field-type=FieldT --template-param=PatchT FieldT DivT" )
#add_test( createExprTest_4
#  ${CMAKE_COMMAND}
#  -D test_cmd=${test_cmd}
#  -D test_args:string=${args}
#  -D output_blessed=${CMAKE_SOURCE_DIR}/util/test/test4.h
#  -D output_test=${CMAKE_BINARY_DIR}/util/test4.h
#  -P ${CMAKE_SOURCE_DIR}/util/run_test.cmake
#)