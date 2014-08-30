message( "${test_cmd}" )

# some argument checking:
# test_cmd is the command to run with all its arguments
if( NOT test_cmd )
  message( FATAL_ERROR "Variable test_cmd not defined" )
endif( NOT test_cmd )

foreach( f ${test_files} )
  set( result ${test_dir}/${f} )
  file( REMOVE ${result} )  
endforeach()

execute_process(
  COMMAND ${test_cmd} ${test_args}
  RESULT_VARIABLE test_not_successful
)
if( test_not_successful )
  message( SEND_ERROR "test failed to produce expected results" )
endif( test_not_successful )

#foreach( f ${test_files} )
#  set( gold "${blessed_dir}/${f}" )
#  set( result "${test_dir}/${f}" )
#  message( "gold  : ${gold}" )
#  message( "result: ${result}" )
#  execute_process(  
#    COMMAND ${CMAKE_COMMAND} -E compare_files ${gold} ${result}
#    RESULT_VARIABLE test_not_successful
#    )
#  if( test_not_successful )
#    message( SEND_ERROR "${gold} does not match ${result} !" )
#  endif( test_not_successful )
#endforeach()
