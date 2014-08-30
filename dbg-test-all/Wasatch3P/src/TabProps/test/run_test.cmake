set( output_test ${CMAKE_CURRENT_BINARY_DIR}/output )

file( COPY ${cantera_input} DESTINATION ${CMAKE_CURRENT_BINARY_DIR} )
file( REMOVE ${outfile} )
message( STATUS ${test_cmd} " -i ${input_file} > ${output_test}" )

execute_process(
  COMMAND ${test_cmd} -i ${input_file}
  RESULT_VARIABLE test_result
  )
if( NOT ${test_result} EQUAL "0" )
  message( SEND_ERROR "Error running ${input_file}" )
endif()

message( STATUS "${diff_cmd} ${outfile} ${outfile_blessed}" )
execute_process( COMMAND ${diff_cmd} ${outfile} ${outfile_blessed}
  RESULT_VARIABLE test_result
  )
if( NOT ${test_result} EQUAL "0" )
  message( SEND_ERROR "${outfile} and ${outfile_blessed} are different!" )
endif()
  