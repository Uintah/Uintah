nebo_cuda_prep_dir()

link_directories( ${exprlib_BINARY_DIR} )

nebo_add_executable( cleave cleave.cpp )
target_link_libraries( cleave exprlib )

set( test_files orig.dot tree_0.dot tree_1.dot tree_2.dot )
set( test_cmd ${CMAKE_BINARY_DIR}/test/treeCleave/cleave )
add_test( tree_cleaving
  ${CMAKE_COMMAND}
  -D test_files=${test_files}
  -D test_cmd=${test_cmd}
  -D blessed_dir=${CMAKE_CURRENT_SOURCE_DIR}
  -D test_dir=${CMAKE_CURRENT_BINARY_DIR}
  -P ${CMAKE_CURRENT_SOURCE_DIR}/run_test.cmake
)


nebo_add_executable( cleave2 cleave2.cpp )
target_link_libraries( cleave2 exprlib )
set( test_files orig2.dot tree2_0.dot tree2_1.dot tree2_2.dot )
set( test_cmd ${CMAKE_BINARY_DIR}/test/treeCleave/cleave2 )
add_test( tree_cleaving2
  ${CMAKE_COMMAND}
  -D test_files=${test_files}
  -D test_cmd=${test_cmd}
  -D blessed_dir=${CMAKE_CURRENT_SOURCE_DIR}
  -D test_dir=${CMAKE_CURRENT_BINARY_DIR}
  -P ${CMAKE_CURRENT_SOURCE_DIR}/run_test.cmake
)


nebo_add_executable( cleave3 cleave3.cpp )
target_link_libraries( cleave3 exprlib )
set( test_files orig3.dot tree3_0.dot tree3_1.dot )
set( test_cmd ${CMAKE_BINARY_DIR}/test/treeCleave/cleave3 )
add_test( tree_cleaving3
  ${CMAKE_COMMAND}
  -D test_files=${test_files}
  -D test_cmd=${test_cmd}
  -D blessed_dir=${CMAKE_CURRENT_SOURCE_DIR}
  -D test_dir=${CMAKE_CURRENT_BINARY_DIR}
  -P ${CMAKE_CURRENT_SOURCE_DIR}/run_test.cmake
)

nebo_add_executable( cleave4 cleave4.cpp )
target_link_libraries( cleave4 exprlib )
add_test( tree_cleaving4 cleave4 )

nebo_add_executable( cleave5 cleave5.cpp )
target_link_libraries( cleave5 exprlib )
add_test( tree_cleaving5 cleave5 )