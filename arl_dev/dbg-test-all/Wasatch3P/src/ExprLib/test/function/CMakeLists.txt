nebo_cuda_prep_dir()

link_directories( ${exprlib_BINARY_DIR} )

nebo_add_executable( exprfunction function_test.cpp )
target_link_libraries( exprfunction exprlib )

add_test( function_test exprfunction )
