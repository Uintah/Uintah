nebo_cuda_prep_dir()

link_directories( ${exprlib_BINARY_DIR} )

nebo_add_executable( test_bcs testBC.cpp )
target_link_libraries( test_bcs exprlib )

add_test( test_bcs test_bcs )
