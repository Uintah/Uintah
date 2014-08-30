nebo_cuda_prep_dir()

link_directories( ${exprlib_BINARY_DIR} )

nebo_add_executable( srctest driver.cpp )
target_link_libraries( srctest exprlib )

add_test( source_terms srctest )