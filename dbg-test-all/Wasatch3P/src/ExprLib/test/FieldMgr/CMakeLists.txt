nebo_cuda_prep_dir()

nebo_add_executable( fmtest main.cpp )
link_directories( ${exprlib_BINARY_DIR} )
target_link_libraries( fmtest exprlib )
add_test( field_mgr_test fmtest )