nebo_cuda_prep_dir()

link_directories( ${exprlib_BINARY_DIR} )

nebo_add_executable( multFields driver.cpp BandC.cpp )
target_link_libraries( multFields exprlib )

add_test( multiple_fields multFields )
