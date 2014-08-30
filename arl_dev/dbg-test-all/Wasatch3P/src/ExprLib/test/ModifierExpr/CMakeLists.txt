nebo_cuda_prep_dir()

link_directories( ${exprlib_BINARY_DIR} )

nebo_add_executable( modifier main.cpp )
target_link_libraries( modifier exprlib )
add_test( modifier modifier )

nebo_add_executable( multiple_modifier multipleFieldModifier.cpp )
target_link_libraries( multiple_modifier exprlib )
add_test( multiple_modifier multiple_modifier )