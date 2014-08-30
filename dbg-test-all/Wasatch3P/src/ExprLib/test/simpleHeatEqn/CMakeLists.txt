nebo_cuda_prep_dir()

link_directories( ${exprlib_BINARY_DIR} )

nebo_add_executable( heateqn main.cpp MonolithicRHS.cpp )
target_link_libraries( heateqn exprlib )

add_test( heateqn heateqn )