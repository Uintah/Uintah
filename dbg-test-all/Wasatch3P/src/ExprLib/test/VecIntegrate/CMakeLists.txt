nebo_cuda_prep_dir()

nebo_add_executable( vecintegrator main.cpp )
target_link_libraries( vecintegrator exprlib )
add_test( vecintegrator vecintegrator )
