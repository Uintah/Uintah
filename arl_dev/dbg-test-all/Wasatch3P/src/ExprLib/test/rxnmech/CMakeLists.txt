nebo_cuda_prep_dir()

nebo_add_executable( rxnmech main.cpp RHS.cpp )
target_link_libraries( rxnmech exprlib )
add_test( reactions rxnmech )
