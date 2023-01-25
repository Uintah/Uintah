add_executable( test_blas test_blas.c )
target_link_libraries( test_blas ${BLAS_LIBRARIES} )
target_include_directories( test_blas PRIVATE ${CMAKE_BINARY_DIR} )

add_executable( test_lapack test_lapack.c )
target_link_libraries( test_lapack ${LAPACK_LIBRARIES} )
target_include_directories( test_lapack PRIVATE ${CMAKE_BINARY_DIR} )
