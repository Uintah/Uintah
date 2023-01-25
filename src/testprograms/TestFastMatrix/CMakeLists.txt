add_executable( testfastmatrix testfastmatrix.cc )
target_link_libraries( testfastmatrix UintahCore ${LAPACK_LIBRARIES} ${BLAS_LIBRARIES} m )

add_executable( perffastmatrix perffastmatrix.cc )
target_link_libraries( perffastmatrix UintahCore ${LAPACK_LIBRARIES} ${BLAS_LIBRARIES} m )