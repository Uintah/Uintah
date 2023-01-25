add_executable( async_mpi_test  async_mpi_test.cc )
add_executable( mpi_hang        mpi_hang.cc )
add_executable( mpi_test        mpi_test.cc )

target_link_libraries( async_mpi_test Core_Parallel )
target_link_libraries( mpi_hang       Core_Parallel )
target_link_libraries( mpi_test       Core_Parallel )