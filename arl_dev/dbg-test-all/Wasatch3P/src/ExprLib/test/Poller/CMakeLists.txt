nebo_cuda_prep_dir()

link_directories( ${exprlib_BINARY_DIR} )

nebo_add_executable( pollertest PollerTest.cpp )
target_link_libraries( pollertest exprlib )

# The poller test dies on threaded builds occassionally.
# Disable it until we can figure out why.
if( NOT ENABLE_THREADS AND NOT ENABLE_CUDA )
  add_test( poller_test pollertest )
endif( NOT ENABLE_THREADS AND NOT ENABLE_CUDA )