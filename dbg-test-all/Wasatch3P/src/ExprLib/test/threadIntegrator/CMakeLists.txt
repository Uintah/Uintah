nebo_cuda_prep_dir()

link_directories( ${exprlib_BINARY_DIR} )

set( nt ${NTHREADS} )
nebo_add_executable( threadIntegrate_${nt} driver.cpp )
target_link_libraries( threadIntegrate_${nt} exprlib )

# only run the test on Release builds
if( ${CMAKE_BUILD_TYPE} STREQUAL "Release" ) 
  add_test( thread_integrator_${nt} threadIntegrate_${nt} )
endif()
