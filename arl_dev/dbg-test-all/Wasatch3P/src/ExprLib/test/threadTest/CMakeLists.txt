nebo_cuda_prep_dir()

link_directories( ${exprlib_BINARY_DIR} )

set( nt ${NTHREADS} )

nebo_add_executable( simpleThreadTest_${nt} main.cpp test.cpp )
target_link_libraries( simpleThreadTest_${nt} exprlib )

# only run these tests in release mode.
if( ${CMAKE_BUILD_TYPE} STREQUAL "Release" ) 
  add_test( thread-test-diffusion            simpleThreadTest_${nt} --nvar 16 --nstep 10 --nx 15 --ny 15 --nz 15 --diffusion                 )
  add_test( thread-test-diffusion-rxncoupled simpleThreadTest_${nt} --nvar 16 --nstep 10 --nx 15 --ny 15 --nz 15 --diffusion --coupled-source )
  add_test( thread-test-diffusion-rxn        simpleThreadTest_${nt} --nvar 16 --nstep 10 --nx 15 --ny 15 --nz 15 --diffusion --indep-source)
  add_test( thread-test-rxncoupled           simpleThreadTest_${nt} --nvar 16 --nstep 10 --nx 15 --ny 15 --nz 15 --coupled-source )
  add_test( thread-test-rxn                  simpleThreadTest_${nt} --nvar 16 --nstep 10 --nx 15 --ny 15 --nz 15 --indep-source )
endif()