nebo_cuda_prep_dir()

link_directories( ${exprlib_BINARY_DIR} )

nebo_add_executable( integrator testIntegrator.cpp )
target_link_libraries( integrator exprlib )
add_test( expression_integrator integrator )

nebo_add_executable( timedep testTimeDepIntegrator.cpp )
target_link_libraries( timedep exprlib )
add_test( timedep_integrator timedep )