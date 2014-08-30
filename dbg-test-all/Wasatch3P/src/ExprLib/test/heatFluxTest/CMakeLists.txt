nebo_cuda_prep_dir()

link_directories( ${exprlib_BINARY_DIR} )

nebo_add_executable( heatflux driver.cpp )
target_link_libraries( heatflux exprlib )

nebo_add_executable( integrate integrate.cpp TemperatureTransEqn.cpp )
target_link_libraries( integrate exprlib )

add_test( simple_heat_flux heatflux )

if( NOT ENABLE_CUDA )
  add_test( heat_eqn_integrator integrate )
endif( NOT ENABLE_CUDA )
