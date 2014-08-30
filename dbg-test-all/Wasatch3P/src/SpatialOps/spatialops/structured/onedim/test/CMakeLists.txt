add_executable( lagrange testLagrange.cpp )
target_link_libraries( lagrange spatialops-onedim spatialops ${TPL_LIBRARIES} )
add_test( lagrange_coefficients lagrange )
