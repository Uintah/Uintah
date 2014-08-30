set( libs spatialops ${TPL_LIBRARIES} )

add_executable( particle_ops TestParticleOperators.cpp )
target_link_libraries( particle_ops ${libs} )
add_test( particle_ops particle_ops )
