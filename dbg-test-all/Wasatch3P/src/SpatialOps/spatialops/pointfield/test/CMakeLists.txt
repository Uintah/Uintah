set( libs spatialops ${TPL_LIBRARIES} )

add_executable( point_field testPointField.cpp )
target_link_libraries( point_field ${libs} )
add_test( point_field point_field )