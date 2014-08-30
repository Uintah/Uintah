nebo_cuda_prep_dir()

nebo_add_executable( field_creation field_creation.cpp )
target_link_libraries( field_creation spatialops )
add_test( field_creation field_creation )
install( TARGETS field_creation RUNTIME DESTINATION bin/spatialops )

nebo_add_executable( field_operations field_operations.cpp )
target_link_libraries( field_operations spatialops )
add_test( field_operations field_operations )
install( TARGETS field_operations RUNTIME DESTINATION bin/spatialops )

nebo_add_executable( field_reductions field_reductions.cpp )
target_link_libraries( field_reductions spatialops )
add_test( field_reductions field_reductions )
install( TARGETS field_reductions RUNTIME DESTINATION bin/spatialops )

nebo_add_executable( field_types field_types.cpp )
target_link_libraries( field_types spatialops )
add_test( field_types field_types )
install( TARGETS field_types RUNTIME DESTINATION bin/spatialops )

nebo_add_executable( field_type_inference field_type_inference.cpp )
target_link_libraries( field_type_inference spatialops )
add_test( field_type_inference field_type_inference )
install( TARGETS field_type_inference RUNTIME DESTINATION bin/spatialops )

nebo_add_executable( stencil_types stencil_type_inference.cpp )
target_link_libraries( stencil_types spatialops )
add_test( stencil_types stencil_types )
install( TARGETS stencil_types RUNTIME DESTINATION bin/spatialops )

nebo_add_executable( stencils stencils.cpp )
target_link_libraries( stencils spatialops )
add_test( stencils stencils )
install( TARGETS stencils RUNTIME DESTINATION bin/spatialops )

nebo_add_executable( masks masks.cpp )
target_link_libraries( masks spatialops )
add_test( masks masks )
install( TARGETS masks RUNTIME DESTINATION bin/spatialops )

nebo_add_executable( heat_equation heat_equation.cpp )
target_link_libraries( heat_equation spatialops )
add_test( heat_equation heat_equation )
install( TARGETS heat_equation RUNTIME DESTINATION bin/spatialops )
