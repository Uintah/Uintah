get_filename_component( dir ${CMAKE_CURRENT_LIST_FILE} PATH )

set( rxn_SRC ${rxn_SRC} ${dir}/PhilCoalTableImporter.cpp PARENT_SCOPE )

add_executable( coal testPhilImporter.cpp PhilCoalTableImporter.cpp )
target_link_libraries( coal tabprops )

file( COPY oxyflam1.mix DESTINATION ${CMAKE_CURRENT_BINARY_DIR} )
add_test( phil_coaltable_test coal )
