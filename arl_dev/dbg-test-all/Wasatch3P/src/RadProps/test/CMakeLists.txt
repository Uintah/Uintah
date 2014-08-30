include_directories( .. )

# copy input files to the test dir
file( GLOB datafiles AbsCoeff*.txt )
file( COPY ${datafiles} DESTINATION ${CMAKE_CURRENT_BINARY_DIR} )

# build the test
add_executable( testFSK testFSK.cpp )
target_link_libraries( testFSK radprops )
add_test( testFSK testFSK )

add_executable( testGreyGas testGrey.cpp )
target_link_libraries( testGreyGas radprops )
add_test( test_grey_gas testGreyGas )

add_executable( testParticles testParticles.cpp )
target_link_libraries( testParticles radprops )
add_test( test_particles testParticles )
