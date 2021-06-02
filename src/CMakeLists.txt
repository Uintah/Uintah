cmake_minimum_required( VERSION 3.10 )
enable_language( CXX )
set( CMAKE_CXX_STANDARD 11 ) # minimum C++ compiler standard conformity

project( Uintah )

# jcs probably need to collect information to be dumped at the end of configuration
#     Maybe collect into a list called "CONFIG_INFO" or something?
message("Uintah build system on platform: ${CMAKE_SYSTEM}")

include( configure_options.cmake       )  # general, high-level options. Components should add specific options within their own scope
include( configure_introspection.cmake )  # system introspection

# probably not a great idea, but this will set the root directory on the inclusion path
include_directories(
        ${CMAKE_SOURCE_DIR}
        ${CMAKE_CURRENT_BINARY_DIR}/include
    )

#----------------------------------------------------------
# Set up core Uintah stuff
add_subdirectory( Core )
add_subdirectory( CCA )
add_subdirectory( StandAlone )  # contains sus
if( ENABLE_UNIT_TESTS )
    add_subdirectory( testprograms )
endif()
add_subdirectory( tools )
add_subdirectory( include )
#----------------------------------------------------------
