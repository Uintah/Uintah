cmake_minimum_required( VERSION 3.10 )
enable_language( CXX )
set( CMAKE_CXX_STANDARD 11 ) # minimum C++ compiler standard conformity

project( Uintah )

# jcs probably need to collect information to be dumped at the end of configuration
#     Maybe collect into a list called "CONFIG_INFO" or something?
message("Uintah build system on platform: ${CMAKE_SYSTEM}")

# warn about building in source
if( CMAKE_SOURCE_DIR STREQUAL CMAKE_BINARY_DIR )
    message( WARNING "\nIt appears that you are building directly in the source directory.\n"
            "This is strongly discouraged.\n"
            "Suggestion:\n\tmkdir build; cd build\n"
            "Then run cmake from the 'build' directory" )
endif( CMAKE_SOURCE_DIR STREQUAL CMAKE_BINARY_DIR )

list( APPEND CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR} )

include( configure_options.cmake       )  # general, high-level options. Components should add specific options within their own scope
include( configure_introspection.cmake )  # system introspection

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
