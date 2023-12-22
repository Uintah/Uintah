cmake_minimum_required( VERSION 3.10 )
enable_language( CXX )
set( CMAKE_CXX_STANDARD 11 ) # minimum C++ compiler standard conformity

project( Uintah )

# jcs probably need to collect information to be dumped at the end of configuration
#     Maybe collect into a list called "CONFIG_INFO" or something?
message("Uintah build system on platform: ${CMAKE_SYSTEM}")

# default to release builds
if( NOT CMAKE_BUILD_TYPE )
    set( CMAKE_BUILD_TYPE "Release" )
endif( NOT CMAKE_BUILD_TYPE )

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
include( GNUInstallDirs                )  # sets cache variables for install destinations

message( STATUS "Installation path: ${CMAKE_INSTALL_PREFIX}\n")

#----------------------------------------------------------
#               Set up core Uintah stuff
#----------------------------------------------------------
add_subdirectory( Core )
add_subdirectory( CCA )
add_subdirectory( StandAlone )  # contains sus
if( ENABLE_UNIT_TESTS )
    add_subdirectory( testprograms )
endif()
add_subdirectory( tools )
add_subdirectory( VisIt )

# auto-generate header files for compilation.
# Should be done after all system introspection and configuration is complete.
add_subdirectory( include )
#----------------------------------------------------------


#----------------------------------------------------------
#                  Regression Testing
#----------------------------------------------------------
set( GS_PYTHON_FILE ${CMAKE_SOURCE_DIR}/R_Tester/toplevel/generateGoldStandards.py )

if( CMAKE_BUILD_TYPE STREQUAL "Debug" )
    set( IS_DEBUG "yes" )
else()
    set( IS_DEBUG "no" )
endif()

if( ENABLE_SCI_MALLOC )
    set( SCI_MALLOC_ON "yes" )
else()
    set( SCI_MALLOC_ON "no" )
endif()

set( TEST_DATA_DIR ${CMAKE_BINARY_DIR}/TestData )

file( MAKE_DIRECTORY ${TEST_DATA_DIR} )

add_custom_target( gold_standards
    DEPENDS sus
    COMMAND ${CMAKE_COMMAND} -E
        env PYTHONPATH="${CMAKE_SOURCE_DIR}/R_Tester/toplevel:${CMAKE_SOURCE_DIR}/R_Tester"
        ${Python3_EXECUTABLE}
        ${GS_PYTHON_FILE} -d ${IS_DEBUG} -b ${CMAKE_BINARY_DIR} -s ${CMAKE_SOURCE_DIR}
        `${CMAKE_SOURCE_DIR}/R_Tester/helpers/selectComponents.sh ${CMAKE_SOURCE_DIR}/R_Tester`
        -m ${SCI_MALLOC_ON} -j ${NUMBER_OF_PROCESSORS} -v
    WORKING_DIRECTORY ${TEST_DATA_DIR}
    COMMENT "Generating Gold Standards"
    USES_TERMINAL
    )

add_custom_target( runLocalRT
    DEPENDS sus compare_uda
    COMMAND
        ${CMAKE_SOURCE_DIR}/R_Tester/toplevel/startLocalTest
        ${CMAKE_BINARY_DIR} ${CMAKE_SOURCE_DIR} ${IS_DEBUG} ${SCI_MALLOC_ON}
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    COMMENT "Running Regression Tests"
    USES_TERMINAL
    )
#----------------------------------------------------------


#----------------------------------------------------------
#                       INSTALLATION
#----------------------------------------------------------
install( EXPORT uintah_exe_targets DESTINATION ${CMAKE_INSTALL_BINDIR}/cmake )
install( EXPORT uintah_lib_targets FILE uintah_lib_targets.cmake DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake )
#----------------------------------------------------------
