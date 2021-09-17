add_subdirectory( Benchmarks )
add_subdirectory( tools )

add_executable( compare_uda compare_uda.cc )
target_link_libraries( compare_uda UintahCore )

#------------------------------ BUILD SUS -------------------------------
add_executable( sus sus.cc )

if( HAVE_VISIT )
  set( VISIT_INTERFACE VisIt_libsim )
endif()

target_link_libraries( sus
        UintahCore
        CCA_Ports
        Component_Libs
        ${VISIT_INTERFACE}
    )

# ensure symlinks are generated when sus is built
add_dependencies( sus
    input_dir_links
    extract_link
    rtest_link
    scripts_link
    puda_link
    )
#------------------------------ BUILD SUS -------------------------------


# Create symlinks for input dir
add_custom_target( input_dir_links ALL
    COMMAND ${CMAKE_COMMAND} -E create_symlink
            ${CMAKE_CURRENT_SOURCE_DIR}/inputs
            ${CMAKE_CURRENT_BINARY_DIR}/inputs
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    )

# create symlinks for various tools and directories
add_custom_target( rtest_link ALL
    COMMAND ${CMAKE_COMMAND} -E create_symlink ${CMAKE_SOURCE_DIR}/R_Tester/toplevel/localRT ${CMAKE_CURRENT_BINARY_DIR}/localRT
    )
add_custom_target( scripts_link ALL
    COMMAND ${CMAKE_COMMAND} -E create_symlink ${CMAKE_SOURCE_DIR}/scripts ${CMAKE_CURRENT_BINARY_DIR}/scripts
    )
add_custom_target( extract_link ALL
    COMMAND ${CMAKE_COMMAND} -E create_symlink ${CMAKE_BINARY_DIR}/StandAlone/tools/extractors/lineextract lineextract
    COMMAND ${CMAKE_COMMAND} -E create_symlink ${CMAKE_BINARY_DIR}/StandAlone/tools/extractors/timeextract timeextract
    COMMAND ${CMAKE_COMMAND} -E create_symlink ${CMAKE_BINARY_DIR}/StandAlone/tools/extractors/partextract partextract
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    DEPENDS lineextract timeextract partextract
    )
add_custom_target( puda_link ALL
    COMMAND ${CMAKE_COMMAND} -E create_symlink ${CMAKE_CURRENT_BINARY_DIR}/tools/puda/puda puda
    DEPENDS puda
    )
