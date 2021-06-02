add_subdirectory( Benchmarks )
add_subdirectory( tools )

add_executable( compare_uda compare_uda.cc )
target_link_libraries( compare_uda UintahCore )

#------------------------------ BUILD SUS -------------------------------
add_executable( sus sus.cc )

target_link_libraries( sus
        UintahCore
        CCA_Ports
        Component_Libs
    )
#------------------------------ BUILD SUS -------------------------------


# Create symlinks for input dir
message( STATUS ${CMAKE_CURRENT_BINARY_DIR}/inputs )
add_custom_target( input_dir_links ALL
    COMMAND ${CMAKE_COMMAND} -E create_symlink
            ${CMAKE_CURRENT_SOURCE_DIR}/inputs
            ${CMAKE_CURRENT_BINARY_DIR}/inputs
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    )
