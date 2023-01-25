add_executable( makedot makedot.cc )

target_include_directories( makedot
    PRIVATE
        ${CMAKE_SOURCE_DIR}/StandAlone/tools/graphview
    )

target_link_libraries( makedot
        Core_Parallel
    )