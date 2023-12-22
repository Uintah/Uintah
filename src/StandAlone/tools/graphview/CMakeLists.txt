add_executable( graphview
        graphview.cc
        DaVinci.cc
        GV_TaskGraph.cc
    )

target_link_libraries( graphview
        Core_DataArchive
        Core_Parallel
    )

install( TARGETS graphview
    DESTINATION ${CMAKE_INSTALL_BINDIR}
    EXPORT uintah_exe_targets
)
