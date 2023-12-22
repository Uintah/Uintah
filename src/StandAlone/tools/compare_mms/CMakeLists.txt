add_executable( compare_mms
        compare_mms.cc
        ExpMMS.cc
        LinearMMS.cc
        SineMMS.cc
    )

target_include_directories( compare_mms
    PRIVATE
        "$<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}>"
    )

target_link_libraries( compare_mms
        Core_DataArchive
        Core_Grid
        Core_Geometry

        CCA_Components_ProblemSpecification
    )

install( TARGETS compare_mms
    DESTINATION ${CMAKE_INSTALL_BINDIR}
    EXPORT uintah_exe_targets
)
