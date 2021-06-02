add_executable( compare_mms
        compare_mms.cc
        ExpMMS.cc
        LinearMMS.cc
        SineMMS.cc
    )

target_include_directories( compare_mms
    PUBLIC
        ${CMAKE_SOURCE_DIR}
    )

target_link_libraries( compare_mms
        Core_DataArchive
        Core_Grid
#        Core_Grid_Variables
        Core_Geometry

        CCA_Components_ProblemSpecification
    )