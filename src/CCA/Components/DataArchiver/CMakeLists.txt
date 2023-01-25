add_library( CCA_Components_DataArchiver
        DataArchiver.cc
    )

target_link_libraries( CCA_Components_DataArchiver
        CCA_Ports
        CCA_Components_ProblemSpecification
        Core_Exceptions
        Core_DataArchive
        Core_Grid
        Core_GeometryPiece
        Core_OS
        Core_Parallel
        Core_ProblemSpec
        Core_Util
    )

# jcs PIDX

target_include_directories( CCA_Components_DataArchiver
    PUBLIC
        ${CMAKE_SOURCE_DIR}
    )
