add_library( CCA_Components_Application ApplicationCommon.cc )

target_link_libraries( CCA_Components_Application
        CCA_Ports
        Core_DataArchive
        Core_Exceptions
        Core_Grid
    )