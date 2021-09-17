add_library( VisIt_interfaces
    archiveInterface.cc
    warehouseInterface.cc
    utils.cc
    )

target_link_libraries( VisIt_interfaces
    PUBLIC
      Core_DataArchive
      Core_Grid
      Core_Math
      CCA_Ports
    )

target_include_directories( VisIt_interfaces
    PUBLIC
      ${CMAKE_CURRENT_SOURCE_DIR}
    )
