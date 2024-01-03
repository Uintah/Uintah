if ( HAVE_VISIT )

    add_library( VisIt_libsim
        visit_libsim.cc
        visit_libsim_callbacks.cc
        visit_libsim_customUI.cc
        visit_libsim_database.cc
        ../interfaces/warehouseInterface.cc
        ../interfaces/utils.cc
        )

    target_link_libraries( VisIt_libsim
        PUBLIC
          CCA_Ports
          Core_DataArchive
          Core_Grid
          Core_OS
          Core_Parallel
          CCA_Components_Schedulers
          Core_Util
          VISIT::VISIT
        )

    target_include_directories( VisIt_libsim
        PUBLIC
          "$<BUILD_INTERFACE:${CMAKE_BINARY_DIR}/include>"
          "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>"
          "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}../interfaces>"
          "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>"
    )

    install( FILES uintah.ui DESTINATION ${CMAKE_INSTALL_DATADIR} )

endif ( HAVE_VISIT )
