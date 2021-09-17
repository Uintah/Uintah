add_library( VisIt_libsim
    visit_libsim.cc
    visit_libsim_callbacks.cc
    visit_libsim_customUI.cc
    visit_libsim_database.cc
    )

target_link_libraries( VisIt_libsim
    PUBLIC
      CCA_Ports
      Core_DataArchive
      CCA_Components_OnTheFlyAnalysis
      Core_Grid
      Core_Parallel
      Core_Util
      VISIT::VISIT
    )

target_include_directories( Core_Containers
    PUBLIC
      ${CMAKE_BINARY_DIR}/include
      ${CMAKE_CURRENT_SOURCE_DIR}
    )
