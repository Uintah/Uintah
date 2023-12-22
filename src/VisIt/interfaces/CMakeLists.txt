#----------------------------------------------------------
# Build the library that VisIt needs for the Uintah plugin
add_library( Uintah_VisIt_Interface
    archiveInterface.cc
    utils.cc
    )

target_link_libraries( Uintah_VisIt_Interface
    PUBLIC
      Core_DataArchive
      Core_Grid
      Core_Math
      CCA_Ports
    )

target_include_directories( Uintah_VisIt_Interface
    PUBLIC
        "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>"
        "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>"
    )

install( FILES datatypes.h warehouseInterface.h DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/VisIt )

install( TARGETS Uintah_VisIt_Interface
         EXPORT Uintah_VisIt_Interface_targets
         LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
         RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR} )
install( EXPORT Uintah_VisIt_Interface_targets
    FILE Uintah_VisIt_Interface_targets.cmake
    NAMESPACE Uintah::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake
)
#----------------------------------------------------------

install( FILES utils.h DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/VisIt )
