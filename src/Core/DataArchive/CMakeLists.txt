add_library( Core_DataArchive DataArchive.cc )

target_include_directories( Core_DataArchive
    PUBLIC
        "$<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}>"
        "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>"
    PRIVATE
        ${LIBXML2_INCLUDE_DIRS}
    )

# jcs  PIDX_LIBRARY

target_link_libraries( Core_DataArchive
    PUBLIC
        CCA_Components_ProblemSpecification
        CCA_Ports
        Core_Containers
        Core_Exceptions
        Core_Grid
        Core_Math
        Core_Parallel
        Core_ProblemSpec
        Core_Util
        ${LIBXML2_LIBRARIES}
    )

install( TARGETS Core_DataArchive
        EXPORT uintah_lib_targets
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
        INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

install( FILES DataArchive.h DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/Core/DataArchive )