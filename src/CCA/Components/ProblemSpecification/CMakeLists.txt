add_library( CCA_Components_ProblemSpecification
        ProblemSpecReader.cc
    )

target_link_libraries( CCA_Components_ProblemSpecification
        CCA_Ports
        Core_Exceptions
        Core_Malloc
        Core_ProblemSpec
        Core_Util
        ${LIBXML2_LIBRARIES}
    )

target_include_directories( CCA_Components_ProblemSpecification
    PUBLIC
        ${LIBXML2_INCLUDE_DIRS}
        "$<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}>"
        "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>"
    )

install( FILES ProblemSpecReader.h DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/Components/ProblemSpecification )
install( TARGETS CCA_Components_ProblemSpecification
    EXPORT uintah_lib_targets
    DESTINATION ${CMAKE_INSTALL_LIBDIR}
)
