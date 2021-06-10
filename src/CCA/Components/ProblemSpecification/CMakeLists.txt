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
        ${CMAKE_SOURCE_DIR}
    )
