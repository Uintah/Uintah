add_library( Core_DataArchive DataArchive.cc )

target_include_directories( Core_DataArchive
    PUBLIC
        ${CMAKE_SOURCE_DIR}
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
