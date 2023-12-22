add_library( Core_ProblemSpec
        ProblemSpec.cc
    )

target_link_libraries( Core_ProblemSpec
        Core_Exceptions
        Core_Geometry
        Core_Malloc
        Core_Util
        ${LIBXML2_LIBRARIES}
    )

target_include_directories( Core_ProblemSpec
    PUBLIC
        "$<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}>"
        "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>"
    PRIVATE
        ${LIBXML2_INCLUDE_DIRS}
    )


install( FILES ProblemSpec.h ProblemSpecP.h
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/Core/ProblemSpec
)

install( TARGETS Core_ProblemSpec
    EXPORT uintah_lib_targets
    DESTINATION ${CMAKE_INSTALL_LIBDIR}
)
