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
        ${CMAKE_SOURCE_DIR}
    PRIVATE
        ${LIBXML2_INCLUDE_DIRS}
    )