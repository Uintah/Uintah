add_library( Core_Containers
        ConsecutiveRangeSet.cc
        templates.cc
    )

target_link_libraries( Core_Containers
    PUBLIC
        Core_Exceptions
        ${ZLIB_LIBRARIES}
    )

target_include_directories( Core_Containers
    PUBLIC
        ${CMAKE_SOURCE_DIR}
    )
