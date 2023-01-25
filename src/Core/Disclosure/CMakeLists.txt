add_library( Core_Disclosure
        TypeDescription.cc
        TypeUtils.cc
    )

target_link_libraries( Core_Disclosure
        Core_Exceptions
        Core_Parallel
    )

target_include_directories( Core_Disclosure
    PUBLIC
        ${CMAKE_SOURCE_DIR}
    )
