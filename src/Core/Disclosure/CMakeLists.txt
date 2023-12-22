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
        "$<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}>"
        "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>"
    )


install( FILES TypeDescription.h TypeUtils.h
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/Core/Disclosure
)

install( TARGETS Core_Disclosure
    EXPORT uintah_lib_targets
    DESTINATION ${CMAKE_INSTALL_LIBDIR}
)
