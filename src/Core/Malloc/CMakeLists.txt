add_library( Core_Malloc
        Allocator.cc
        AllocOS.cc
        malloc.cc
        new.cc
    )

target_include_directories( Core_Malloc
        PUBLIC
            "$<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}>"
            "$<BUILD_INTERFACE:${CMAKE_BINARY_DIR}/include>"
            "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>"
        PRIVATE
            ${CMAKE_BINARY_DIR}
    )

if( ENABLE_MEMORY_INITIALIZATION )
    set( INITIALIZE_MEMORY true )
endif()
configure_file( testmem_init.h.in ${CMAKE_CURRENT_BINARY_DIR}/mem_init.h )


install( FILES Allocator.h AllocOS.h AllocPriv.h ${CMAKE_CURRENT_BINARY_DIR}/mem_init.h
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/Core/Malloc
)

install( TARGETS Core_Malloc
    EXPORT uintah_lib_targets
    DESTINATION ${CMAKE_INSTALL_LIBDIR}
)
