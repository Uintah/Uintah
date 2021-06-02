add_library( Core_Malloc
        Allocator.cc
        AllocOS.cc
        malloc.cc
        new.cc
    )

target_include_directories( Core_Malloc
        PUBLIC
            ${CMAKE_SOURCE_DIR}
            ${CMAKE_BINARY_DIR}
        PRIVATE
            ${CMAKE_CURRENT_BINARY_DIR}
    )

if( ENABLE_MEMORY_INITIALIZATION )
    set( INITIALIZE_MEMORY true )
endif()
configure_file( testmem_init.h.in ${CMAKE_CURRENT_BINARY_DIR}/mem_init.h )
