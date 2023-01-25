add_library( Core_IO
        UintahIFStreamUtil.cc
        UintahZlibUtil.cc
        )

target_link_libraries( Core_IO
        Core_Exceptions
        Core_Malloc
        ZLIB::ZLIB
    )

target_include_directories( Core_IO PUBLIC ${CMAKE_SOURCE_DIR} )
