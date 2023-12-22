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
        "$<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}>"
        "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>"
    )

install( FILES
    Array1.h
    Array2.h
    ConsecutiveRangeSet.h
    FastHashTable.h
    LockingHandle.h
    OffsetArray1.h
    RangeTree.h
    SuperBox.h
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/Core/Containers
)

install( TARGETS Core_Containers
    EXPORT uintah_lib_targets
    DESTINATION ${CMAKE_INSTALL_LIBDIR} )
