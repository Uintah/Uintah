add_library( Core_Parallel
        BufferInfo.cc
        PackBufferInfo.cc
        Parallel.cc
        ProcessorGroup.cc
        UintahParallelComponent.cc
        UintahParallelPort.cc
    )

target_link_libraries( Core_Parallel
    PUBLIC
        Core_Exceptions
        Core_Malloc
        Core_Util
        MPI::MPI_CXX
    )

target_include_directories( Core_Parallel
    PUBLIC
        ${CMAKE_SOURCE_DIR}
        ${MPI_CXX_INCLUDE_DIRS}
    )
