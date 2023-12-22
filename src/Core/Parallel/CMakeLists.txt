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
        "$<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}>"
        "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>"
        ${MPI_CXX_INCLUDE_DIRS}
    )

install( FILES
    BufferInfo.h
    CommunicationList.hpp
    CrowdMonitor.hpp
    MasterLock.h
    PackBufferInfo.h
    Parallel.h
    ParallelContext.h
    ProcessorGroup.h
    UintahMPI.h
    UintahParallelComponent.h
    UintahParallelPort.h
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/Core/Parallel
)

install( TARGETS Core_Parallel
    EXPORT uintah_lib_targets
    DESTINATION ${CMAKE_INSTALL_LIBDIR}
)
