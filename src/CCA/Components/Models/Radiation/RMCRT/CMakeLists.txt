set( SRC
        ${CMAKE_CURRENT_SOURCE_DIR}/Radiometer.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/RayGPU.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/RMCRTCommon.cc
    )

# jcs sort out kokkos stuff
if( ENABLE_KOKKOS )
    list( APPEND SRC ${CMAKE_CURRENT_SOURCE_DIR}/RayKokkos.cc )
else()
    list( APPEND SRC ${CMAKE_CURRENT_SOURCE_DIR}/Ray.cc )
endif()

# jcs need to sort out cuda stuff
if( ENABLE_CUDA )
    list( APPEND SRC ${CMAKE_CURRENT_SOURCE_DIR}/RayGPUKernel.cu )
endif()

set( MDL_SRC ${MDL_SRC} ${SRC} PARENT_SCOPE )