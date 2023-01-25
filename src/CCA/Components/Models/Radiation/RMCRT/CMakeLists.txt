set( SRC
        ${CMAKE_CURRENT_SOURCE_DIR}/Radiometer.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/RMCRTCommon.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/RayGPU.cc
    )

# jcs sort out kokkos stuff
if( ENABLE_KOKKOS )
    list( APPEND SRC ${CMAKE_CURRENT_SOURCE_DIR}/RayKokkos.cc )
else()
    list( APPEND SRC ${CMAKE_CURRENT_SOURCE_DIR}/Ray.cc )
endif()

# jcs need to sort out cuda stuff
if( ENABLE_CUDA OR ENABLE_HIP )
    list( APPEND SRC ${CMAKE_CURRENT_SOURCE_DIR}/RayGPUKernel.cu )
elseif( ENABLE_SYCL )
    list( APPEND SRC ${CMAKE_CURRENT_SOURCE_DIR}/RayGPUKernel.cc )
endif( )

set( MDL_RADIATION_SRC ${MDL_RADIATION_SRC} ${SRC} PARENT_SCOPE )
