set( LIB_DEPS
        CCA_Ports
        Core_Disclosure
        Core_Exceptions
        Core_Geometry
        Core_GeometryPiece
        Core_Grid
        Core_IO
        Core_Math
        Core_Parallel
        Core_ProblemSpec
        Core_Util
    )

add_subdirectory( FluidsBased )
add_subdirectory( HEChem )
add_subdirectory( MultiMatlExchange )
add_subdirectory( Radiation )
add_subdirectory( SolidReactionModel )

# abb: for now only Radiation/RMCRT is GPU enabled
if( ENABLE_CUDA )
    set_source_files_properties(${MDL_RADIATION_SRC} PROPERTIES LANGUAGE CUDA)
endif( ENABLE_CUDA )

if( ENABLE_HIP )
    find_package(hiprand REQUIRED)

    set_source_files_properties(${MDL_RADIATION_SRC} PROPERTIES LANGUAGE CXX)
    set_source_files_properties(${MDL_RADIATION_SRC} PROPERTIES COMPILE_FLAGS "-D__HIP_ROCclr__ -D__HIP_ARCH_GFX90A__=1 --rocm-path=$ENV{ROCM_PATH} --offload-arch=gfx90a -x hip")
endif( ENABLE_HIP )

add_library( CCA_Components_Models
               ModelFactory.cc
               ${MDL_FLUIDSBASED_SRC}
               ${MDL_HECHEM_SRC}
               ${MDL_MMEXCH_SRC}
               ${MDL_RADIATION_SRC}
               ${MKL_SRM_SRC} )

if( ENABLE_ICE )
    list( APPEND LIB_DEPS
            CCA_Components_ICE_Core
            CCA_Components_ICE_CustomBCs
            CCA_Components_ICE_Materials
        )
endif()
if( ENABLE_MPM_ICE )
    list( APPEND LIB_DEPS
            CCA_Components_MPMICE_Core
            CCA_Components_SwitchingCriteria
        )
endif()

target_link_libraries( CCA_Components_Models PRIVATE ${LIB_DEPS} )
if( ENABLE_CUDA )
    target_link_libraries( CCA_Components_Models PRIVATE CUDA::cudart CUDA::curand )
endif( ENABLE_CUDA )
if( ENABLE_HIP )
    target_compile_options(CCA_Components_Models PUBLIC
          "$<$<COMPILE_LANGUAGE:CXX>:SHELL:-mllvm;-amdgpu-early-inline-all=true;-mllvm;-amdgpu-function-calls=false>"
       )
    target_link_libraries( CCA_Components_Models PUBLIC hip::device hip::hiprand )
endif( ENABLE_HIP )

target_include_directories( CCA_Components_Models
    PUBLIC
        ${CMAKE_SOURCE_DIR}
        ${CMAKE_BINARY_DIR}/include
    )
