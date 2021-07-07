set( MDL_SRC ModelFactory.cc )

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

add_library( CCA_Components_Models ${MDL_SRC} )

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

target_link_libraries( CCA_Components_Models ${LIB_DEPS} )
target_include_directories( CCA_Components_Models
    PUBLIC
        ${CMAKE_SOURCE_DIR}
        ${CMAKE_BINARY_DIR}/include
    )