# each of these subdirectories should append to this list of source files:
set( ICE_SRC
        AMRICE.cc
        ICE.cc
        impAMRICE.cc
        impICE.cc
        customInitialize.cc
    )
set( ICE_INC ${CMAKE_CURRENT_SOURCE_DIR} )

add_subdirectory( Advection )
add_subdirectory( Core )
add_subdirectory( CustomBCs )
##add_subdirectory( Docs )
add_subdirectory( EOS )
add_subdirectory( Materials )
add_subdirectory( PressureSolve )
add_subdirectory( SpecificHeatModel )
add_subdirectory( TurbulenceModel )
add_subdirectory( WallShearStressModel )

message( STATUS "ICE include dirs: ${ICE_INC}" )

add_library( CCA_Components_ICE ${ICE_SRC} )
target_include_directories( CCA_Components_ICE
    PRIVATE
        ${ICE_INC}
    )