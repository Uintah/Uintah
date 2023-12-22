add_subdirectory( Application )
add_subdirectory( DataArchiver )
add_subdirectory( LoadBalancers )
add_subdirectory( Parent )
add_subdirectory( PostProcessUda )
add_subdirectory( ProblemSpecification )
add_subdirectory( Regridder )
add_subdirectory( Schedulers )
add_subdirectory( SimulationController )
add_subdirectory( Solvers )
add_subdirectory( SwitchingCriteria )

set( LIB_COLLECTION
        CCA_Components_Application
        CCA_Components_DataArchiver
        CCA_Components_LoadBalancers
        CCA_Components_Parent
        CCA_Components_PostProcessUda
        CCA_Components_ProblemSpecification
        CCA_Components_Regridder
        CCA_Components_Schedulers
        CCA_Components_SimulationController
        CCA_Components_Solvers
        CCA_Components_SwitchingCriteria
        CCA_Ports
     )

# Build components as requested in configuration
set( NO_COMPONENTS True )

if( ENABLE_ARCHES )
    set( NO_COMPONENTS False )
    add_subdirectory( Arches )
    list( APPEND LIB_COLLECTION CCA_Components_Arches )
endif( ENABLE_ARCHES )

if( ENABLE_EXAMPLES )
    set( NO_COMPONENTS False )
    add_subdirectory( Examples )
    list( APPEND LIB_COLLECTION CCA_Components_Examples )
endif( ENABLE_EXAMPLES )

if( ENABLE_ICE )
    set( NO_COMPONENTS False )
    add_subdirectory( ICE )
    list( APPEND LIB_COLLECTION CCA_Components_ICE )
    if( ENABLE_MPM_ICE )
        add_subdirectory( MPMICE )
        list( APPEND LIB_COLLECTION CCA_Components_MPMICE )
    endif( ENABLE_MPM_ICE )
endif( ENABLE_ICE )

if( ENABLE_MPM )
    set( NO_COMPONENTS False )
    add_subdirectory( MPM )
    list( APPEND LIB_COLLECTION CCA_Components_MPM )
endif( ENABLE_MPM )

if( ENABLE_WASATCH )
    set( NO_COMPONENTS False )
    add_subdirectory( Wasatch )
    list( APPEND LIB_COLLECTION CCA_Components_Wasatch )
endif( ENABLE_WASATCH )

if( NOT NO_COMPONENTS )
    add_subdirectory( OnTheFlyAnalysis )
    add_subdirectory( Models )

    list( APPEND LIB_COLLECTION
            CCA_Components_Models
            CCA_Components_OnTheFlyAnalysis
        )
else()
    message( STATUS "No components were specified. Minimal Uintah build...\n")
endif( NOT NO_COMPONENTS )

# create a library that is just a collection of all of the components we need to link against
# when components are activated, they can add themselves to this interface
# Note that ordering here is important, since dependency analysis was never properly done in Uintah library development
add_library( Component_Libs INTERFACE )
target_link_libraries( Component_Libs INTERFACE
        ${LIB_COLLECTION}
    )
