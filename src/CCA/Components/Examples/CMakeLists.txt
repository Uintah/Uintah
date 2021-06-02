
set( SRCS
        AMRHeat.cpp
        AMRWave.cc
        Benchmark.cc
        Burger.cc
        DOSweep.cc
        ExamplesLabel.cc
        Interpolator.cc
        Heat.cpp
        ParticleTest1.cc
        Poisson1.cc
        Poisson2.cc
        Poisson3.cc
        Poisson4.cc
        RegionDB.cc
        RegridderTest.cc
        SolverTest1.cc
        Wave.cc
    )

if( HAVE_HYPRE )
    set( SRCS ${SRCS} SolverTest2.cc )
endif()

if( BUILD_MODELS_RADIATION )
    set( SRCS ${SRCS} RMCRT_Test.cc )
endif ()

if( HAVE_CUDA )
    set( SRCS ${SRCS} UnifiedSchedulerTest.cc )
    # jcs need to add the cu file...
    # jcs dlink files???
endif()

add_library( CCA_Components_Examples ${SRCS} )

target_link_libraries( CCA_Components_Examples
    PUBLIC
        CCA_Components_Models
        CCA_Components_Application
        Core_GeometryPiece
    )
