#-------------------------------------------
# Set up some Wasatch configuration options
cmake_dependent_option( WASATCH_BUILD_W3P_LIBS
    "Automatically build SpatialOps, ExprLib, TabProps, RadProps and (potentially) PoKiTT - set to 'OFF' if you want to use local development versions"
    ON
    ENABLE_WASATCH
    OFF
    )

cmake_dependent_option( WASATCH_ENABLE_POKITT
    "Enable PoKiTT. Requires Cantera libraries and builds additional Wasatch functionality"
    OFF
    ENABLE_WASATCH
    OFF
    )
#-------------------------------------------

include( Wasatch3P.cmake )
message( STATUS "DONE WITH W3P Builds\n")

if( NOT HYPRE_FOUND )
    message( SEND_ERROR "HYPRE was not found.  Try setting one or more of the following:"
        "\n\tHYPRE_DIR - path to hyper installation"
        "\n\tHYPRE_INC - path to hypre header files"
        "\n\tHYPRE_LIB - path to hypre library files" )
endif()

#-------------------------------------------
# Collect all source files for the Wasatch component over this directory and subdirectories
set( WASATCH_SRC
        BCHelper.cc
        ConvectiveInterpolationMethods.cc
        CoordinateHelper.cc
        DualTimeMatrixManager.cc
        FieldAdaptor.cc
        GraphHelperTools.cc
        NestedGraphHelper.cc
        OldVariable.cc
        ParseTools.cc
        ParticlesHelper.cc
        Properties.cc
        ReductionHelper.cc
        SetProcID.cc
        TagNames.cc
        TaskInterface.cc
        TimeStepper.cc
        Wasatch.cc
        WasatchBCHelper.cc
        WasatchParticlesHelper.cc
    )

if( WASATCH_ENABLE_POKITT )
    add_subdirectory( Coal )
endif()
add_subdirectory( Expressions )
add_subdirectory( Operators )
add_subdirectory( Transport )
#-------------------------------------------

#-------------------------------------------
# Now build the Wasatch library
add_library( CCA_Components_Wasatch ${WASATCH_SRC} )

target_link_libraries( CCA_Components_Wasatch
        CCA_Components_Application
        CCA_Components_Models
        CCA_Components_Schedulers
        CCA_Ports
        Core_Exceptions
        Core_Geometry
        Core_GeometryPiece
        Core_Grid
        Core_Parallel
        Core_ProblemSpec
        Core_Util
        ${W3P_LIBS}
    )

target_include_directories( CCA_Components_Wasatch
    PUBLIC
        ${CMAKE_SOURCE_DIR}
        ${CMAKE_BINARY_DIR}/include
    )
#-------------------------------------------
