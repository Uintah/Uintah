add_library( CCA_Components_ICE_SpecificHeatModel
        Component.cc
        Debye.cc
        NASAPolynomial.cc
        Polynomial.cc
        SpecificHeat.cc
        SpecificHeatFactory.cc
    )

target_link_libraries( CCA_Components_ICE_SpecificHeatModel
        Core_Exceptions
        Core_Geometry
        Core_ProblemSpec
        Core_Util
    )

target_include_directories( CCA_Components_ICE_SpecificHeatModel
    PUBLIC
        ${CMAKE_SOURCE_DIR}
        ${CMAKE_BINARY_DIR}/include
    )