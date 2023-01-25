add_library( CCA_Components_ICE_EOS
        BirchMurnaghan.cc
        EquationOfState.cc
        EquationOfStateFactory.cc
        Gruneisen.cc
        HardSphereGas.cc
        IdealGas.cc
        JWL.cc
        JWLC.cc
        KnaussSeaWater.cc
        KumariDass.cc
        Murnaghan.cc
        Thomsen_Hartka_water.cc
        Tillotson.cc
        TST.cc
    )

target_link_libraries( CCA_Components_ICE_EOS
        Core_Exceptions
        Core_Geometry
        Core_GeometryPiece
        Core_ProblemSpec
        Core_Util
    )

target_include_directories( CCA_Components_ICE_EOS
    PUBLIC
        ${CMAKE_SOURCE_DIR}
        ${CMAKE_BINARY_DIR}/include
    )