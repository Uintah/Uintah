add_subdirectory( ArenaSoilBanerjeeBrannon )
add_subdirectory( PlasticityModels )
add_subdirectory( PortableTongeRamesh )
add_subdirectory( ViscoElastic )
add_subdirectory( QuocAnh )
#add_subdirectory( Biswajit )

set( MPM_SRC ${MPM_SRC}
        ${CMAKE_CURRENT_SOURCE_DIR}/Arenisca.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/Arenisca3.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/Arenisca4.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/ArrudaBoyce8Chain.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/CNH_MMS.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/CompMooneyRivlin.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/ConstitutiveModel.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/ConstitutiveModelFactory.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/ElasticPlasticHP.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/HypoElastic.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/HypoElasticImplicit.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/IdealGasMP.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/ImplicitCM.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/JWLppMPM.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/MurnaghanMPM.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/MWViscoElastic.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/NonLocalDruckerPrager.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/P_Alpha.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/ProgramBurn.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/RFElasticPlastic.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/RigidMaterial.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/ShellMaterial.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/SoilFoam.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/TH_Water.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/TongeRamesh_gitInfo.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/TransIsoHyper.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/TransIsoHyperImplicit.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/TransIsoHypoFrictional.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/UCNH.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/ViscoPlastic.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/ViscoScram.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/ViscoSCRAMHotSpot.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/ViscoScramImplicit.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/TransIsoHyper.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/ViscoTransIsoHyper.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/ViscoTransIsoHyperImplicit.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/Water.cc
        PARENT_SCOPE
    )

# Fortran:
#Diamm.cc
#HypoElasticFortran.cc
#Kayenta.cc
