#include <CCA/Components/Arches/ParticleModels/RateDeposition.h>
#include <CCA/Components/Arches/BoundaryCond_new.h>
#include <CCA/Components/Arches/ParticleModels/ParticleTools.h>
#include <Core/Exceptions/ProblemSetupException.h>
#include <CCA/Components/Arches/ParticleModels/CoalHelper.h>
#include <iostream>
#include <iomanip>

using namespace Uintah;
//using namespace std;

Uintah::RateDeposition::RateDeposition( std::string task_name, int matl_index, const int N ) :
TaskInterface( task_name, matl_index ), _Nenv(N) {
}

RateDeposition::~RateDeposition(){

}

void
RateDeposition::problemSetup( ProblemSpecP& db ){

  const ProblemSpecP db_root = db->getRootNode();
  _Tmelt = ParticleTools::getAshHemisphericalTemperature(db);
  db->getWithDefault("CaO",_CaO,26.49/100.0);
  db->getWithDefault("MgO",_MgO,4.47/100.0);
  db->getWithDefault("AlO",_AlO,14.99/100.0);
  db->getWithDefault("SiO",_SiO,38.9/100.0);

  _ParticleTemperature_base_name  = ParticleTools::parse_for_role_to_label(db,"temperature");
  _MaxParticleTemperature_base_name= ParticleTools::parse_for_role_to_label(db,"max_temperature");

  _ProbParticleX_base_name = "ProbParticleX";
  _ProbParticleY_base_name = "ProbParticleY";
  _ProbParticleZ_base_name = "ProbParticleZ";

  _ProbDepositionX_base_name = "ProbDepositionX";
  _ProbDepositionY_base_name = "ProbDepositionY";
  _ProbDepositionZ_base_name = "ProbDepositionZ";

  _RateDepositionX_base_name= "RateDepositionX";
  _RateDepositionY_base_name= "RateDepositionY";
  _RateDepositionZ_base_name= "RateDepositionZ";

  _ProbSurfaceX_name = "ProbSurfaceX";
  _ProbSurfaceY_name = "ProbSurfaceY";
  _ProbSurfaceZ_name = "ProbSurfaceZ";

  _WallTemperature_name = "Temperature";

  _xvel_base_name  = ParticleTools::parse_for_role_to_label(db,"uvel");
  _yvel_base_name  = ParticleTools::parse_for_role_to_label(db,"vvel");
  _zvel_base_name  = ParticleTools::parse_for_role_to_label(db,"wvel");

  _weight_base_name  = "w";
  _rho_base_name  = ParticleTools::parse_for_role_to_label(db,"density");
  _diameter_base_name  = ParticleTools::parse_for_role_to_label(db,"size");

  _FluxPx_base_name  = "FluxPx";
  _FluxPy_base_name  = "FluxPy";
  _FluxPz_base_name  = "FluxPz";
  _pi_div_six = acos(-1.0)/6.0;

}


void
RateDeposition::create_local_labels(){
     for (int i =0; i< _Nenv ; i++){
     const std::string ProbParticleX_name = get_env_name(i, _ProbParticleX_base_name);
     const std::string ProbParticleY_name = get_env_name(i, _ProbParticleY_base_name);
     const std::string ProbParticleZ_name = get_env_name(i, _ProbParticleZ_base_name);

     const std::string ProbDepositionX_name = get_env_name(i, _ProbDepositionX_base_name);
     const std::string ProbDepositionY_name = get_env_name(i, _ProbDepositionY_base_name);
     const std::string ProbDepositionZ_name = get_env_name(i, _ProbDepositionZ_base_name);

     const std::string FluxPx_name = get_env_name(i, _FluxPx_base_name);
     const std::string FluxPy_name = get_env_name(i, _FluxPy_base_name);
     const std::string FluxPz_name = get_env_name(i, _FluxPz_base_name);

     const std::string RateDepositionX_name = get_env_name(i, _RateDepositionX_base_name);
     const std::string RateDepositionY_name = get_env_name(i, _RateDepositionY_base_name);
     const std::string RateDepositionZ_name = get_env_name(i, _RateDepositionZ_base_name);

     register_new_variable< SFCXVariable<double> >(RateDepositionX_name );
     register_new_variable< SFCYVariable<double> >(RateDepositionY_name );
     register_new_variable< SFCZVariable<double> >(RateDepositionZ_name );

     register_new_variable< SFCXVariable<double> >(ProbParticleX_name );
     register_new_variable< SFCYVariable<double> >(ProbParticleY_name );
     register_new_variable< SFCZVariable<double> >(ProbParticleZ_name );

     register_new_variable< SFCXVariable<double> >(FluxPx_name );
     register_new_variable< SFCYVariable<double> >(FluxPy_name );
     register_new_variable< SFCZVariable<double> >(FluxPz_name );

     register_new_variable< SFCXVariable<double> >(ProbDepositionX_name );
     register_new_variable< SFCYVariable<double> >(ProbDepositionY_name );
     register_new_variable< SFCZVariable<double> >(ProbDepositionZ_name );
     }
     register_new_variable< SFCXVariable<double> >(_ProbSurfaceX_name );
     register_new_variable< SFCYVariable<double> >(_ProbSurfaceY_name );
     register_new_variable< SFCZVariable<double> >(_ProbSurfaceZ_name );

}
//
//------------------------------------------------
//-------------- INITIALIZATION ------------------
//------------------------------------------------
//
void
RateDeposition::register_initialize( std::vector<ArchesFieldContainer::VariableInformation>& variable_registry , const bool packed_tasks){

  //FUNCITON CALL     STRING NAME(VL)     TYPE       DEPENDENCY    GHOST DW     VR
  for ( int i=0; i< _Nenv;i++){
    const std::string ProbParticleX_name = get_env_name(i, _ProbParticleX_base_name);
    const std::string ProbParticleY_name = get_env_name(i, _ProbParticleY_base_name);
    const std::string ProbParticleZ_name = get_env_name(i, _ProbParticleZ_base_name);

    const std::string ProbDepositionX_name = get_env_name(i, _ProbDepositionX_base_name);
    const std::string ProbDepositionY_name = get_env_name(i, _ProbDepositionY_base_name);
    const std::string ProbDepositionZ_name = get_env_name(i, _ProbDepositionZ_base_name);

    const std::string FluxPx_name = get_env_name(i, _FluxPx_base_name);
    const std::string FluxPy_name = get_env_name(i, _FluxPy_base_name);
    const std::string FluxPz_name = get_env_name(i, _FluxPz_base_name);

    const std::string RateDepositionX_name = get_env_name(i, _RateDepositionX_base_name);
    const std::string RateDepositionY_name = get_env_name(i, _RateDepositionY_base_name);
    const std::string RateDepositionZ_name = get_env_name(i, _RateDepositionZ_base_name);

    register_variable(  RateDepositionX_name   , ArchesFieldContainer::COMPUTES , variable_registry );
    register_variable(  RateDepositionY_name   , ArchesFieldContainer::COMPUTES , variable_registry );
    register_variable(  RateDepositionZ_name   , ArchesFieldContainer::COMPUTES , variable_registry );

    register_variable(  FluxPx_name    ,  ArchesFieldContainer::COMPUTES , variable_registry );
    register_variable(  FluxPy_name    ,  ArchesFieldContainer::COMPUTES , variable_registry );
    register_variable(  FluxPz_name    ,  ArchesFieldContainer::COMPUTES , variable_registry );

    register_variable(  ProbParticleX_name    ,  ArchesFieldContainer::COMPUTES , variable_registry );
    register_variable(  ProbParticleY_name    ,  ArchesFieldContainer::COMPUTES , variable_registry );
    register_variable(  ProbParticleZ_name    ,  ArchesFieldContainer::COMPUTES , variable_registry );

    register_variable(  ProbDepositionX_name    ,  ArchesFieldContainer::COMPUTES , variable_registry );
    register_variable(  ProbDepositionY_name    ,  ArchesFieldContainer::COMPUTES , variable_registry );
    register_variable(  ProbDepositionZ_name    ,  ArchesFieldContainer::COMPUTES , variable_registry );
  }

  register_variable(  _ProbSurfaceX_name     , ArchesFieldContainer::COMPUTES ,  variable_registry );
  register_variable(  _ProbSurfaceY_name     , ArchesFieldContainer::COMPUTES ,  variable_registry );
  register_variable(  _ProbSurfaceZ_name     , ArchesFieldContainer::COMPUTES ,  variable_registry );

}

void
RateDeposition::initialize( const Patch* patch, ArchesTaskInfoManager* tsk_info ){

  for ( int e=0; e< _Nenv;e++){
    const std::string ProbParticleX_name = get_env_name(e, _ProbParticleX_base_name);
    const std::string ProbParticleY_name = get_env_name(e, _ProbParticleY_base_name);
    const std::string ProbParticleZ_name = get_env_name(e, _ProbParticleZ_base_name);

    const std::string ProbDepositionX_name = get_env_name(e, _ProbDepositionX_base_name);
    const std::string ProbDepositionY_name = get_env_name(e, _ProbDepositionY_base_name);
    const std::string ProbDepositionZ_name = get_env_name(e, _ProbDepositionZ_base_name);

    const std::string FluxPx_name = get_env_name(e, _FluxPx_base_name);
    const std::string FluxPy_name = get_env_name(e, _FluxPy_base_name);
    const std::string FluxPz_name = get_env_name(e, _FluxPz_base_name);

    const std::string RateDepositionX_name = get_env_name(e, _RateDepositionX_base_name);
    const std::string RateDepositionY_name = get_env_name(e, _RateDepositionY_base_name);
    const std::string RateDepositionZ_name = get_env_name(e, _RateDepositionZ_base_name);

    SFCXVariable<double>& FluxPx   =         *(tsk_info->get_uintah_field<SFCXVariable<double> >(FluxPx_name));
    SFCYVariable<double>& FluxPy   =         *(tsk_info->get_uintah_field<SFCYVariable<double> >(FluxPy_name));
    SFCZVariable<double>& FluxPz   =         *(tsk_info->get_uintah_field<SFCZVariable<double> >(FluxPz_name));

    SFCXVariable<double>& ProbParticleX  =   *(tsk_info->get_uintah_field<SFCXVariable<double> >(ProbParticleX_name) );
    SFCYVariable<double>& ProbParticleY  =   *(tsk_info->get_uintah_field<SFCYVariable<double> >(ProbParticleY_name) );
    SFCZVariable<double>& ProbParticleZ  =   *(tsk_info->get_uintah_field<SFCZVariable<double> >(ProbParticleZ_name) );

    SFCXVariable<double>& ProbDepositionX   =  *(tsk_info->get_uintah_field<SFCXVariable<double> >(ProbDepositionX_name) );
    SFCYVariable<double>& ProbDepositionY   =  *(tsk_info->get_uintah_field<SFCYVariable<double> >(ProbDepositionY_name) );
    SFCZVariable<double>& ProbDepositionZ   =  *(tsk_info->get_uintah_field<SFCZVariable<double> >(ProbDepositionZ_name) );

    SFCXVariable<double>& RateDepositionX   =  *(tsk_info->get_uintah_field<SFCXVariable<double> >( RateDepositionX_name) );
    SFCYVariable<double>& RateDepositionY   =  *(tsk_info->get_uintah_field<SFCYVariable<double> >( RateDepositionY_name) );
    SFCZVariable<double>& RateDepositionZ   =  *(tsk_info->get_uintah_field<SFCZVariable<double> >( RateDepositionZ_name) );

  Uintah::BlockRange range( patch->getExtraCellLowIndex(), patch->getExtraCellHighIndex() );
  Uintah::parallel_for( range, [&](int i, int j, int k){
    RateDepositionX(i,j,k) = 0.0;
    RateDepositionY(i,j,k) = 0.0;
    RateDepositionZ(i,j,k) = 0.0;

    ProbParticleX(i,j,k)   = 0.0;
    ProbParticleY(i,j,k)   = 0.0;
    ProbParticleZ(i,j,k)   = 0.0;

    ProbDepositionX(i,j,k) = 0.0;
    ProbDepositionY(i,j,k) = 0.0;
    ProbDepositionZ(i,j,k) = 0.0;

    FluxPx(i,j,k)  = 0.0;
    FluxPy(i,j,k)  = 0.0;
    FluxPz(i,j,k) = 0.0;
      });
   }
    SFCXVariable<double>& ProbSurfaceX =  *(tsk_info->get_uintah_field<SFCXVariable<double> >(_ProbSurfaceX_name) );
    SFCYVariable<double>& ProbSurfaceY =  *(tsk_info->get_uintah_field<SFCYVariable<double> >(_ProbSurfaceY_name) );
    SFCZVariable<double>& ProbSurfaceZ =  *(tsk_info->get_uintah_field<SFCZVariable<double> >(_ProbSurfaceZ_name) );
    Uintah::BlockRange range( patch->getExtraCellLowIndex(), patch->getExtraCellHighIndex() );
    Uintah::parallel_for( range, [&](int i, int j, int k){
      ProbSurfaceX(i,j,k) = 0.0;
      ProbSurfaceY(i,j,k) = 0.0;
      ProbSurfaceZ(i,j,k) = 0.0;
      });
}
//
//------------------------------------------------
//------------- TIMESTEP INIT --------------------
//------------------------------------------------
//
void
RateDeposition::register_timestep_init( std::vector<ArchesFieldContainer::VariableInformation>& variable_registry , const bool packed_tasks){
  for ( int i=0; i< _Nenv;i++){
    const std::string ProbParticleX_name = get_env_name(i, _ProbParticleX_base_name);
    const std::string ProbParticleY_name = get_env_name(i, _ProbParticleY_base_name);
    const std::string ProbParticleZ_name = get_env_name(i, _ProbParticleZ_base_name);

    const std::string ProbDepositionX_name = get_env_name(i, _ProbDepositionX_base_name);
    const std::string ProbDepositionY_name = get_env_name(i, _ProbDepositionY_base_name);
    const std::string ProbDepositionZ_name = get_env_name(i, _ProbDepositionZ_base_name);

    const std::string FluxPx_name = get_env_name(i, _FluxPx_base_name);
    const std::string FluxPy_name = get_env_name(i, _FluxPy_base_name);
    const std::string FluxPz_name = get_env_name(i, _FluxPz_base_name);

    const std::string RateDepositionX_name = get_env_name(i, _RateDepositionX_base_name);
    const std::string RateDepositionY_name = get_env_name(i, _RateDepositionY_base_name);
    const std::string RateDepositionZ_name = get_env_name(i, _RateDepositionZ_base_name);

    register_variable(  RateDepositionX_name   , ArchesFieldContainer::COMPUTES , variable_registry );
    register_variable(  RateDepositionY_name   , ArchesFieldContainer::COMPUTES , variable_registry );
    register_variable(  RateDepositionZ_name   , ArchesFieldContainer::COMPUTES , variable_registry );

    register_variable(  FluxPx_name    ,  ArchesFieldContainer::COMPUTES , variable_registry );
    register_variable(  FluxPy_name    ,  ArchesFieldContainer::COMPUTES , variable_registry );
    register_variable(  FluxPz_name    ,  ArchesFieldContainer::COMPUTES , variable_registry );

    register_variable(  ProbParticleX_name    ,  ArchesFieldContainer::COMPUTES , variable_registry );
    register_variable(  ProbParticleY_name    ,  ArchesFieldContainer::COMPUTES , variable_registry );
    register_variable(  ProbParticleZ_name    ,  ArchesFieldContainer::COMPUTES , variable_registry );

    register_variable(  ProbDepositionX_name    ,  ArchesFieldContainer::COMPUTES , variable_registry );
    register_variable(  ProbDepositionY_name    ,  ArchesFieldContainer::COMPUTES , variable_registry );
    register_variable(  ProbDepositionZ_name    ,  ArchesFieldContainer::COMPUTES , variable_registry );
  }

  register_variable(  _ProbSurfaceX_name     , ArchesFieldContainer::COMPUTES ,  variable_registry );
  register_variable(  _ProbSurfaceY_name     , ArchesFieldContainer::COMPUTES ,  variable_registry );
  register_variable(  _ProbSurfaceZ_name     , ArchesFieldContainer::COMPUTES ,  variable_registry );

}

void
RateDeposition::timestep_init( const Patch* patch, ArchesTaskInfoManager* tsk_info ){

  for ( int e=0; e< _Nenv;e++){
    const std::string ProbParticleX_name = get_env_name(e, _ProbParticleX_base_name);
    const std::string ProbParticleY_name = get_env_name(e, _ProbParticleY_base_name);
    const std::string ProbParticleZ_name = get_env_name(e, _ProbParticleZ_base_name);

    const std::string ProbDepositionX_name = get_env_name(e, _ProbDepositionX_base_name);
    const std::string ProbDepositionY_name = get_env_name(e, _ProbDepositionY_base_name);
    const std::string ProbDepositionZ_name = get_env_name(e, _ProbDepositionZ_base_name);

    const std::string FluxPx_name = get_env_name(e, _FluxPx_base_name);
    const std::string FluxPy_name = get_env_name(e, _FluxPy_base_name);
    const std::string FluxPz_name = get_env_name(e, _FluxPz_base_name);

    const std::string RateDepositionX_name = get_env_name(e, _RateDepositionX_base_name);
    const std::string RateDepositionY_name = get_env_name(e, _RateDepositionY_base_name);
    const std::string RateDepositionZ_name = get_env_name(e, _RateDepositionZ_base_name);

    SFCXVariable<double>& FluxPx   =  *(tsk_info->get_uintah_field<SFCXVariable<double> >(FluxPx_name));
    SFCYVariable<double>& FluxPy   =  *(tsk_info->get_uintah_field<SFCYVariable<double> >(FluxPy_name));
    SFCZVariable<double>& FluxPz   =  *(tsk_info->get_uintah_field<SFCZVariable<double> >(FluxPz_name));

    SFCXVariable<double>& ProbParticleX  =  *(tsk_info->get_uintah_field<SFCXVariable<double> >(ProbParticleX_name) );
    SFCYVariable<double>& ProbParticleY  =  *(tsk_info->get_uintah_field<SFCYVariable<double> >(ProbParticleY_name) );
    SFCZVariable<double>& ProbParticleZ  =  *(tsk_info->get_uintah_field<SFCZVariable<double> >(ProbParticleZ_name) );

    SFCXVariable<double>& ProbDepositionX   =  *(tsk_info->get_uintah_field<SFCXVariable<double> >(ProbDepositionX_name) );
    SFCYVariable<double>& ProbDepositionY   =  *(tsk_info->get_uintah_field<SFCYVariable<double> >(ProbDepositionY_name) );
    SFCZVariable<double>& ProbDepositionZ   =  *(tsk_info->get_uintah_field<SFCZVariable<double> >(ProbDepositionZ_name) );

    SFCXVariable<double>& RateDepositionX   =  *(tsk_info->get_uintah_field<SFCXVariable<double> >( RateDepositionX_name)  );
    SFCYVariable<double>& RateDepositionY   =  *(tsk_info->get_uintah_field<SFCYVariable<double> >( RateDepositionY_name) );
    SFCZVariable<double>& RateDepositionZ   =  *(tsk_info->get_uintah_field<SFCZVariable<double> >( RateDepositionZ_name) );

    Uintah::BlockRange range( patch->getExtraCellLowIndex(), patch->getExtraCellHighIndex() );
    Uintah::parallel_for( range, [&](int i, int j, int k){
       RateDepositionX(i,j,k) = 0.0;
       RateDepositionY(i,j,k) = 0.0;
       RateDepositionZ(i,j,k) = 0.0;

       ProbParticleX(i,j,k)   = 0.0;
       ProbParticleY(i,j,k)   = 0.0;
       ProbParticleZ(i,j,k)   = 0.0;

       ProbDepositionX(i,j,k) = 0.0;
       ProbDepositionY(i,j,k) = 0.0;
       ProbDepositionZ(i,j,k) = 0.0;

       FluxPx(i,j,k)  = 0.0;
       FluxPy(i,j,k)  = 0.0;
       FluxPz(i,j,k) = 0.0;
      });

  }

  SFCXVariable<double>& ProbSurfaceX =  *(tsk_info->get_uintah_field<SFCXVariable<double> >(_ProbSurfaceX_name) );
  SFCYVariable<double>& ProbSurfaceY =  *(tsk_info->get_uintah_field<SFCYVariable<double> >(_ProbSurfaceY_name) );
  SFCZVariable<double>& ProbSurfaceZ =  *(tsk_info->get_uintah_field<SFCZVariable<double> >(_ProbSurfaceZ_name) );

   Uintah::BlockRange range( patch->getExtraCellLowIndex(), patch->getExtraCellHighIndex() );
   Uintah::parallel_for( range, [&](int i, int j, int k){
      ProbSurfaceX(i,j,k) = 0.0;
      ProbSurfaceY(i,j,k) = 0.0;
      ProbSurfaceZ(i,j,k) = 0.0;
      });

}


//
//------------------------------------------------
//------------- TIMESTEP WORK --------------------
//------------------------------------------------
//

void
RateDeposition::register_timestep_eval( std::vector<ArchesFieldContainer::VariableInformation>& variable_registry, const int time_substep , const bool packed_tasks){

   for(int e= 0; e< _Nenv; e++){
      const std::string MaxParticleTemperature_name = ParticleTools::append_env(_MaxParticleTemperature_base_name ,e);
      const std::string ParticleTemperature_name = ParticleTools::append_env(_ParticleTemperature_base_name ,e);
      const std::string weight_name = ParticleTools::append_env(_weight_base_name ,e);
      const std::string rho_name = ParticleTools::append_env(_rho_base_name ,e);
      const std::string diameter_name = ParticleTools::append_env(_diameter_base_name ,e);

      const std::string  xvel_name = ParticleTools::append_env(_xvel_base_name ,e);
      const std::string  yvel_name = ParticleTools::append_env(_yvel_base_name ,e);
      const std::string  zvel_name = ParticleTools::append_env(_zvel_base_name ,e);

      const std::string ProbParticleX_name = get_env_name(e, _ProbParticleX_base_name);
      const std::string ProbParticleY_name = get_env_name(e, _ProbParticleY_base_name);
      const std::string ProbParticleZ_name = get_env_name(e, _ProbParticleZ_base_name);

      const std::string ProbDepositionX_name = get_env_name(e, _ProbDepositionX_base_name);
      const std::string ProbDepositionY_name = get_env_name(e, _ProbDepositionY_base_name);
      const std::string ProbDepositionZ_name = get_env_name(e, _ProbDepositionZ_base_name);

      const std::string FluxPx_name = get_env_name(e, _FluxPx_base_name);
      const std::string FluxPy_name = get_env_name(e, _FluxPy_base_name);
      const std::string FluxPz_name = get_env_name(e, _FluxPz_base_name);

      const std::string RateDepositionX_name = get_env_name(e, _RateDepositionX_base_name);
      const std::string RateDepositionY_name = get_env_name(e, _RateDepositionY_base_name);
      const std::string RateDepositionZ_name = get_env_name(e, _RateDepositionZ_base_name);

      register_variable( RateDepositionX_name      , ArchesFieldContainer::MODIFIES , variable_registry );
      register_variable( RateDepositionY_name      , ArchesFieldContainer::MODIFIES , variable_registry );
      register_variable( RateDepositionZ_name      , ArchesFieldContainer::MODIFIES , variable_registry );

      register_variable( MaxParticleTemperature_name   , ArchesFieldContainer::REQUIRES , 1 , ArchesFieldContainer::NEWDW, variable_registry );
      register_variable( ParticleTemperature_name   , ArchesFieldContainer::REQUIRES , 1 , ArchesFieldContainer::NEWDW, variable_registry );
      register_variable( weight_name   ,              ArchesFieldContainer::REQUIRES , 1 , ArchesFieldContainer::NEWDW, variable_registry );
      register_variable( rho_name   ,                 ArchesFieldContainer::REQUIRES , 1 , ArchesFieldContainer::NEWDW, variable_registry );
      register_variable( diameter_name   ,            ArchesFieldContainer::REQUIRES , 1 , ArchesFieldContainer::NEWDW, variable_registry );


      register_variable( xvel_name   , ArchesFieldContainer::REQUIRES , 1 , ArchesFieldContainer::NEWDW, variable_registry );
      register_variable( yvel_name   , ArchesFieldContainer::REQUIRES , 1 , ArchesFieldContainer::NEWDW, variable_registry );
      register_variable( zvel_name   , ArchesFieldContainer::REQUIRES , 1 , ArchesFieldContainer::NEWDW, variable_registry );

      register_variable(  ProbParticleX_name    ,  ArchesFieldContainer::MODIFIES,  variable_registry );
      register_variable(  ProbParticleY_name    ,  ArchesFieldContainer::MODIFIES,  variable_registry );
      register_variable(  ProbParticleZ_name    ,  ArchesFieldContainer::MODIFIES,  variable_registry );

      register_variable(  FluxPx_name    ,  ArchesFieldContainer::MODIFIES,  variable_registry );
      register_variable(  FluxPy_name    ,  ArchesFieldContainer::MODIFIES,  variable_registry );
      register_variable(  FluxPz_name    ,  ArchesFieldContainer::MODIFIES,  variable_registry );

      register_variable(  ProbDepositionX_name    ,  ArchesFieldContainer::MODIFIES,  variable_registry );
      register_variable(  ProbDepositionY_name    ,  ArchesFieldContainer::MODIFIES,  variable_registry );
      register_variable(  ProbDepositionZ_name    ,  ArchesFieldContainer::MODIFIES,  variable_registry );
     }
  register_variable( _ProbSurfaceX_name      , ArchesFieldContainer::MODIFIES, variable_registry );
  register_variable( _ProbSurfaceY_name      , ArchesFieldContainer::MODIFIES, variable_registry );
  register_variable( _ProbSurfaceZ_name      , ArchesFieldContainer::MODIFIES, variable_registry );

  register_variable( "surf_out_normX" , ArchesFieldContainer::REQUIRES , 1 , ArchesFieldContainer::OLDDW  , variable_registry , time_substep );
  register_variable( "surf_out_normY" , ArchesFieldContainer::REQUIRES , 1 , ArchesFieldContainer::OLDDW  , variable_registry , time_substep );
  register_variable( "surf_out_normZ" , ArchesFieldContainer::REQUIRES , 1 , ArchesFieldContainer::OLDDW  , variable_registry , time_substep );
  register_variable( "surf_in_normX" , ArchesFieldContainer::REQUIRES , 1 , ArchesFieldContainer::OLDDW  , variable_registry , time_substep );
  register_variable( "surf_in_normY" , ArchesFieldContainer::REQUIRES , 1 , ArchesFieldContainer::OLDDW  , variable_registry , time_substep );
  register_variable( "surf_in_normZ" , ArchesFieldContainer::REQUIRES , 1 , ArchesFieldContainer::OLDDW  , variable_registry , time_substep );
  register_variable( "temperature" , ArchesFieldContainer::REQUIRES , 1 , ArchesFieldContainer::LATEST  , variable_registry );

  register_variable( "areaFractionFX", ArchesFieldContainer::REQUIRES, 1, ArchesFieldContainer::OLDDW , variable_registry );
  register_variable( "areaFractionFY", ArchesFieldContainer::REQUIRES, 1, ArchesFieldContainer::OLDDW , variable_registry );
  register_variable( "areaFractionFZ", ArchesFieldContainer::REQUIRES, 1, ArchesFieldContainer::OLDDW , variable_registry );

}

void
RateDeposition::eval( const Patch* patch, ArchesTaskInfoManager* tsk_info ){

  // computed probability variables:
  SFCXVariable<double>& ProbSurfaceX =  *(tsk_info->get_uintah_field<SFCXVariable<double> >(_ProbSurfaceX_name) );
  SFCYVariable<double>& ProbSurfaceY =  *(tsk_info->get_uintah_field<SFCYVariable<double> >(_ProbSurfaceY_name) );
  SFCZVariable<double>& ProbSurfaceZ =  *(tsk_info->get_uintah_field<SFCZVariable<double> >(_ProbSurfaceZ_name) );

  // constant surface normals
  constSFCXVariable<double>&  Norm_in_X =        *(tsk_info->get_const_uintah_field<constSFCXVariable<double> >("surf_in_normX") );
  constSFCYVariable<double>&  Norm_in_Y =        *(tsk_info->get_const_uintah_field<constSFCYVariable<double> >("surf_in_normY") );
  constSFCZVariable<double>&  Norm_in_Z =        *(tsk_info->get_const_uintah_field<constSFCZVariable<double> >("surf_in_normZ") );
  constSFCXVariable<double>&  Norm_out_X =        *(tsk_info->get_const_uintah_field<constSFCXVariable<double> >("surf_out_normX") );
  constSFCYVariable<double>&  Norm_out_Y =        *(tsk_info->get_const_uintah_field<constSFCYVariable<double> >("surf_out_normY") );
  constSFCZVariable<double>&  Norm_out_Z =        *(tsk_info->get_const_uintah_field<constSFCZVariable<double> >("surf_out_normZ") );

  // constant area fractions
  constSFCXVariable<double>&  areaFractionX =   *(tsk_info->get_const_uintah_field<constSFCXVariable<double> >("areaFractionFX") );
  constSFCYVariable<double>&  areaFractionY =   *(tsk_info->get_const_uintah_field<constSFCYVariable<double> >("areaFractionFY") );
  constSFCZVariable<double>&  areaFractionZ =   *(tsk_info->get_const_uintah_field<constSFCZVariable<double> >("areaFractionFZ") );

  // constant gas temperature
  constCCVariable<double>&   WallTemperature =  *(tsk_info->get_const_uintah_field<constCCVariable<double> >("temperature"));

 //Compute the probability of sticking for each face using the wall temperature.
  Uintah::BlockRange range( patch->getCellLowIndex(), patch->getExtraCellHighIndex() );
  Uintah::parallel_for( range, [&](int i, int j, int k){

     const double areaX_temp= areaFractionX(i,j,k);
     const double areaY_temp = areaFractionY(i,j,k);
     const double areaZ_temp = areaFractionZ(i,j,k);

      double Prob_self=0.0;
      double Prob_near=0.0;

     const double T_temp   = WallTemperature(i,j,k);
     const double MaxT_temp= WallTemperature(i,j,k);

   // X direction
     {   Prob_self= compute_prob_stick( areaX_temp, T_temp,MaxT_temp);
         Prob_near= compute_prob_stick( areaX_temp, WallTemperature(i-1,j,k),WallTemperature(i-1,j,k));

        ProbSurfaceX(i,j,k) =    Norm_out_X(i,j,k)>0 ? Prob_near: Prob_self;
     }

    // Y direction
    {   Prob_self= compute_prob_stick( areaY_temp, T_temp,MaxT_temp);
        Prob_near= compute_prob_stick( areaY_temp, WallTemperature(i,j-1,k),WallTemperature(i,j-1,k));

        ProbSurfaceY(i,j,k) =    Norm_out_Y(i,j,k)>0 ? Prob_near: Prob_self;
     }
   // Z direction
     {  Prob_self= compute_prob_stick( areaZ_temp, T_temp,MaxT_temp);
        Prob_near= compute_prob_stick( areaZ_temp, WallTemperature(i,j,k-1),WallTemperature(i,j,k-1));

        ProbSurfaceZ(i,j,k) =    Norm_out_Z(i,j,k)>0 ? Prob_near: Prob_self;
     }
  });

  for(int e=0; e<_Nenv; e++){
    const std::string ParticleTemperature_name = ParticleTools::append_env(_ParticleTemperature_base_name ,e);
    const std::string MaxParticleTemperature_name = ParticleTools::append_env(_MaxParticleTemperature_base_name ,e);
    const std::string weight_name = ParticleTools::append_env(_weight_base_name ,e);
    const std::string rho_name = ParticleTools::append_env(_rho_base_name ,e);
    const std::string diameter_name = ParticleTools::append_env(_diameter_base_name ,e);

    const std::string xvel_name = ParticleTools::append_env(_xvel_base_name ,e);
    const std::string yvel_name = ParticleTools::append_env(_yvel_base_name ,e);
    const std::string zvel_name = ParticleTools::append_env(_zvel_base_name ,e);

    const std::string ProbParticleX_name = get_env_name(e, _ProbParticleX_base_name);
    const std::string ProbParticleY_name = get_env_name(e, _ProbParticleY_base_name);
    const std::string ProbParticleZ_name = get_env_name(e, _ProbParticleZ_base_name);

    const std::string ProbDepositionX_name = get_env_name(e, _ProbDepositionX_base_name);
    const std::string ProbDepositionY_name = get_env_name(e, _ProbDepositionY_base_name);
    const std::string ProbDepositionZ_name = get_env_name(e, _ProbDepositionZ_base_name);

    const std::string FluxPx_name = get_env_name(e, _FluxPx_base_name);
    const std::string FluxPy_name = get_env_name(e, _FluxPy_base_name);
    const std::string FluxPz_name = get_env_name(e, _FluxPz_base_name);

    const std::string RateDepositionX_name = get_env_name(e, _RateDepositionX_base_name);
    const std::string RateDepositionY_name = get_env_name(e, _RateDepositionY_base_name);
    const std::string RateDepositionZ_name = get_env_name(e, _RateDepositionZ_base_name);

    SFCXVariable<double>& FluxPx   =  *(tsk_info->get_uintah_field<SFCXVariable<double> >(FluxPx_name));
    SFCYVariable<double>& FluxPy   =  *(tsk_info->get_uintah_field<SFCYVariable<double> >(FluxPy_name));
    SFCZVariable<double>& FluxPz   =  *(tsk_info->get_uintah_field<SFCZVariable<double> >(FluxPz_name));

    SFCXVariable<double>& ProbParticleX  =  *(tsk_info->get_uintah_field<SFCXVariable<double> >(ProbParticleX_name) );
    SFCYVariable<double>& ProbParticleY  =  *(tsk_info->get_uintah_field<SFCYVariable<double> >(ProbParticleY_name) );
    SFCZVariable<double>& ProbParticleZ  =  *(tsk_info->get_uintah_field<SFCZVariable<double> >(ProbParticleZ_name) );

    SFCXVariable<double>& ProbDepositionX   =  *(tsk_info->get_uintah_field<SFCXVariable<double> >(ProbDepositionX_name) );
    SFCYVariable<double>& ProbDepositionY   =  *(tsk_info->get_uintah_field<SFCYVariable<double> >(ProbDepositionY_name) );
    SFCZVariable<double>& ProbDepositionZ   =  *(tsk_info->get_uintah_field<SFCZVariable<double> >(ProbDepositionZ_name) );

    SFCXVariable<double>& RateDepositionX   =  *(tsk_info->get_uintah_field<SFCXVariable<double> >( RateDepositionX_name)  );
    SFCYVariable<double>& RateDepositionY   =  *(tsk_info->get_uintah_field<SFCYVariable<double> >( RateDepositionY_name) );
    SFCZVariable<double>& RateDepositionZ   =  *(tsk_info->get_uintah_field<SFCZVariable<double> >( RateDepositionZ_name) );

    constCCVariable<double>&  MaxParticleTemperature  =  *(tsk_info->get_const_uintah_field<constCCVariable<double> >( MaxParticleTemperature_name) );
    constCCVariable<double>&  ParticleTemperature  =     *(tsk_info->get_const_uintah_field<constCCVariable<double> >( ParticleTemperature_name) );
    constCCVariable<double>&  weight  =                  *(tsk_info->get_const_uintah_field<constCCVariable<double> >( weight_name) );
    constCCVariable<double>&  rho              =         *(tsk_info->get_const_uintah_field<constCCVariable<double> >( rho_name) );
    constCCVariable<double>&  diameter         =         *(tsk_info->get_const_uintah_field<constCCVariable<double> >( diameter_name) );

    constCCVariable<double>& xvel   =  *(tsk_info->get_const_uintah_field<constCCVariable<double> >( xvel_name) );
    constCCVariable<double>& yvel   =  *(tsk_info->get_const_uintah_field<constCCVariable<double> >( yvel_name) );
    constCCVariable<double>& zvel   =  *(tsk_info->get_const_uintah_field<constCCVariable<double> >( zvel_name) );

    //Compute the probability of sticking for each particle using particle temperature.
   Uintah::BlockRange range( patch->getCellLowIndex(), patch->getExtraCellHighIndex() );
   Uintah::parallel_for( range, [&](int i, int j, int k){
     const double areaX_temp= areaFractionX(i,j,k);
     const double areaY_temp = areaFractionY(i,j,k);
     const double areaZ_temp = areaFractionZ(i,j,k);

     double Prob_self=0.0;
     double Prob_near=0.0;
     double flux_self=0.0;
     double flux_near=0.0;

     const double rho_temp=rho(i,j,k);
     const double weight_temp=weight(i,j,k);
     const double dia_temp=diameter(i,j,k);
     const double u_temp=  xvel(i,j,k);
     const double v_temp=  yvel(i,j,k);
     const double w_temp=  zvel(i,j,k);
     const double T_temp = ParticleTemperature(i,j,k);
     const double MaxT_temp= MaxParticleTemperature(i,j,k);

    //--------------------compute the particle flux --------------------------------------------------------------
   // X direction
     {   Prob_self= compute_prob_stick( areaX_temp, T_temp,MaxT_temp);
         Prob_near= compute_prob_stick( areaX_temp, ParticleTemperature(i-1,j,k), MaxParticleTemperature(i-1,j,k));
         ProbParticleX(i,j,k) =    Norm_in_X(i,j,k)>0 ? Prob_near: Prob_self;
         ProbDepositionX(i,j,k)= std::min(1.0, 0.5*(ProbParticleX(i,j,k)+(ProbParticleX(i,j,k)*ProbParticleX(i,j,k) +4*(1-ProbParticleX(i,j,k))*ProbSurfaceX(i,j,k))));

         flux_self=rho_temp*u_temp*weight_temp*_pi_div_six*pow(dia_temp,3);
         flux_near=rho(i-1,j,k)*xvel(i-1,j,k)*weight(i-1,j,k)*_pi_div_six*pow(diameter(i-1,j,k),3);
         FluxPx(i,j,k)=          Norm_in_X(i,j,k)>0? flux_near:flux_self;

         RateDepositionX(i,j,k)= FluxPx(i,j,k)*Norm_in_X(i,j,k)>0.0 ? FluxPx(i,j,k)*ProbDepositionX(i,j,k):0.0;
     }

    // Y direction
     {   Prob_self= compute_prob_stick( areaY_temp, T_temp,MaxT_temp);
         Prob_near= compute_prob_stick( areaY_temp, ParticleTemperature(i,j-1,k), MaxParticleTemperature(i,j-1,k));
         ProbParticleY(i,j,k) =    Norm_in_Y(i,j,k)>0 ? Prob_near: Prob_self;
         ProbDepositionY(i,j,k)= std::min(1.0, 0.5*(ProbParticleY(i,j,k)+(ProbParticleY(i,j,k)*ProbParticleY(i,j,k) +4*(1-ProbParticleY(i,j,k))*ProbSurfaceY(i,j,k))));

         flux_self=rho_temp*v_temp*weight_temp*_pi_div_six*pow(dia_temp,3);
         flux_near=rho(i,j-1,k)*yvel(i,j-1,k)*weight(i,j-1,k)*_pi_div_six*pow(diameter(i,j-1,k),3);
         FluxPy(i,j,k)=          Norm_in_Y(i,j,k)>0? flux_near:flux_self;

         RateDepositionY(i,j,k)= FluxPy(i,j,k)*Norm_in_Y(i,j,k)>0.0 ? FluxPy(i,j,k)*ProbDepositionY(i,j,k):0.0;
     }

    // Z direction
     {   Prob_self= compute_prob_stick( areaZ_temp, T_temp,MaxT_temp);
         Prob_near= compute_prob_stick( areaZ_temp, ParticleTemperature(i,j,k-1), MaxParticleTemperature(i,j,k-1));
         ProbParticleZ(i,j,k) =    Norm_in_Z(i,j,k)>0 ? Prob_near: Prob_self;
         ProbDepositionZ(i,j,k)= std::min(1.0, 0.5*(ProbParticleZ(i,j,k)+(ProbParticleZ(i,j,k)*ProbParticleZ(i,j,k) +4*(1-ProbParticleZ(i,j,k))*ProbSurfaceZ(i,j,k))));

         flux_self=rho_temp*w_temp*weight_temp*_pi_div_six*pow(dia_temp,3);
         flux_near=rho(i,j,k-1)*zvel(i,j,k-1)*weight(i,j,k-1)*_pi_div_six*pow(diameter(i,j,k-1),3);
         FluxPz(i,j,k)=        Norm_in_Z(i,j,k)>0? flux_near:flux_self;

         RateDepositionZ(i,j,k)= FluxPz(i,j,k)*Norm_in_Z(i,j,k)>0.0 ? FluxPz(i,j,k)*ProbDepositionZ(i,j,k):0.0;

     }


    });
  } // end for environment

}

//
//------------------------------------------------
//------------- BOUNDARY CONDITIONS --------------
//------------------------------------------------
//

void
RateDeposition::register_compute_bcs( std::vector<ArchesFieldContainer::VariableInformation>& variable_registry, const int time_substep ){}

void
RateDeposition::compute_bcs( const Patch* patch, ArchesTaskInfoManager* tsk_info ){}
