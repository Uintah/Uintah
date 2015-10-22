#include <CCA/Components/Arches/ParticleModels/RateDeposition.h>
#include <CCA/Components/Arches/Operators/Operators.h>
#include <CCA/Components/Arches/BoundaryCond_new.h>
#include <spatialops/structured/FVStaggered.h>
#include <CCA/Components/Arches/ParticleModels/ParticleTools.h>
#include <Core/Exceptions/ProblemSetupException.h>
#include <CCA/Components/Arches/ParticleModels/CoalHelper.h>
#include <iostream>
#include <spatialops/structured/FVStaggered.h>
#include <iomanip>

using namespace Uintah;

using namespace SpatialOps;
using SpatialOps::operator *;
typedef SSurfXField SurfX;
typedef SSurfYField SurfY;
typedef SSurfZField SurfZ;
typedef SVolField  SVolF;
typedef XVolField  XVolF;
typedef YVolField  YVolF;
typedef ZVolField  ZVolF;
typedef SpatialOps::SpatFldPtr<SpatialOps::SVolField> SVolFP;
typedef SpatialOps::SpatFldPtr<SpatialOps::XVolField> XVolFP;
typedef SpatialOps::SpatFldPtr<SpatialOps::YVolField> YVolFP;
typedef SpatialOps::SpatFldPtr<SpatialOps::ZVolField> ZVolFP;
 

Uintah::RateDeposition::RateDeposition( std::string task_name, int matl_index, const int N ) :
TaskInterface( task_name, matl_index ), _Nenv(N) {
}

RateDeposition::~RateDeposition(){

}

void
RateDeposition::problemSetup( ProblemSpecP& db ){

     const ProblemSpecP db_root = db->getRootNode();
     db->require("Melting_Temperature",_Tmelt);
  
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
RateDeposition::register_initialize( std::vector<ArchesFieldContainer::VariableInformation>& variable_registry ){

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
RateDeposition::initialize( const Patch* patch, ArchesTaskInfoManager* tsk_info,
                       SpatialOps::OperatorDatabase& opr ){

  using namespace SpatialOps;
  using SpatialOps::operator *;

  
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

  XVolFP FluxPx   =  tsk_info->get_so_field<XVolF>(FluxPx_name);
  YVolFP FluxPy   =  tsk_info->get_so_field<YVolF>(FluxPy_name);
  ZVolFP FluxPz   =  tsk_info->get_so_field<ZVolF>(FluxPz_name);
 
  XVolFP ProbParticleX   =  tsk_info->get_so_field<XVolF>(ProbParticleX_name);
  YVolFP ProbParticleY   =  tsk_info->get_so_field<YVolF>(ProbParticleY_name);
  ZVolFP ProbParticleZ   =  tsk_info->get_so_field<ZVolF>(ProbParticleZ_name);

  XVolFP ProbDepositionX   =  tsk_info->get_so_field<XVolF>(ProbDepositionX_name);
  YVolFP ProbDepositionY   =  tsk_info->get_so_field<YVolF>(ProbDepositionY_name);
  ZVolFP ProbDepositionZ   =  tsk_info->get_so_field<ZVolF>(ProbDepositionZ_name);

  XVolFP RateDepositionX = tsk_info->get_so_field<XVolF>(RateDepositionX_name);
  YVolFP RateDepositionY = tsk_info->get_so_field<YVolF>(RateDepositionY_name);
  ZVolFP RateDepositionZ = tsk_info->get_so_field<ZVolF>(RateDepositionZ_name);
  *RateDepositionX <<= 0.0;
  *RateDepositionY <<= 0.0;
  *RateDepositionZ <<= 0.0;

  *ProbParticleX <<= 0.0;
  *ProbParticleY <<= 0.0;
  *ProbParticleZ <<= 0.0;
 
  *ProbDepositionX <<= 0.0;
  *ProbDepositionY <<= 0.0;
  *ProbDepositionZ <<= 0.0;
  
  *FluxPx <<= 0.0;
  *FluxPy <<= 0.0;
  *FluxPz <<= 0.0;
  }

  XVolFP ProbSurfaceX = tsk_info->get_so_field<XVolF>(_ProbSurfaceX_name);
  YVolFP ProbSurfaceY = tsk_info->get_so_field<YVolF>(_ProbSurfaceY_name);
  ZVolFP ProbSurfaceZ = tsk_info->get_so_field<ZVolF>(_ProbSurfaceZ_name);
   *ProbSurfaceX <<= 0.0;
   *ProbSurfaceY <<= 0.0;
   *ProbSurfaceZ <<= 0.0;
}
//
//------------------------------------------------
//------------- TIMESTEP INIT --------------------
//------------------------------------------------
//
void
RateDeposition::register_timestep_init( std::vector<ArchesFieldContainer::VariableInformation>& variable_registry ){
}

void
RateDeposition::timestep_init( const Patch* patch, ArchesTaskInfoManager* tsk_info,
                          SpatialOps::OperatorDatabase& opr ){

  using namespace SpatialOps;
  using SpatialOps::operator *;

}


//
//------------------------------------------------
//------------- TIMESTEP WORK --------------------
//------------------------------------------------
//

void
RateDeposition::register_timestep_eval( std::vector<ArchesFieldContainer::VariableInformation>& variable_registry, const int time_substep ){

   for(int i= 0; i< _Nenv; i++){
      const std::string MaxParticleTemperature_name = ParticleTools::append_env(_MaxParticleTemperature_base_name ,i);
      const std::string ParticleTemperature_name = ParticleTools::append_env(_ParticleTemperature_base_name ,i);
      const std::string weight_name = ParticleTools::append_env(_weight_base_name ,i);
      const std::string rho_name = ParticleTools::append_env(_rho_base_name ,i);
      const std::string diameter_name = ParticleTools::append_env(_diameter_base_name ,i);
    
      const std::string  xvel_name = ParticleTools::append_env(_xvel_base_name ,i);
      const std::string  yvel_name = ParticleTools::append_env(_yvel_base_name ,i);
      const std::string  zvel_name = ParticleTools::append_env(_zvel_base_name ,i);
     
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

      register_variable( RateDepositionX_name      , ArchesFieldContainer::COMPUTES , variable_registry );
      register_variable( RateDepositionY_name      , ArchesFieldContainer::COMPUTES , variable_registry );
      register_variable( RateDepositionZ_name      , ArchesFieldContainer::COMPUTES , variable_registry );
 
      register_variable( MaxParticleTemperature_name   , ArchesFieldContainer::REQUIRES , 0 , ArchesFieldContainer::LATEST , variable_registry );
      register_variable( ParticleTemperature_name   , ArchesFieldContainer::REQUIRES , 0 , ArchesFieldContainer::LATEST , variable_registry );
      register_variable( weight_name   ,              ArchesFieldContainer::REQUIRES , 0 , ArchesFieldContainer::LATEST , variable_registry );
      register_variable( rho_name   ,                 ArchesFieldContainer::REQUIRES , 0 , ArchesFieldContainer::LATEST , variable_registry );
      register_variable( diameter_name   ,            ArchesFieldContainer::REQUIRES , 0 , ArchesFieldContainer::LATEST , variable_registry );


      register_variable( xvel_name   , ArchesFieldContainer::REQUIRES , 0 , ArchesFieldContainer::LATEST , variable_registry );
      register_variable( yvel_name   , ArchesFieldContainer::REQUIRES , 0 , ArchesFieldContainer::LATEST , variable_registry );
      register_variable( zvel_name   , ArchesFieldContainer::REQUIRES , 0 , ArchesFieldContainer::LATEST , variable_registry );
      
      register_variable(  ProbParticleX_name    ,  ArchesFieldContainer::COMPUTES,  variable_registry );
      register_variable(  ProbParticleY_name    ,  ArchesFieldContainer::COMPUTES,  variable_registry );
      register_variable(  ProbParticleZ_name    ,  ArchesFieldContainer::COMPUTES,  variable_registry );
     
      register_variable(  FluxPx_name    ,  ArchesFieldContainer::COMPUTES,  variable_registry );
      register_variable(  FluxPy_name    ,  ArchesFieldContainer::COMPUTES,  variable_registry );
      register_variable(  FluxPz_name    ,  ArchesFieldContainer::COMPUTES,  variable_registry );

      register_variable(  ProbDepositionX_name    ,  ArchesFieldContainer::COMPUTES,  variable_registry );
      register_variable(  ProbDepositionY_name    ,  ArchesFieldContainer::COMPUTES,  variable_registry );
      register_variable(  ProbDepositionZ_name    ,  ArchesFieldContainer::COMPUTES,  variable_registry );
     }
  register_variable( _ProbSurfaceX_name      , ArchesFieldContainer::COMPUTES , 0 , ArchesFieldContainer::NEWDW  , variable_registry , time_substep );
  register_variable( _ProbSurfaceY_name      , ArchesFieldContainer::COMPUTES , 0 , ArchesFieldContainer::NEWDW  , variable_registry , time_substep );
  register_variable( _ProbSurfaceZ_name      , ArchesFieldContainer::COMPUTES , 0 , ArchesFieldContainer::NEWDW  , variable_registry , time_substep );

  register_variable( "surf_out_normX" , ArchesFieldContainer::REQUIRES , 1 , ArchesFieldContainer::OLDDW  , variable_registry , time_substep );
  register_variable( "surf_out_normY" , ArchesFieldContainer::REQUIRES , 1 , ArchesFieldContainer::OLDDW  , variable_registry , time_substep );
  register_variable( "surf_out_normZ" , ArchesFieldContainer::REQUIRES , 1 , ArchesFieldContainer::OLDDW  , variable_registry , time_substep );
  register_variable( "surf_in_normX" , ArchesFieldContainer::REQUIRES , 1 , ArchesFieldContainer::OLDDW  , variable_registry , time_substep );
  register_variable( "surf_in_normY" , ArchesFieldContainer::REQUIRES , 1 , ArchesFieldContainer::OLDDW  , variable_registry , time_substep );
  register_variable( "surf_in_normZ" , ArchesFieldContainer::REQUIRES , 1 , ArchesFieldContainer::OLDDW  , variable_registry , time_substep );
  register_variable( "temperature" , ArchesFieldContainer::REQUIRES , 0 , ArchesFieldContainer::LATEST  , variable_registry );

  register_variable( "areaFractionFX", ArchesFieldContainer::REQUIRES, 1, ArchesFieldContainer::OLDDW , variable_registry ); 
  register_variable( "areaFractionFY", ArchesFieldContainer::REQUIRES, 1, ArchesFieldContainer::OLDDW , variable_registry ); 
  register_variable( "areaFractionFZ", ArchesFieldContainer::REQUIRES, 1, ArchesFieldContainer::OLDDW , variable_registry ); 

}

void
RateDeposition::eval( const Patch* patch, ArchesTaskInfoManager* tsk_info,
                 SpatialOps::OperatorDatabase& opr ){
  using namespace SpatialOps;
  using SpatialOps::operator *;

  // computed probability variables:
  XVolFP ProbSurfaceX = tsk_info->get_so_field<XVolF>( _ProbSurfaceX_name );
  YVolFP ProbSurfaceY = tsk_info->get_so_field<YVolF>( _ProbSurfaceY_name );
  ZVolFP ProbSurfaceZ = tsk_info->get_so_field<ZVolF>( _ProbSurfaceZ_name );

  // constant surface normals 
  XVolFP const Norm_in_X = tsk_info->get_const_so_field<XVolF>( "surf_in_normX" );
  YVolFP const Norm_in_Y = tsk_info->get_const_so_field<YVolF>( "surf_in_normY" );
  ZVolFP const Norm_in_Z = tsk_info->get_const_so_field<ZVolF>( "surf_in_normZ" );
  XVolFP const Norm_out_X = tsk_info->get_const_so_field<XVolF>( "surf_out_normX" );
  YVolFP const Norm_out_Y = tsk_info->get_const_so_field<YVolF>( "surf_out_normY" );
  ZVolFP const Norm_out_Z = tsk_info->get_const_so_field<ZVolF>( "surf_out_normZ" );
  
  // constant area fractions 
  XVolFP const areaFractionX = tsk_info->get_const_so_field<SpatialOps::XVolField>("areaFractionFX");
  YVolFP const areaFractionY = tsk_info->get_const_so_field<SpatialOps::YVolField>("areaFractionFY");
  ZVolFP const areaFractionZ = tsk_info->get_const_so_field<SpatialOps::ZVolField>("areaFractionFZ");

  // constant gas temperature 
  SVolFP WallTemperature = tsk_info->get_const_so_field<SVolF>("temperature");
  
  //Compute the probability of sticking for each face using the wall temperature.
  compute_prob_stick<SpatialOps::SSurfXField, SpatialOps::XVolField>( opr,Norm_out_X,areaFractionX,WallTemperature,WallTemperature, ProbSurfaceX );
  compute_prob_stick<SpatialOps::SSurfYField, SpatialOps::YVolField>( opr,Norm_out_Y,areaFractionY,WallTemperature,WallTemperature, ProbSurfaceY );
  compute_prob_stick<SpatialOps::SSurfZField, SpatialOps::ZVolField>( opr,Norm_out_Z,areaFractionZ,WallTemperature,WallTemperature, ProbSurfaceZ );

  for(int i=0; i<_Nenv; i++){
     const std::string ParticleTemperature_name = ParticleTools::append_env(_ParticleTemperature_base_name ,i);
    const std::string  MaxParticleTemperature_name = ParticleTools::append_env(_MaxParticleTemperature_base_name ,i);
    const std::string weight_name = ParticleTools::append_env(_weight_base_name ,i);
    const std::string rho_name = ParticleTools::append_env(_rho_base_name ,i);
    const std::string diameter_name = ParticleTools::append_env(_diameter_base_name ,i);

    const std::string xvel_name = ParticleTools::append_env(_xvel_base_name ,i);
    const std::string yvel_name = ParticleTools::append_env(_yvel_base_name ,i);
    const std::string zvel_name = ParticleTools::append_env(_zvel_base_name ,i);
    
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
    
    XVolFP RateDepositionX   = tsk_info->get_so_field<XVolF>( RateDepositionX_name   );
    YVolFP RateDepositionY   = tsk_info->get_so_field<YVolF>( RateDepositionY_name   );
    ZVolFP RateDepositionZ   = tsk_info->get_so_field<ZVolF>( RateDepositionZ_name   );
 
    SVolFP MaxParticleTemperature = tsk_info->get_const_so_field<SVolF>(MaxParticleTemperature_name);
    SVolFP ParticleTemperature = tsk_info->get_const_so_field<SVolF>(ParticleTemperature_name);
    SVolFP weight = tsk_info->get_const_so_field<SVolF>(weight_name);
    SVolFP rho = tsk_info->get_const_so_field<SVolF>(rho_name);
    SVolFP diameter = tsk_info->get_const_so_field<SVolF>(diameter_name);


    SVolFP xvel = tsk_info->get_const_so_field<SVolF>(xvel_name);
    SVolFP yvel = tsk_info->get_const_so_field<SVolF>(yvel_name);
    SVolFP zvel = tsk_info->get_const_so_field<SVolF>(zvel_name);
    
    XVolFP FluxPx = tsk_info->get_so_field<XVolF>(FluxPx_name);
    YVolFP FluxPy = tsk_info->get_so_field<YVolF>(FluxPy_name);
    ZVolFP FluxPz = tsk_info->get_so_field<ZVolF>(FluxPz_name);
   
    XVolFP ProbParticleX   =  tsk_info->get_so_field<XVolF>(ProbParticleX_name);
    YVolFP ProbParticleY   =  tsk_info->get_so_field<YVolF>(ProbParticleY_name);
    ZVolFP ProbParticleZ   =  tsk_info->get_so_field<ZVolF>(ProbParticleZ_name);
    
    XVolFP ProbDepositionX   =  tsk_info->get_so_field<XVolF>(ProbDepositionX_name);
    YVolFP ProbDepositionY   =  tsk_info->get_so_field<YVolF>(ProbDepositionY_name);
    ZVolFP ProbDepositionZ   =  tsk_info->get_so_field<ZVolF>(ProbDepositionZ_name);

    //Compute the probability of sticking for each particle using particle temperature.
    compute_prob_stick<SpatialOps::SSurfXField, SpatialOps::XVolField>( opr,Norm_in_X,areaFractionX,ParticleTemperature,MaxParticleTemperature, ProbParticleX );
    *ProbDepositionX <<= 0.5* (*ProbParticleX +(*ProbParticleX * *ProbParticleX + 4*(1-*ProbParticleX) * *ProbSurfaceX ));
    flux_compute<SpatialOps::SSurfXField, SpatialOps::XVolField>( opr,Norm_in_X,rho,xvel,weight,diameter,FluxPx );
    *RateDepositionX <<= cond( *FluxPx * *Norm_out_X > 0.0, 0.0 )
                         ( *FluxPx* *ProbDepositionX );
    
    compute_prob_stick<SpatialOps::SSurfYField, SpatialOps::YVolField>( opr,Norm_in_Y,areaFractionY,ParticleTemperature, MaxParticleTemperature, ProbParticleY );
    *ProbDepositionY <<= 0.5* (*ProbParticleY +(*ProbParticleY * *ProbParticleY + 4*(1-*ProbParticleY) * *ProbSurfaceY ));
    flux_compute<SpatialOps::SSurfYField, SpatialOps::YVolField>( opr,Norm_in_Y,rho,yvel,weight,diameter,FluxPy );
    *RateDepositionY <<= cond( *FluxPy * *Norm_out_Y > 0.0, 0.0 )
                         ( *FluxPy* *ProbDepositionY );
    //Z
    compute_prob_stick<SpatialOps::SSurfZField, SpatialOps::ZVolField>( opr,Norm_in_Z,areaFractionZ,ParticleTemperature, MaxParticleTemperature, ProbParticleZ );
    *ProbDepositionZ <<= 0.5* (*ProbParticleZ +(*ProbParticleZ * *ProbParticleZ + 4*(1-*ProbParticleZ) * *ProbSurfaceZ ));
    flux_compute<SpatialOps::SSurfZField, SpatialOps::ZVolField>( opr,Norm_in_Z,rho,zvel,weight,diameter,FluxPz );
    *RateDepositionZ <<= cond( *FluxPz * *Norm_out_Z > 0.0, 0.0 )
                         ( *FluxPz* *ProbDepositionZ );
    // for verification   
    //SpatialOps::IntVec ijk1(13,10,10);
    //std::cout << "e: " << i << " surface temperature: " << (*WallTemperature)(ijk1) << " norm_outX: " << (*Norm_out_X)(ijk1) << " probx: " << (*ProbParticleX)(ijk1) << " probSx: " << (*ProbSurfaceX)(ijk1)  << " probDepx: " << (*ProbDepositionX)(ijk1) << " fluxx: " << (*FluxPx)(ijk1) <<   std::endl; 
    //SpatialOps::IntVec ijk1cc(12,10,10);
    //double computed_fluxx = (*rho)(ijk1cc) * (*xvel)(ijk1cc) * (*weight)(ijk1cc) * 0.52 * pow((*diameter)(ijk1cc),3); // rho * vel * weight * pi/6 * D^3
    //std::cout << "my fluxx: " << computed_fluxx << " particle temperature: " << (*ParticleTemperature)(ijk1cc) << " rho: " << (*rho)(ijk1cc) << " xvel: " << (*xvel)(ijk1cc) << " weight: " << (*weight)(ijk1cc) << " diam: " << (*diameter)(ijk1cc) << std::endl; 
  
  }
    
}

//
//------------------------------------------------
//------------- BOUNDARY CONDITIONS --------------
//------------------------------------------------
//

void
RateDeposition::register_compute_bcs( std::vector<ArchesFieldContainer::VariableInformation>& variable_registry, const int time_substep ){
}
void
RateDeposition::compute_bcs( const Patch* patch, ArchesTaskInfoManager* tsk_info,
                        SpatialOps::OperatorDatabase& opr ){


  using namespace SpatialOps;
  using SpatialOps::operator *;

}



