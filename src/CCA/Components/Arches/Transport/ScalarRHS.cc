#include <CCA/Components/Arches/Transport/ScalarRHS.h>
#include <CCA/Components/Arches/Operators/Operators.h>
#include <CCA/Components/Arches/TransportEqns/Discretization_new.h>
#include <spatialops/structured/FVStaggered.h>

using namespace Uintah;

using namespace SpatialOps;
using SpatialOps::operator *; 
typedef SSurfXField SurfX;
typedef SSurfYField SurfY;
typedef SSurfZField SurfZ;
typedef SVolField   SVolF;
typedef SpatialOps::SpatFldPtr<SVolF> SVolFP; 
typedef SpatialOps::SpatFldPtr<SpatialOps::XVolField> XVolPtr; 
typedef SpatialOps::SpatFldPtr<SpatialOps::YVolField> YVolPtr; 
typedef SpatialOps::SpatFldPtr<SpatialOps::ZVolField> ZVolPtr; 
//gradient
typedef BasicOpTypes<SVolF>::GradX GradX;
typedef BasicOpTypes<SVolF>::GradY GradY;
typedef BasicOpTypes<SVolF>::GradZ GradZ;
//interpolants
typedef BasicOpTypes<SVolF>::InterpC2FX InterpX;
typedef BasicOpTypes<SVolF>::InterpC2FY InterpY;
typedef BasicOpTypes<SVolF>::InterpC2FZ InterpZ;
//divergence
typedef BasicOpTypes<SVolF>::DivX DivX;
typedef BasicOpTypes<SVolF>::DivY DivY;
typedef BasicOpTypes<SVolF>::DivZ DivZ;


ScalarRHS::ScalarRHS( std::string task_name, int matl_index ) : 
TaskInterface( task_name, matl_index ) { 

  _rhs_name = task_name+"_RHS";
  _D_name = task_name+"_D"; 
  _Fconv_name = task_name+"_Fconv"; 
  _Fdiff_name = task_name+"_Fdiff"; 

  _disc = scinew Discretization_new(); 

}

ScalarRHS::~ScalarRHS(){ 

  delete _disc; 
}

void 
ScalarRHS::problemSetup( ProblemSpecP& db ){ 

  _do_conv = false; 
  if ( db->findBlock("convection")){ 
    db->findBlock("convection")->getAttribute("scheme", _conv_scheme); 
    _do_conv = true; 
  }

  _do_diff = false; 
  if ( db->findBlock("diffusion")){ 
    _do_diff = true; 
  }

  _do_clip = false;
  if ( db->findBlock("clip")){ 
    _do_clip = true; 
    db->findBlock("clip")->getAttribute("low", _low_clip); 
    db->findBlock("clip")->getAttribute("high", _high_clip); 
  }

  
  for (ProblemSpecP src_db = db->findBlock("src"); src_db != 0; src_db = src_db->findNextBlock("src")){

    std::string src_label; 
    double weight = 1.0; 

    src_db->getAttribute("label",src_label); 

    if ( src_db->findBlock("weight")){ 
      src_db->findBlock("weight")->getAttribute("value",weight); 
    }

    SourceInfo info; 
    info.name = src_label; 
    info.weight = weight; 

    _source_info.push_back(info); 

  }

}

//
//------------------------------------------------
//-------------- INITIALIZATION ------------------
//------------------------------------------------
//

void 
ScalarRHS::register_initialize( std::vector<VariableInformation>& variable_registry ){ 

  //FUNCITON CALL     STRING NAME(VL)     TYPE       DEPENDENCY    GHOST DW     VR
  register_variable(  _rhs_name  , CC_DOUBLE , LOCAL_COMPUTES , 0 , NEWDW , variable_registry );
  register_variable(  _task_name , CC_DOUBLE , LOCAL_COMPUTES , 0 , NEWDW , variable_registry );
  register_variable(  _D_name    , CC_DOUBLE , LOCAL_COMPUTES , 0 , NEWDW , variable_registry );
  register_variable(  _Fconv_name, CC_DOUBLE , LOCAL_COMPUTES , 0 , NEWDW , variable_registry );
  register_variable(  _Fdiff_name, CC_DOUBLE , LOCAL_COMPUTES , 0 , NEWDW , variable_registry );

}

void 
ScalarRHS::initialize( const Patch* patch, ArchesTaskInfoManager* tsk_info, 
                       SpatialOps::OperatorDatabase& opr ){ 

  using namespace SpatialOps;
  using SpatialOps::operator *; 

  SVolFP rhs   = tsk_info->get_so_field<SVolF>(_rhs_name);
  SVolFP phi   = tsk_info->get_so_field<SVolF>(_task_name);
  SVolFP gamma = tsk_info->get_so_field<SVolF>(_D_name);
  SVolFP Fdiff = tsk_info->get_so_field<SVolF>(_Fdiff_name);
  SVolFP Fconv = tsk_info->get_so_field<SVolF>(_Fconv_name);

  *rhs <<= 0.0;
  *phi <<= 0.0;
  *gamma <<= 0.0001; 
  *Fdiff <<= 0.0;
  *Fconv <<= 0.0; 

}
//
//------------------------------------------------
//------------- TIMESTEP INIT --------------------
//------------------------------------------------
//
void 
ScalarRHS::register_timestep_init( std::vector<VariableInformation>& variable_registry ){ 
  register_variable( _D_name    , CC_DOUBLE , COMPUTES , 0 , NEWDW , variable_registry  );
  register_variable( _D_name    , CC_DOUBLE , REQUIRES , 0 , OLDDW , variable_registry );
  register_variable( _task_name , CC_DOUBLE , COMPUTES , 0 , NEWDW , variable_registry  );
  register_variable( _task_name , CC_DOUBLE , REQUIRES , 0 , OLDDW , variable_registry  );
}

void 
ScalarRHS::timestep_init( const Patch* patch, ArchesTaskInfoManager* tsk_info, 
                          SpatialOps::OperatorDatabase& opr ){ 

  using namespace SpatialOps;
  using SpatialOps::operator *; 

  SVolFP gamma     = tsk_info->get_so_field<SVolF>( _D_name );
  SVolFP old_gamma = tsk_info->get_const_so_field<SVolF>( _D_name );
  SVolFP phi       = tsk_info->get_so_field<SVolF>( _task_name );
  SVolFP old_phi   = tsk_info->get_const_so_field<SVolF>( _task_name );

  *gamma <<= *old_gamma;
  *phi <<= *old_phi;

}


//
//------------------------------------------------
//------------- TIMESTEP WORK --------------------
//------------------------------------------------
//

void 
ScalarRHS::register_timestep_eval( std::vector<VariableInformation>& variable_registry, const int time_substep ){ 

//  //FUNCITON CALL     STRING NAME(VL)     TYPE       DEPENDENCY    GHOST DW     VR
  register_variable( _rhs_name        , CC_DOUBLE , COMPUTES , 0 , NEWDW  , variable_registry , time_substep );
  register_variable( _D_name          , CC_DOUBLE , REQUIRES,  1 , NEWDW  , variable_registry , time_substep );
  register_variable( _task_name       , CC_DOUBLE , REQUIRES , 1 , LATEST , variable_registry , time_substep );
  register_variable( _Fconv_name      , CC_DOUBLE , COMPUTES , 0 , NEWDW  , variable_registry , time_substep );
  register_variable( _Fdiff_name      , CC_DOUBLE , COMPUTES , 0 , NEWDW  , variable_registry , time_substep );
  register_variable( "uVelocitySPBC"  , FACEX     , REQUIRES , 1 , LATEST , variable_registry , time_substep );
  register_variable( "vVelocitySPBC"  , FACEY     , REQUIRES , 1 , LATEST , variable_registry , time_substep );
  register_variable( "wVelocitySPBC"  , FACEZ     , REQUIRES , 1 , LATEST , variable_registry , time_substep );
  register_variable( "areaFractionFX" , FACEX     , REQUIRES , 1 , OLDDW  , variable_registry , time_substep );
  register_variable( "areaFractionFY" , FACEY     , REQUIRES , 1 , OLDDW  , variable_registry , time_substep );
  register_variable( "areaFractionFZ" , FACEZ     , REQUIRES , 1 , OLDDW  , variable_registry , time_substep );
  register_variable( "density"        , CC_DOUBLE , REQUIRES , 1 , LATEST , variable_registry , time_substep );
//  //register_variable( "areaFraction"   , CC_VEC    , REQUIRES , 2 , LATEST , variable_registry , time_substep );
//
//  typedef std::vector<SourceInfo> VS; 
//  for (VS::iterator i = _source_info.begin(); i != _source_info.end(); i++){ 
//    register_variable( i->name, CC_DOUBLE, REQUIRES, 0, LATEST, variable_registry, time_substep ); 
//  }

}

void 
ScalarRHS::eval( const Patch* patch, ArchesTaskInfoManager* tsk_info, 
                 SpatialOps::OperatorDatabase& opr ){ 


  using namespace SpatialOps;
  using SpatialOps::operator *; 
  SVolFP rhs       = tsk_info->get_so_field<SVolF>( _rhs_name        );

  SVolFP Fdiff     = tsk_info->get_so_field<SVolF>( _Fdiff_name      );
  SVolFP Fconv     = tsk_info->get_so_field<SVolF>( _Fconv_name      );
  SVolFP phi   = tsk_info->get_const_so_field<SVolF>( _task_name       );
  XVolPtr epsX = tsk_info->get_const_so_field<SpatialOps::XVolField>( "areaFractionFX" );
  YVolPtr epsY = tsk_info->get_const_so_field<SpatialOps::YVolField>( "areaFractionFY" );
  ZVolPtr epsZ = tsk_info->get_const_so_field<SpatialOps::ZVolField>( "areaFractionFZ" );
  SVolFP rho   = tsk_info->get_const_so_field<SVolF>( "density"        );
  SVolFP gamma = tsk_info->get_const_so_field<SVolF>( _D_name          );

  XVolPtr const u         = tsk_info->get_const_so_field<SpatialOps::XVolField>( "uVelocitySPBC" );
  YVolPtr const v         = tsk_info->get_const_so_field<SpatialOps::YVolField>( "vVelocitySPBC" );
  ZVolPtr const w         = tsk_info->get_const_so_field<SpatialOps::ZVolField>( "wVelocitySPBC" );

  //operators: 
  const GradX* const gradx = opr.retrieve_operator<GradX>();
  const GradY* const grady = opr.retrieve_operator<GradY>();
  const GradZ* const gradz = opr.retrieve_operator<GradZ>();

  const InterpX* const ix = opr.retrieve_operator<InterpX>();
  const InterpY* const iy = opr.retrieve_operator<InterpY>();
  const InterpZ* const iz = opr.retrieve_operator<InterpZ>();

  const DivX* const dx = opr.retrieve_operator<DivX>();
  const DivY* const dy = opr.retrieve_operator<DivY>();
  const DivZ* const dz = opr.retrieve_operator<DivZ>();

  Vector DX = patch->dCell(); 
  double vol = DX.x()*DX.y()*DX.z(); 
  const double dt = tsk_info->get_dt(); 

  typedef OperatorTypeBuilder< SpatialOps::Interpolant, SpatialOps::XVolField, SpatialOps::SSurfXField >::type InterpTX;
  typedef OperatorTypeBuilder< SpatialOps::Interpolant, SpatialOps::YVolField, SpatialOps::SSurfYField >::type InterpTY;
  typedef OperatorTypeBuilder< SpatialOps::Interpolant, SpatialOps::ZVolField, SpatialOps::SSurfZField >::type InterpTZ;

  const InterpTX* const interpx = opr.retrieve_operator<InterpTX>();
  const InterpTY* const interpy = opr.retrieve_operator<InterpTY>();
  const InterpTZ* const interpz = opr.retrieve_operator<InterpTZ>();

  SpatialOps::SpatFldPtr<SSurfXField> phiLowX = SpatialOps::SpatialFieldStore::get<SSurfXField>( *u );
  SpatialOps::SpatFldPtr<SSurfXField> ufx     = SpatialOps::SpatialFieldStore::get<SSurfXField>( *u );
  typedef UpwindInterpolant<SVolF,SurfX> UpwindX; 
  UpwindX* upx = opr.retrieve_operator<UpwindX>();  

  *ufx <<= (*interpx)(*u); 

  SpatialOps::SpatFldPtr<SSurfYField> phiLowY = SpatialOps::SpatialFieldStore::get<SSurfYField>( *v );
  SpatialOps::SpatFldPtr<SSurfYField> vfy     = SpatialOps::SpatialFieldStore::get<SSurfYField>( *v );
  typedef UpwindInterpolant<SVolF,SurfY> UpwindY; 
  UpwindY* upy = opr.retrieve_operator<UpwindY>();  

  *vfy <<= (*interpy)(*v); 

  SpatialOps::SpatFldPtr<SSurfZField> phiLowZ = SpatialOps::SpatialFieldStore::get<SSurfZField>( *w );
  SpatialOps::SpatFldPtr<SSurfZField> wfz     = SpatialOps::SpatialFieldStore::get<SSurfZField>( *w );
  typedef UpwindInterpolant<SVolF,SurfZ> UpwindZ; 
  UpwindZ* upz = opr.retrieve_operator<UpwindZ>();  

  *wfz <<= (*interpz)(*w); 

  //
  //--------------- actual work below this line ---------------------
  //

  //diffusion: 
  if ( _do_diff ){ 

    *Fdiff <<= (*dx)( (*ix)( *gamma * *rho ) * (*gradx)(*phi)* (*interpx)(*epsX) )
             + (*dy)( (*iy)( *gamma * *rho ) * (*grady)(*phi)* (*interpy)(*epsY) )
             + (*dz)( (*iz)( *gamma * *rho ) * (*gradz)(*phi)* (*interpz)(*epsZ) );

  } else { 

    *Fdiff <<= 0.0; 

  }

  //convection: 
  if ( _do_conv ){
    //upwind fluxes for now...
    upx->set_advective_velocity( *ufx ); 
    upx->apply_to_field(*phi, *phiLowX);

    upy->set_advective_velocity( *vfy ); 
    upy->apply_to_field(*phi, *phiLowY);

    upz->set_advective_velocity( *wfz ); 
    upz->apply_to_field(*phi, *phiLowZ);

    *Fconv <<= (*dx)(*phiLowX) + (*dy)(*phiLowY) + (*dz)(*phiLowZ); 

  } else { 

    *Fconv <<= 0; 

  } 

  *rhs <<= *rho * *phi + dt * ( *Fdiff - *Fconv );

  //add sources:
  typedef std::vector<SourceInfo> VS; 
  for (VS::iterator i = _source_info.begin(); i != _source_info.end(); i++){ 

    SVolFP const src = tsk_info->get_const_so_field<SVolF>( i->name );

    *rhs <<= *rhs + dt * i->weight * *src;

  }

}

  //mask example
  //std::vector<SpatialOps::IntVec> maskset;  

  //for (int i = 0; i < 11; i++){ 
    //maskset.push_back(SpatialOps::IntVec(4,i,1)); 
  //}

  //SpatialOps::SpatialMask<SpatialOps::SVolField> mask(*phi,maskset); 

  //*Fdiff <<= cond( mask, 3.0 )
                 //( *Fdiff ); 

//
//  //-->convection: 
//  if ( _do_conv ){ 
//    //not working yet:
//    //_disc->computeConv( patch, *ui_fconv, *ui_old_phi, *ui_u, *ui_v, *ui_w, *ui_rho, *ui_eps, _conv_scheme );
//  } else { 
//    *Fconv <<= 0.0; 
//  }
//
//  //Divide by volume because Nebo is using a differential form
//  //and the computeConv function is finite volume.
//  *rhs <<= *rho * *phi + dt * ( *Fdiff - *Fconv/vol ) ;
