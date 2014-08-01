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
ScalarRHS::initialize( const Patch* patch, FieldCollector* field_collector, 
                       SpatialOps::OperatorDatabase& opr ){ 

  using namespace SpatialOps;
  using SpatialOps::operator *; 
  typedef SpatialOps::SVolField   SVolF;

  SVolF* const rhs   = field_collector->get_so_field<SVolF>( _rhs_name   );
  SVolF* const phi   = field_collector->get_so_field<SVolF>( _task_name  );
  SVolF* const gamma = field_collector->get_so_field<SVolF>( _D_name     );
  SVolF* const Fdiff = field_collector->get_so_field<SVolF>( _Fdiff_name );
  SVolF* const Fconv = field_collector->get_so_field<SVolF>( _Fconv_name );

  *rhs <<= 0.0;
  *phi <<= 0.0;
  *gamma <<= 0.0; 
  *Fdiff <<= 0.0;
  *Fconv <<= 0.0; 

}

//
//------------------------------------------------
//------------- TIMESTEP WORK --------------------
//------------------------------------------------
//

void 
ScalarRHS::register_all_variables( std::vector<VariableInformation>& variable_registry, const int time_substep ){ 

  //FUNCITON CALL     STRING NAME(VL)     TYPE       DEPENDENCY    GHOST DW     VR
  register_variable( _rhs_name        , CC_DOUBLE , COMPUTES , 0 , NEWDW  , variable_registry , time_substep );
  register_variable( _D_name          , CC_DOUBLE , COMPUTES , 0 , NEWDW  , variable_registry , time_substep );
  register_variable( _task_name       , CC_DOUBLE , COMPUTES , 0 , NEWDW  , variable_registry , time_substep );
  register_variable( _task_name       , CC_DOUBLE , REQUIRES , 2 , LATEST , variable_registry , time_substep );
  register_variable( _D_name          , CC_DOUBLE , REQUIRES , 1 , LATEST , variable_registry , time_substep );
  register_variable( _Fconv_name      , CC_DOUBLE , COMPUTES , 0 , NEWDW  , variable_registry , time_substep );
  register_variable( _Fdiff_name      , CC_DOUBLE , COMPUTES , 0 , NEWDW  , variable_registry , time_substep );
  register_variable( "uVelocitySPBC"  , FACEX     , REQUIRES , 1 , LATEST , variable_registry , time_substep );
  register_variable( "vVelocitySPBC"  , FACEY     , REQUIRES , 1 , LATEST , variable_registry , time_substep );
  register_variable( "wVelocitySPBC"  , FACEZ     , REQUIRES , 1 , LATEST , variable_registry , time_substep );
  register_variable( "areaFractionFX" , FACEX     , REQUIRES , 1 , OLDDW  , variable_registry , time_substep );
  register_variable( "areaFractionFY" , FACEY     , REQUIRES , 1 , OLDDW  , variable_registry , time_substep );
  register_variable( "areaFractionFZ" , FACEZ     , REQUIRES , 1 , OLDDW  , variable_registry , time_substep );
  register_variable( "density"        , CC_DOUBLE , REQUIRES , 1 , LATEST , variable_registry , time_substep );
  register_variable( "areaFraction"        , CC_VEC , REQUIRES , 2 , LATEST , variable_registry , time_substep );

  typedef std::vector<SourceInfo> VS; 
  for (VS::iterator i = _source_info.begin(); i != _source_info.end(); i++){ 
    register_variable( i->name, CC_DOUBLE, REQUIRES, 0, LATEST, variable_registry, time_substep ); 
  }

}

void 
ScalarRHS::eval( const Patch* patch, FieldCollector* field_collector, 
                 SpatialOps::OperatorDatabase& opr, 
                 SchedToTaskInfo& info ){ 


  using namespace SpatialOps;
  using SpatialOps::operator *; 
  SVolF* const rhs       = field_collector->get_so_field<SVolF>( _rhs_name         );
  SVolF* const old_phi   = field_collector->get_so_field<SVolF>( _task_name        );
  SVolF* const phi       = field_collector->get_so_field<SVolF>( _task_name        );
  SVolF* const rho       = field_collector->get_so_field<SVolF>( "density"         );
  SVolF* const gamma     = field_collector->get_so_field<SVolF>( _D_name           );
  SVolF* const old_gamma = field_collector->get_so_field<SVolF>( _D_name           );
  SVolF* const Fdiff     = field_collector->get_so_field<SVolF>( _Fdiff_name       );
  SVolF* const Fconv     = field_collector->get_so_field<SVolF>( _Fconv_name       );
  SurfX* const epsX      = field_collector->get_so_field<SurfX>( "areaFractionFX" );
  SurfY* const epsY      = field_collector->get_so_field<SurfY>( "areaFractionFY" );
  SurfZ* const epsZ      = field_collector->get_so_field<SurfZ>( "areaFractionFZ" );
  SurfX* const u         = field_collector->get_so_field<SurfX>( "uVelocitySPBC"  );
  SurfY* const v         = field_collector->get_so_field<SurfY>( "vVelocitySPBC"  );
  SurfZ* const w         = field_collector->get_so_field<SurfZ>( "wVelocitySPBC"  );

  //Note: we can just grab references to the Uintah grid types here because the registration 
  //process has already required the variables 
  //For now, we will be computing convection the old fashioned way until SpatialOps gets
  //flux limiters. 
  CCVariable<double>* ui_fconv        = field_collector->get_uintah_field<CCVariable<double> >(_Fconv_name            );
  constCCVariable<double>* ui_rho     = field_collector->get_uintah_field<constCCVariable<double> >("density"         );
  constCCVariable<double>* ui_old_phi = field_collector->get_uintah_field<constCCVariable<double> >(_task_name        );
  constCCVariable<Vector>* ui_eps     = field_collector->get_uintah_field<constCCVariable<Vector> >("areaFraction"    );
  constSFCXVariable<double>* ui_u     = field_collector->get_uintah_field<constSFCXVariable<double> >("uVelocitySPBC" );
  constSFCYVariable<double>* ui_v     = field_collector->get_uintah_field<constSFCYVariable<double> >("vVelocitySPBC" );
  constSFCZVariable<double>* ui_w     = field_collector->get_uintah_field<constSFCZVariable<double> >("wVelocitySPBC" );

  const GradX* const gradx = opr.retrieve_operator<GradX>();
  const GradY* const grady = opr.retrieve_operator<GradY>();
  const GradZ* const gradz = opr.retrieve_operator<GradZ>();

  const InterpX* const ix = opr.retrieve_operator<InterpX>();
  const InterpY* const iy = opr.retrieve_operator<InterpY>();
  const InterpZ* const iz = opr.retrieve_operator<InterpZ>();

  const DivX* const dx = opr.retrieve_operator<DivX>();
  const DivY* const dy = opr.retrieve_operator<DivY>();
  const DivZ* const dz = opr.retrieve_operator<DivZ>();

  //-->carry forward gamma and phi:
  *gamma <<= *old_gamma;
  *phi <<= *old_phi; 

  Vector DX = patch->dCell(); 
  double vol = DX.x()*DX.y()*DX.z(); 

  //
  //--------------- actual work below this line ---------------------
  //
  
  //-->diffusion: 
  if ( _do_diff ){ 
    *Fdiff <<= (*dx)( (*ix)( *old_gamma * *rho ) * (*gradx)(*old_phi) * *epsX )
             + (*dy)( (*iy)( *old_gamma * *rho ) * (*grady)(*old_phi) * *epsY )
             + (*dz)( (*iz)( *old_gamma * *rho ) * (*gradz)(*old_phi) * *epsZ );
  } else { 
    *Fdiff <<= 0.0; 
  }

  //-->convection: 
  if ( _do_conv ){ 
    _disc->computeConv( patch, *ui_fconv, *ui_old_phi, *ui_u, *ui_v, *ui_w, *ui_rho, *ui_eps, _conv_scheme );
  } else { 
    *Fconv <<= 0.0; 
  }

  //Divide by volume because Nebo is using a differential form
  //and the computeConv function is finite volume.
  *rhs <<= *rho * *old_phi + info.dt * ( *Fdiff - *Fconv/vol );

  //-->add sources
  typedef std::vector<SourceInfo> VS; 
  for (VS::iterator i = _source_info.begin(); i != _source_info.end(); i++){ 

    SVolF* const src = field_collector->get_so_field<SVolF>( i->name );

    *rhs <<= *rhs + info.dt * i->weight * *src;

  }

}
