#include <CCA/Components/Arches/Transport/ScalarRHS.h>
#include <CCA/Components/Arches/Operators/Operators.h>
#include <CCA/Components/Arches/BoundaryCond_new.h>
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

}

ScalarRHS::~ScalarRHS(){

}

void
ScalarRHS::problemSetup( ProblemSpecP& db ){

  _do_conv = false;
  if ( db->findBlock("convection")){
    db->findBlock("convection")->getAttribute("scheme", _conv_scheme);
    _do_conv = true;

    if ( _conv_scheme == "superbee"){
      _limiter_type = Wasatch::SUPERBEE;
    } else if ( _conv_scheme == "central"){
      _limiter_type = Wasatch::CENTRAL;
    } else if ( _conv_scheme == "upwind"){
      _limiter_type = Wasatch::UPWIND;
    } else if ( _conv_scheme == "charm"){
      _limiter_type = Wasatch::CHARM;
    } else if ( _conv_scheme == "koren"){
      _limiter_type = Wasatch::KOREN;
    } else if ( _conv_scheme == "mc"){
      _limiter_type = Wasatch::MC;
    } else if ( _conv_scheme == "ospre"){
      _limiter_type = Wasatch::OSPRE;
    } else if ( _conv_scheme == "smart"){
      _limiter_type = Wasatch::SMART;
    } else if ( _conv_scheme == "vanleer"){
      _limiter_type = Wasatch::VANLEER;
    } else if ( _conv_scheme == "hcus"){
      _limiter_type = Wasatch::HCUS;
    } else if ( _conv_scheme == "minmod"){
      _limiter_type = Wasatch::MINMOD;
    } else if ( _conv_scheme == "hquick"){
      _limiter_type = Wasatch::HQUICK;
    } else {
      throw InvalidValue("Error: Convection scheme not supported for scalar.",__FILE__,__LINE__);
    }

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

void
ScalarRHS::create_local_labels(){

  register_new_variable<CCVariable<double> >( _rhs_name );
  register_new_variable<CCVariable<double> >( _task_name );
  register_new_variable<CCVariable<double> >( _D_name );
  register_new_variable<CCVariable<double> >( _Fconv_name );
  register_new_variable<CCVariable<double> >( _Fdiff_name );

}

//
//------------------------------------------------
//-------------- INITIALIZATION ------------------
//------------------------------------------------
//

void
ScalarRHS::register_initialize( std::vector<ArchesFieldContainer::VariableInformation>& variable_registry ){

  //FUNCITON CALL     STRING NAME(VL)     TYPE       DEPENDENCY    GHOST DW     VR
  register_variable(  _rhs_name   , ArchesFieldContainer::COMPUTES , 0 , ArchesFieldContainer::NEWDW , variable_registry );
  register_variable(  _task_name  , ArchesFieldContainer::COMPUTES , 0 , ArchesFieldContainer::NEWDW , variable_registry );
  register_variable(  _D_name     , ArchesFieldContainer::COMPUTES , 0 , ArchesFieldContainer::NEWDW , variable_registry );
  register_variable(  _Fconv_name , ArchesFieldContainer::COMPUTES , 0 , ArchesFieldContainer::NEWDW , variable_registry );
  register_variable(  _Fdiff_name , ArchesFieldContainer::COMPUTES , 0 , ArchesFieldContainer::NEWDW , variable_registry );

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
ScalarRHS::register_timestep_init( std::vector<ArchesFieldContainer::VariableInformation>& variable_registry ){
  register_variable( _D_name     , ArchesFieldContainer::COMPUTES , 0 , ArchesFieldContainer::NEWDW , variable_registry  );
  register_variable( _D_name     , ArchesFieldContainer::REQUIRES , 0 , ArchesFieldContainer::OLDDW , variable_registry );
  register_variable( _task_name  , ArchesFieldContainer::COMPUTES , 0 , ArchesFieldContainer::NEWDW , variable_registry  );
  register_variable( _task_name  , ArchesFieldContainer::REQUIRES , 0 , ArchesFieldContainer::OLDDW , variable_registry  );
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
ScalarRHS::register_timestep_eval( std::vector<ArchesFieldContainer::VariableInformation>& variable_registry, const int time_substep ){

//  //FUNCITON CALL     STRING NAME(VL)     DEPENDENCY    GHOST DW     VR
  register_variable( _rhs_name        , ArchesFieldContainer::COMPUTES , 0 , ArchesFieldContainer::NEWDW  , variable_registry , time_substep );
  register_variable( _D_name          , ArchesFieldContainer::REQUIRES,  1 , ArchesFieldContainer::NEWDW  , variable_registry , time_substep );
  register_variable( _task_name       , ArchesFieldContainer::REQUIRES , 1 , ArchesFieldContainer::LATEST , variable_registry , time_substep );
  register_variable( _Fconv_name      , ArchesFieldContainer::COMPUTES , 0 , ArchesFieldContainer::NEWDW  , variable_registry , time_substep );
  register_variable( _Fdiff_name      , ArchesFieldContainer::COMPUTES , 0 , ArchesFieldContainer::NEWDW  , variable_registry , time_substep );
  register_variable( "uVelocitySPBC"  , ArchesFieldContainer::REQUIRES , 1 , ArchesFieldContainer::LATEST , variable_registry , time_substep );
  register_variable( "vVelocitySPBC"  , ArchesFieldContainer::REQUIRES , 1 , ArchesFieldContainer::LATEST , variable_registry , time_substep );
  register_variable( "wVelocitySPBC"  , ArchesFieldContainer::REQUIRES , 1 , ArchesFieldContainer::LATEST , variable_registry , time_substep );
  register_variable( "areaFractionFX" , ArchesFieldContainer::REQUIRES , 1 , ArchesFieldContainer::OLDDW  , variable_registry , time_substep );
  register_variable( "areaFractionFY" , ArchesFieldContainer::REQUIRES , 1 , ArchesFieldContainer::OLDDW  , variable_registry , time_substep );
  register_variable( "areaFractionFZ" , ArchesFieldContainer::REQUIRES , 1 , ArchesFieldContainer::OLDDW  , variable_registry , time_substep );
  register_variable( "volFraction"    , ArchesFieldContainer::REQUIRES , 1 , ArchesFieldContainer::OLDDW  , variable_registry , time_substep );
  register_variable( "density"        , ArchesFieldContainer::REQUIRES , 1 , ArchesFieldContainer::LATEST , variable_registry , time_substep );
//  //register_variable( "areaFraction"   , ArchesFieldContainer::REQUIRES , 2 , ArchesFieldContainer::LATEST , variable_registry , time_substep );
//
//  typedef std::vector<SourceInfo> VS;
//  for (VS::iterator i = _source_info.begin(); i != _source_info.end(); i++){
//    register_variable( i->name, ArchesFieldContainer::REQUIRES, 0, ArchesFieldContainer::LATEST, variable_registry, time_substep );
//  }

}

void
ScalarRHS::eval( const Patch* patch, ArchesTaskInfoManager* tsk_info,
                 SpatialOps::OperatorDatabase& opr ){


  using namespace SpatialOps;
  using SpatialOps::operator *;

  //variables:
  SVolFP rhs   = tsk_info->get_so_field<SVolF>( _rhs_name        );
  SVolFP Fdiff = tsk_info->get_so_field<SVolF>( _Fdiff_name      );
  SVolFP Fconv = tsk_info->get_so_field<SVolF>( _Fconv_name      );
  SVolFP phi   = tsk_info->get_const_so_field<SVolF>( _task_name       );
  XVolPtr epsX = tsk_info->get_const_so_field<SpatialOps::XVolField>( "areaFractionFX" );
  YVolPtr epsY = tsk_info->get_const_so_field<SpatialOps::YVolField>( "areaFractionFY" );
  ZVolPtr epsZ = tsk_info->get_const_so_field<SpatialOps::ZVolField>( "areaFractionFZ" );
  SVolFP rho   = tsk_info->get_const_so_field<SVolF>( "density"        );
  SVolFP gamma = tsk_info->get_const_so_field<SVolF>( _D_name          );
  SVolFP eps   = tsk_info->get_const_so_field<SVolF>( "volFraction" );

  XVolPtr const u = tsk_info->get_const_so_field<SpatialOps::XVolField>( "uVelocitySPBC" );
  YVolPtr const v = tsk_info->get_const_so_field<SpatialOps::YVolField>( "vVelocitySPBC" );
  ZVolPtr const w = tsk_info->get_const_so_field<SpatialOps::ZVolField>( "wVelocitySPBC" );

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

  typedef OperatorTypeBuilder< SpatialOps::Interpolant, SpatialOps::XVolField, SpatialOps::SSurfXField >::type InterpTX;
  typedef OperatorTypeBuilder< SpatialOps::Interpolant, SpatialOps::YVolField, SpatialOps::SSurfYField >::type InterpTY;
  typedef OperatorTypeBuilder< SpatialOps::Interpolant, SpatialOps::ZVolField, SpatialOps::SSurfZField >::type InterpTZ;

  const InterpTX* const interpx = opr.retrieve_operator<InterpTX>();
  const InterpTY* const interpy = opr.retrieve_operator<InterpTY>();
  const InterpTZ* const interpz = opr.retrieve_operator<InterpTZ>();

  //Vector DX = patch->dCell();
  //double vol = DX.x()*DX.y()*DX.z();
  const double dt = tsk_info->get_dt();

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
  *Fconv <<= 0.0;
  if ( _do_conv ){

    //X
    compute_convective_flux<SpatialOps::SSurfXField, SpatialOps::XVolField, DivX>( opr, dx, u, epsX, phi, rho, eps, Fconv );
    //Y
    compute_convective_flux<SpatialOps::SSurfYField, SpatialOps::YVolField, DivY>( opr, dy, v, epsY, phi, rho, eps, Fconv );
    //Z
    compute_convective_flux<SpatialOps::SSurfZField, SpatialOps::ZVolField, DivZ>( opr, dz, w, epsZ, phi, rho, eps, Fconv );

  }

  *rhs <<= *rho * *phi + dt * ( *Fdiff - *Fconv );

  //add sources:
  typedef std::vector<SourceInfo> VS;
  for (VS::iterator i = _source_info.begin(); i != _source_info.end(); i++){

    SVolFP const src = tsk_info->get_const_so_field<SVolF>( i->name );

    *rhs <<= *rhs + dt * i->weight * *src;

  }
}

//
//------------------------------------------------
//------------- BOUNDARY CONDITIONS --------------
//------------------------------------------------
//

void
ScalarRHS::register_compute_bcs( std::vector<ArchesFieldContainer::VariableInformation>& variable_registry, const int time_substep ){
}

void
ScalarRHS::compute_bcs( const Patch* patch, ArchesTaskInfoManager* tsk_info,
                        SpatialOps::OperatorDatabase& opr ){


  using namespace SpatialOps;
  using SpatialOps::operator *;

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
  //------------test run on new mask stuff-------------//
  ////create a base boundary object which holds the masks
  //BoundaryCondition_new::MaskContainer<SpatialOps::SVolField> bc;
  //SpatialOps::IntVec ijk(1,1,0);
  //std::vector<SpatialOps::IntVec> pass_ijk;
  //pass_ijk.push_back(ijk);

  ////create the mask with the points created above
  //bc.create_mask( patch, 0, pass_ijk, BoundaryCondition_new::BOUNDARY_FACE );
  ////insert the object into permanent storage
  ////make a fake bc:
  //BoundaryCondition_new::NameToSVolMask some_boundary;
  //std::string bc_name = "some_boundary";
  //some_boundary.insert(std::make_pair(bc_name,bc));
  ////now insert it into the patch id->mask storage
  //BoundaryCondition_new::patch_svol_masks.insert(std::make_pair(0,some_boundary));

  ////---get the mask back--
  //const int pid = 0;
  ////retrieve a reference to the object which contains the mask
  //BoundaryCondition_new::MaskContainer<SpatialOps::SVolField>& bc_ref = BoundaryCondition_new::get_bc_info<SpatialOps::SVolField>(pid,bc_name);
  ////get a mask pointer
  //SpatialOps::SpatialMask<SpatialOps::SVolField>* a_mask = bc_ref.get_mask(BoundaryCondition_new::BOUNDARY_FACE);

  //*rhs <<= cond(*a_mask, 0.0)
               //(*rhs);

  //// Given a variable and a patch.
  //// loop through all bc's on this patch
  //// get the bc type for this variable
  //// given the bc name + patch, get the mask
  //// apply bcs


  //----------end test run on new mask stuff-------------//
