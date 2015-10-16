#include <CCA/Components/Arches/Transport/KScalarRHS.h>
#include <CCA/Components/Arches/Operators/Operators.h>
#include <CCA/Components/Arches/BoundaryCond_new.h>
#include <spatialops/structured/FVStaggered.h>

using namespace Uintah;

KScalarRHS::KScalarRHS( std::string task_name, int matl_index ) :
TaskInterface( task_name, matl_index ) {

  _rhs_name = task_name+"_RHS";
  _D_name = task_name+"_D";
  _Fconv_name = task_name+"_Fconv";
  _Fdiff_name = task_name+"_Fdiff";

}

KScalarRHS::~KScalarRHS(){

}

void
KScalarRHS::problemSetup( ProblemSpecP& db ){

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
KScalarRHS::create_local_labels(){

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
namespace {
  struct VariableInitializeFunctor{

    CCVariable<double>& var;
    double value;

    VariableInitializeFunctor( CCVariable<double>& var, double value ) : var(var), value(value){}

    void operator()(int i, int j, int k) const {

      const IntVector c(i,j,k);

      var[c] = value;

    }
  };

  template<typename UT>
  struct ComputeConvection{

    constCCVariable<double>& phi;
    constCCVariable<double>& rho;
    CCVariable<double>& rhs;
    UT& u;
    IntVector dir;
    double A;

    ComputeConvection( constCCVariable<double>& phi, CCVariable<double>& rhs,
      constCCVariable<double>& rho, UT& u, IntVector dir, double A)
      : phi(phi), rhs(rhs), rho(rho), u(u), dir(dir), A(A){}

    void
    operator()(int i, int j, int k ){

      IntVector c(i,j,k);
      IntVector cm(i,j,k);
      IntVector cp(i,j,k);
      IntVector cmm(i,j,k);
      IntVector cpp(i,j,k);
      cm -= dir;
      cp -= dir;
      cmm -= 2*dir;
      cpp += 2*dir;

      double Sup_up = phi[cm];
      double Sdn_up = phi[c];
      double r = ( phi[cm] - phi[cmm] ) / ( phi[c] - phi[cm] );

      double psi_up = std::max( std::min( 2.*r, 1.0), std::min(r, 2.0 ) );
      psi_up = std::max( 0.0, psi_up );

      double Sup_dn = phi[c];
      double Sdn_dn = phi[cm];
      r = ( phi[cp] - phi[c] ) / ( phi[c] - phi[cm] );

      double psi_dn = std::max( std::min( 2.*r, 1.0), std::min(r, 2.0 ) );
      psi_dn = std::max( 0.0, psi_dn );

      double face_up = Sup_up + 0.5 * psi_up * ( Sdn_up - Sup_up );
      double face_dn = Sup_dn + 0.5 * psi_dn * ( Sdn_dn - Sup_dn );

      double face_value_m = ( u[c] > 0.0 ) ? face_up : face_dn;

      Sup_up = phi[c];
      Sdn_up = phi[cp];
      r = ( phi[c] - phi[cm] ) / ( phi[cp] - phi[c] );

      psi_up = std::max( std::min( 2.*r, 1.0), std::min(r, 2.0 ) );
      psi_up = std::max( 0.0, psi_up );

      Sup_dn = phi[cp];
      Sdn_dn = phi[c];
      r = ( phi[cpp] - phi[cp] ) / ( phi[cp] - phi[c] );

      psi_dn = std::max( std::min( 2.*r, 1.0), std::min(r, 2.0 ) );
      psi_dn = std::max( 0.0, psi_dn );

      face_up = Sup_up + 0.5 * psi_up * ( Sdn_up - Sup_up );
      face_dn = Sup_dn + 0.5 * psi_dn * ( Sdn_dn - Sup_dn );

      double face_value_p = ( u[cp] > 0.0 ) ? face_up : face_dn;

      //Done with interpolation, now compute conv and add to RHS:
      rhs[c] += A * ( 0.5 * ( rho[c] + rho[cp] ) * face_value_p * u[cp] -
                0.5 * ( rho[c] + rho[cm] ) * face_value_m * u[c] );

    }
  };
}

void
KScalarRHS::register_initialize(
  std::vector<ArchesFieldContainer::VariableInformation>& variable_registry ){

  //FUNCITON CALL     STRING NAME(VL)     TYPE       DEPENDENCY    GHOST DW     VR
  register_variable(  _rhs_name   , ArchesFieldContainer::COMPUTES , 0 ,
    ArchesFieldContainer::NEWDW , variable_registry );
  register_variable(  _task_name  , ArchesFieldContainer::COMPUTES , 0 ,
    ArchesFieldContainer::NEWDW , variable_registry );
  register_variable(  _D_name     , ArchesFieldContainer::COMPUTES , 0 ,
    ArchesFieldContainer::NEWDW , variable_registry );
  register_variable(  _Fconv_name , ArchesFieldContainer::COMPUTES , 0 ,
    ArchesFieldContainer::NEWDW , variable_registry );
  register_variable(  _Fdiff_name , ArchesFieldContainer::COMPUTES , 0 ,
    ArchesFieldContainer::NEWDW , variable_registry );

}

void
KScalarRHS::initialize( const Patch* patch, ArchesTaskInfoManager* tsk_info,
                       SpatialOps::OperatorDatabase& opr ){

  // SVolFP rhs   = tsk_info->get_so_field<SVolF>(_rhs_name);
  // SVolFP phi   = tsk_info->get_so_field<SVolF>(_task_name);
  // SVolFP gamma = tsk_info->get_so_field<SVolF>(_D_name);
  // SVolFP Fdiff = tsk_info->get_so_field<SVolF>(_Fdiff_name);
  // SVolFP Fconv = tsk_info->get_so_field<SVolF>(_Fconv_name);

  CCVariable<double>& rhs = *(tsk_info->get_uintah_field<CCVariable<double> >(_rhs_name));

  VariableInitializeFunctor my_functor(rhs,0.0);

  // *rhs <<= 0.0;
  // *phi <<= 0.0;
  // *gamma <<= 0.0001;
  // *Fdiff <<= 0.0;
  // *Fconv <<= 0.0;
  //

}
//
//------------------------------------------------
//------------- TIMESTEP INIT --------------------
//------------------------------------------------
//
void
KScalarRHS::register_timestep_init( std::vector<ArchesFieldContainer::VariableInformation>& variable_registry ){
  register_variable( _D_name     , ArchesFieldContainer::COMPUTES , 0 , ArchesFieldContainer::NEWDW , variable_registry  );
  register_variable( _D_name     , ArchesFieldContainer::REQUIRES , 0 , ArchesFieldContainer::OLDDW , variable_registry );
  register_variable( _task_name  , ArchesFieldContainer::COMPUTES , 0 , ArchesFieldContainer::NEWDW , variable_registry  );
  register_variable( _task_name  , ArchesFieldContainer::REQUIRES , 0 , ArchesFieldContainer::OLDDW , variable_registry  );
}

void
KScalarRHS::timestep_init( const Patch* patch, ArchesTaskInfoManager* tsk_info,
                          SpatialOps::OperatorDatabase& opr ){

  // using namespace SpatialOps;
  // using SpatialOps::operator *;
  //
  // SVolFP gamma     = tsk_info->get_so_field<SVolF>( _D_name );
  // SVolFP old_gamma = tsk_info->get_const_so_field<SVolF>( _D_name );
  // SVolFP phi       = tsk_info->get_so_field<SVolF>( _task_name );
  // SVolFP old_phi   = tsk_info->get_const_so_field<SVolF>( _task_name );
  //
  // *gamma <<= *old_gamma;
  // *phi <<= *old_phi;

}


//
//------------------------------------------------
//------------- TIMESTEP WORK --------------------
//------------------------------------------------
//

void
KScalarRHS::register_timestep_eval( std::vector<ArchesFieldContainer::VariableInformation>& variable_registry, const int time_substep ){

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
KScalarRHS::eval( const Patch* patch, ArchesTaskInfoManager* tsk_info,
                 SpatialOps::OperatorDatabase& opr ){


  CCVariable<double>& rhs = *(tsk_info->get_uintah_field<CCVariable<double> >(_rhs_name));
  constCCVariable<double>& phi = *(tsk_info->get_const_uintah_field<constCCVariable<double> >(_task_name));
  constCCVariable<double>& rho = *(tsk_info->get_const_uintah_field<constCCVariable<double> >("density"));
  constSFCXVariable<double>& u = *(tsk_info->get_const_uintah_field<constSFCXVariable<double> >("uVelocitySPBC"));
  constSFCYVariable<double>& v = *(tsk_info->get_const_uintah_field<constSFCYVariable<double> >("vVelocitySPBC"));
  constSFCZVariable<double>& w = *(tsk_info->get_const_uintah_field<constSFCZVariable<double> >("wVelocitySPBC"));

  Vector DX = patch->dCell();
  double A = DX.y() * DX.z();
  
  IntVector dir(1,0,0);
  ComputeConvection<constSFCXVariable<double> > XConv( phi, rhs, rho, u, dir, A);

  dir = IntVector(0,1,0);
  ComputeConvection<constSFCYVariable<double> > YConv( phi, rhs, rho, v, dir, A);

  dir = IntVector(0,0,1);
  ComputeConvection<constSFCZVariable<double> > ZConv( phi, rhs, rho, w, dir, A);

}

//
//------------------------------------------------
//------------- BOUNDARY CONDITIONS --------------
//------------------------------------------------
//

void
KScalarRHS::register_compute_bcs( std::vector<ArchesFieldContainer::VariableInformation>& variable_registry, const int time_substep ){
  //register_variable( _task_name, ArchesFieldContainer::MODIFIES, variable_registry, time_substep );
}

void
KScalarRHS::compute_bcs( const Patch* patch, ArchesTaskInfoManager* tsk_info,
                        SpatialOps::OperatorDatabase& opr ){


}
