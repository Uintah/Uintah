#include <CCA/Components/Arches/PropertyModelsV2/DensityPredictor.h>
#include <CCA/Components/Arches/Operators/Operators.h>

#include <spatialops/Nebo.h>
#include <spatialops/structured/stencil/FVStaggeredOperatorTypes.h>


using namespace Uintah;
using namespace SpatialOps;
using SpatialOps::operator *;
typedef SVolField   SVolF;
typedef SSurfXField SurfX;
typedef SSurfYField SurfY;
typedef SSurfZField SurfZ;
typedef SpatialOps::SpatFldPtr<SVolF> SVolFP;
typedef SpatialOps::SpatFldPtr<SurfX> SurfXP;
typedef SpatialOps::SpatFldPtr<SurfY> SurfYP;
typedef SpatialOps::SpatFldPtr<SurfZ> SurfZP;

DensityPredictor::DensityPredictor( std::string task_name, int matl_index ) :
TaskInterface( task_name, matl_index ) {
  _use_exact_guess = false;
}

DensityPredictor::~DensityPredictor(){
}

void
DensityPredictor::problemSetup( ProblemSpecP& db ){

  if (db->findBlock("use_exact_guess")){
    _use_exact_guess = true;
    ProblemSpecP db_prop = db->getRootNode()->findBlock("CFD")->findBlock("ARCHES")->findBlock("Properties")->findBlock("ColdFlow");
    if ( db_prop == 0 ){
      throw InvalidValue("Error: For the density predictor, you must be using cold flow model when computing the exact rho/rhof relationship.", __FILE__, __LINE__);
    }
    db_prop->findBlock("stream_0")->getAttribute("density",_rho0);
    db_prop->findBlock("stream_1")->getAttribute("density",_rho1);
    _f_name = "NA";
    db_prop->findBlock("mixture_fraction")->getAttribute("label",_f_name);
    if ( _f_name == "NA" ){
      throw InvalidValue("Error: Mixture fraction name not recognized: "+_f_name,__FILE__, __LINE__);
    }
  }

  ProblemSpecP press_db = db->getRootNode()->findBlock("CFD")->findBlock("ARCHES")->findBlock("ExplicitSolver")->findBlock("PressureSolver");
  //__________________________________
  // allow for addition of mass source terms
  if (press_db->findBlock("src")){
    std::string srcname;
    for (ProblemSpecP src_db = press_db->findBlock("src"); src_db != 0; src_db = src_db->findNextBlock("src")){
      src_db->getAttribute("label", srcname);
      _mass_sources.push_back( srcname );
    }
  }
}

void
DensityPredictor::create_local_labels(){
  register_new_variable<CCVariable<double> >( "new_densityGuess" );
}

//
//------------------------------------------------
//-------------- INITIALIZATION ------------------
//------------------------------------------------
//

void
DensityPredictor::register_initialize( std::vector<ArchesFieldContainer::VariableInformation>& variable_registry ){

  register_variable( "new_densityGuess", ArchesFieldContainer::COMPUTES, variable_registry );

}

void
DensityPredictor::initialize( const Patch* patch, ArchesTaskInfoManager* tsk_info,
                        SpatialOps::OperatorDatabase& opr ){


  using namespace SpatialOps;
  using SpatialOps::operator *;

  typedef SpatialOps::SVolField     SVolF;
  typedef SpatialOps::SpatFldPtr<SVolF> SVolFP;

  SVolFP rho = tsk_info->get_so_field<SVolF>("new_densityGuess");

  *rho <<= 0.0;

}

//
//------------------------------------------------
//------TIMESTEP INITIALIZATION ------------------
//------------------------------------------------
//

void
DensityPredictor::register_timestep_init( std::vector<ArchesFieldContainer::VariableInformation>& variable_registry ){

  register_variable( "new_densityGuess", ArchesFieldContainer::COMPUTES, variable_registry );

}

void
DensityPredictor::timestep_init( const Patch* patch, ArchesTaskInfoManager* tsk_info,
                        SpatialOps::OperatorDatabase& opr ){


  using namespace SpatialOps;
  using SpatialOps::operator *;

  typedef SpatialOps::SVolField     SVolF;
  typedef SpatialOps::SpatFldPtr<SVolF> SVolFP;

  SVolFP rho = tsk_info->get_so_field<SVolF>("new_densityGuess");

  *rho <<= 0.0;

}

//
//------------------------------------------------
//------------- TIMESTEP WORK --------------------
//------------------------------------------------
//

void
DensityPredictor::register_timestep_eval( std::vector<ArchesFieldContainer::VariableInformation>& variable_registry, const int time_substep ){

  register_variable( "new_densityGuess"  , ArchesFieldContainer::MODIFIES,  variable_registry, time_substep );
  register_variable( "densityGuess"  , ArchesFieldContainer::MODIFIES,  variable_registry, time_substep );
  register_variable( "densityCP"     , ArchesFieldContainer::REQUIRES , 1 , ArchesFieldContainer::NEWDW  , variable_registry , time_substep );
  register_variable( "volFraction"   , ArchesFieldContainer::REQUIRES , 1 , ArchesFieldContainer::LATEST , variable_registry , time_substep );
  register_variable( "uVelocitySPBC" , ArchesFieldContainer::REQUIRES , 1 , ArchesFieldContainer::LATEST , variable_registry , time_substep );
  register_variable( "vVelocitySPBC" , ArchesFieldContainer::REQUIRES , 1 , ArchesFieldContainer::LATEST , variable_registry , time_substep );
  register_variable( "wVelocitySPBC" , ArchesFieldContainer::REQUIRES , 1 , ArchesFieldContainer::LATEST , variable_registry , time_substep );
  register_variable( "sm_cont" , ArchesFieldContainer::REQUIRES , 0 , ArchesFieldContainer::NEWDW  , variable_registry , time_substep );
  if ( !_use_exact_guess ){
    //typedef std::vector<std::string> SVec;
    //for (SVec::iterator i = _mass_sources.begin(); i != _mass_sources.end(); i++ ){
      //register_variable( *i , ArchesFieldContainer::REQUIRES , 0 , ArchesFieldContainer::NEWDW  , variable_registry , time_substep );
    //}
  }
  if ( _use_exact_guess )
    register_variable( _f_name     , ArchesFieldContainer::REQUIRES , 0 , ArchesFieldContainer::NEWDW  , variable_registry , time_substep );

}

void
DensityPredictor::eval( const Patch* patch, ArchesTaskInfoManager* tsk_info,
                  SpatialOps::OperatorDatabase& opr ){

  using namespace SpatialOps;
  using SpatialOps::operator *;

  typedef SpatialOps::SVolField         SVolF;
  typedef SpatialOps::SSurfXField       SSurfX;
  typedef SpatialOps::SSurfYField       SSurfY;
  typedef SpatialOps::SSurfZField       SSurfZ;
  typedef SpatialOps::XVolField         XVolF;
  typedef SpatialOps::YVolField         YVolF;
  typedef SpatialOps::ZVolField         ZVolF;
  typedef SpatialOps::SpatFldPtr<XVolF> XVolFP;
  typedef SpatialOps::SpatFldPtr<YVolF> YVolFP;
  typedef SpatialOps::SpatFldPtr<ZVolF> ZVolFP;
  typedef SpatialOps::SpatFldPtr<SVolF> SVolFP;

  typedef SpatialOps::OperatorTypeBuilder< SpatialOps::Interpolant, SVolF, SSurfX >::type SVolToSX;
  typedef SpatialOps::OperatorTypeBuilder< SpatialOps::Interpolant, SVolF, SSurfY >::type SVolToSY;
  typedef SpatialOps::OperatorTypeBuilder< SpatialOps::Interpolant, SVolF, SSurfZ >::type SVolToSZ;

  typedef SpatialOps::OperatorTypeBuilder< SpatialOps::Interpolant, SpatialOps::XVolField, SSurfX>::type XVolToSX;
  typedef SpatialOps::OperatorTypeBuilder< SpatialOps::Interpolant, SpatialOps::YVolField, SSurfY>::type YVolToSY;
  typedef SpatialOps::OperatorTypeBuilder< SpatialOps::Interpolant, SpatialOps::ZVolField, SSurfZ>::type ZVolToSZ;

  typedef SpatialOps::OperatorTypeBuilder< SpatialOps::Divergence, SSurfX, SVolF>::type DivX;
  typedef SpatialOps::OperatorTypeBuilder< SpatialOps::Divergence, SSurfY, SVolF>::type DivY;
  typedef SpatialOps::OperatorTypeBuilder< SpatialOps::Divergence, SSurfZ, SVolF>::type DivZ;

  SVolFP rho_guess = tsk_info->get_so_field<SVolF>( "new_densityGuess");
  SVolFP rho_guess_a = tsk_info->get_so_field<SVolF>( "densityGuess");
  SVolFP const rho = tsk_info->get_const_so_field<SVolF>( "densityCP" );
  SVolFP const vf = tsk_info->get_const_so_field<SVolF>( "volFraction" );

  XVolFP const u = tsk_info->get_const_so_field<XVolF>( "uVelocitySPBC" );
  YVolFP const v = tsk_info->get_const_so_field<YVolF>( "vVelocitySPBC" );
  ZVolFP const w = tsk_info->get_const_so_field<ZVolF>( "wVelocitySPBC" );

  //operators
  const SVolToSX* const interpx = opr.retrieve_operator<SVolToSX>();
  const SVolToSY* const interpy = opr.retrieve_operator<SVolToSY>();
  const SVolToSZ* const interpz = opr.retrieve_operator<SVolToSZ>();
  const XVolToSX* const uinterpx = opr.retrieve_operator<XVolToSX>();
  const YVolToSY* const vinterpy = opr.retrieve_operator<YVolToSY>();
  const ZVolToSZ* const winterpz = opr.retrieve_operator<ZVolToSZ>();
  const DivX* const divx = opr.retrieve_operator<DivX>();
  const DivY* const divy = opr.retrieve_operator<DivY>();
  const DivZ* const divz = opr.retrieve_operator<DivZ>();

  //---work---
  double dt = tsk_info->get_dt();

  if ( _use_exact_guess ){

    SVolFP const f = tsk_info->get_const_so_field<SVolF>( _f_name );

    *rho_guess <<= ( _rho1 - *rho * *f *( _rho1 / _rho0 - 1.) ) * *vf;

  } else {

    *rho_guess <<= ( *rho - dt * ((*divx)( (*interpx)(*rho) * (*uinterpx)(*u) ) +
                                (*divy)( (*interpy)(*rho) * (*vinterpy)(*v) ) +
                                (*divz)( (*interpz)(*rho) * (*winterpz)(*w) ) ) )* *vf;

    //adding extra mass sources
    typedef std::vector<std::string> SVec;
    for (SVec::iterator i = _mass_sources.begin(); i != _mass_sources.end(); i++ ){

      SVolFP const src = tsk_info->get_const_so_field<SVolF>( *i );
      *rho_guess <<= *rho_guess + dt * *src;

    }

  }

  //this kludge is needed until the old version goes away...
  *rho_guess_a <<= *rho_guess;

}
