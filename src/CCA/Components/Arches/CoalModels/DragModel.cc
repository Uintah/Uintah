#include <CCA/Components/Arches/CoalModels/DragModel.h>
#include <CCA/Components/Arches/TransportEqns/EqnFactory.h>
#include <CCA/Components/Arches/TransportEqns/EqnBase.h>
#include <CCA/Components/Arches/TransportEqns/DQMOMEqn.h>
#include <CCA/Components/Arches/TransportEqns/DQMOMEqnFactory.h>
#include <CCA/Components/Arches/CoalModels/PartVel.h>
#include <CCA/Components/Arches/ParticleModels/ParticleTools.h>
#include <CCA/Components/Arches/ArchesLabel.h>
#include <Core/Exceptions/ProblemSetupException.h>

#include <Core/ProblemSpec/ProblemSpec.h>
#include <CCA/Ports/Scheduler.h>
#include <Core/Grid/MaterialManager.h>
#include <Core/Grid/Variables/VarTypes.h>
#include <Core/Grid/Variables/CCVariable.h>
#include <Core/Exceptions/InvalidValue.h>

//===========================================================================

using namespace std;
using namespace Uintah;

//---------------------------------------------------------------------------
// Builder:

DragModelBuilder::DragModelBuilder( const std::string         & modelName,
                                    const vector<std::string> & reqICLabelNames,
                                    const vector<std::string> & reqScalarLabelNames,
                                    ArchesLabel         * fieldLabels,
                                    MaterialManagerP          & materialManager,
                                    int qn ) :
  ModelBuilder( modelName, reqICLabelNames, reqScalarLabelNames, fieldLabels, materialManager, qn )
{}

DragModelBuilder::~DragModelBuilder(){}

ModelBase* DragModelBuilder::build(){
  return scinew DragModel( d_modelName, d_materialManager, d_fieldLabels, d_icLabels, d_scalarLabels, d_quadNode );
}

// End Builder
//---------------------------------------------------------------------------

DragModel::DragModel( std::string modelName,
                      MaterialManagerP& materialManager,
                      ArchesLabel* fieldLabels,
                      vector<std::string> icLabelNames,
                      vector<std::string> scalarLabelNames,
                      int qn )
: ModelBase(modelName, materialManager, fieldLabels, icLabelNames, scalarLabelNames, qn)
{
  // Create a label for this model
  d_modelLabel = VarLabel::create( modelName, CCVariable<double>::getTypeDescription() );

  // Create the gas phase source term associated with this model
  std::string gasSourceName = modelName + "_gasSource";
  d_gasLabel = VarLabel::create( gasSourceName, CCVariable<double>::getTypeDescription() );

  //initialize
  _birth_label = nullptr;

}

DragModel::~DragModel()
{}

//---------------------------------------------------------------------------
// Method: Problem Setup
//---------------------------------------------------------------------------
void
DragModel::problemSetup(const ProblemSpecP& params, int qn)
{

  ProblemSpecP db = params;

  DQMOMEqnFactory& dqmom_eqn_factory = DQMOMEqnFactory::self();

  // check for gravity
  const ProblemSpecP params_root = db->getRootNode();
  if (params_root->findBlock("PhysicalConstants")) {
    ProblemSpecP db_phys = params_root->findBlock("PhysicalConstants");
    db_phys->require("gravity", _gravity);
    db_phys->require("viscosity", _kvisc);
  } else {
    throw InvalidValue("Error: Missing <PhysicalConstants> section in input file required for drag model.",__FILE__,__LINE__);
  }

  std::string coord;
  db->require("direction",coord);

  if ( coord == "x" || coord == "X" ){
    _dir = 0;
  } else if ( coord == "y" || coord == "Y" ){
    _dir = 1;
  } else {
    _dir = 2;
  }

  // Need a size IC:
  std::string length_root = ArchesCore::parse_for_particle_role_to_label(db, ArchesCore::P_SIZE);
  std::string length_name = ArchesCore::append_env( length_root, d_quadNode );
  _length_varlabel = VarLabel::find(length_name);

  // Need a density
  std::string density_root = ArchesCore::parse_for_particle_role_to_label(db, ArchesCore::P_DENSITY);
  _density_name = ArchesCore::append_env( density_root, d_quadNode );

  // Need velocity scaling constant
  std::string vel_root;
  if ( _dir == 0 ){
    vel_root = ArchesCore::parse_for_particle_role_to_label(db, ArchesCore::P_XVEL);
  } else if ( _dir == 1){
    vel_root = ArchesCore::parse_for_particle_role_to_label(db, ArchesCore::P_YVEL);
  } else {
    vel_root = ArchesCore::parse_for_particle_role_to_label(db, ArchesCore::P_ZVEL);
  }

  
  //EqnBase& temp_current_eqn = dqmom_eqn_factory.retrieve_scalar_eqn(vel_root);
  //DQMOMEqn& current_eqn = dynamic_cast<DQMOMEqn&>(temp_current_eqn);
  
  _vel_scaling_constant = ArchesCore::get_scaling_constant(db,vel_root, d_quadNode);//current_eqn.getScalingConstant(d_quadNode);
  const std::string birth_name = ArchesCore::getModelNameByType( db, vel_root, "BirthDeath");

  vel_root = ArchesCore::append_qn_env( vel_root, d_quadNode );
  
  std::string ic_RHS = vel_root+"_RHS";
  //std::string ic_RHS = vel_root+"_rhs";
  _RHS_source_varlabel = VarLabel::find(ic_RHS);

  std::string weight_name = ArchesCore::append_qn_env("w", qn);
  std::string weight_RHS_name = weight_name + "_RHS";
  _RHS_weight_varlabel = VarLabel::find(weight_RHS_name);

  //get the birth term if any:
//  const std::string birth_name = current_eqn.get_model_by_type( "BirthDeath" );

  std::string birth_qn_name = ArchesCore::append_qn_env(birth_name, d_quadNode);
  if ( birth_name != "NULLSTRING" ){
    _birth_label = VarLabel::find( birth_qn_name );
  }

  // Need weight name and scaling constant
  std::string unscaled_weight_name = ArchesCore::append_env("w", d_quadNode);
  _weight_varlabel = VarLabel::find(unscaled_weight_name);
  std::string scaled_weight_name = ArchesCore::append_qn_env("w", d_quadNode);
  _scaled_weight_varlabel = VarLabel::find(scaled_weight_name);
  std::string weightqn_name = ArchesCore::append_qn_env("w", d_quadNode);
  EqnBase& temp_current_eqn2 = dqmom_eqn_factory.retrieve_scalar_eqn(weightqn_name);
  DQMOMEqn& current_eqn2 = dynamic_cast<DQMOMEqn&>(temp_current_eqn2);
  _weight_small = current_eqn2.getSmallClipPlusTol();
  _weight_scaling_constant = current_eqn2.getScalingConstant(d_quadNode);

}

//---------------------------------------------------------------------------
// Method: Schedule the initialization of special variables unique to model
//---------------------------------------------------------------------------
void
DragModel::sched_initVars( const LevelP& level, SchedulerP& sched )
{
  string taskname = "DragModel::initVars";
  Task* tsk = scinew Task(taskname, this, &DragModel::initVars);

  tsk->computes(d_modelLabel);
  tsk->computes(d_gasLabel);

  sched->addTask(tsk, level->eachPatch(), d_materialManager->allMaterials( "Arches" ));
}

//-------------------------------------------------------------------------
// Method: Initialize special variables unique to the model
//-------------------------------------------------------------------------
void
DragModel::initVars( const ProcessorGroup * pc,
                            const PatchSubset    * patches,
                            const MaterialSubset * matls,
                            DataWarehouse        * old_dw,
                            DataWarehouse        * new_dw )
{
  //patch loop
  for (int p=0; p < patches->size(); p++){
    const Patch* patch = patches->get(p);
    int archIndex = 0;
    int matlIndex = d_materialManager->getMaterial( "Arches", archIndex)->getDWIndex();

    CCVariable<double> model;
    CCVariable<double> gas_source;

    new_dw->allocateAndPut( model, d_modelLabel, matlIndex, patch );
    model.initialize(0.0);
    new_dw->allocateAndPut( gas_source, d_gasLabel, matlIndex, patch );
    gas_source.initialize(0.0);


  }
}



//---------------------------------------------------------------------------
// Method: Schedule the calculation of the model
//---------------------------------------------------------------------------
void
DragModel::sched_computeModel( const LevelP& level, SchedulerP& sched, int timeSubStep )
{
  std::string taskname = "DragModel::computeModel";
  Task* tsk = scinew Task( taskname, this, &DragModel::computeModel, timeSubStep );

  Ghost::GhostType  gn  = Ghost::None;

  Task::WhichDW which_dw;

  _rhop_varlabel = VarLabel::find(_density_name);

  if ( _rhop_varlabel == 0 ){
    throw InvalidValue("Error: Rho label not found for particle drag model.",__FILE__,__LINE__);
  }

  if (timeSubStep == 0 ) {
    tsk->computes(d_modelLabel);
    tsk->computes(d_gasLabel);
    which_dw = Task::OldDW;
  } else {
    tsk->modifies(d_modelLabel);
    tsk->modifies(d_gasLabel);
    which_dw = Task::NewDW;
  }

  if ( _dir == 0 ){
    std::string name = ArchesCore::append_qn_env("ux", d_quadNode );
    const VarLabel* label = VarLabel::find(name);
    tsk->requires( which_dw, label, gn, 0 );
  } else if ( _dir == 1 ){
    std::string name = ArchesCore::append_qn_env("uy", d_quadNode );
    const VarLabel* label = VarLabel::find(name);
    tsk->requires( which_dw, label, gn, 0 );
  } else {
    std::string name = ArchesCore::append_qn_env("uz", d_quadNode );
    const VarLabel* label = VarLabel::find(name);
    tsk->requires( which_dw, label, gn, 0 );
  }
  tsk->requires( which_dw, _rhop_varlabel, gn, 0 );
  tsk->requires( which_dw, _length_varlabel, gn, 0 );
  tsk->requires( which_dw, _weight_varlabel, gn, 0 );
  tsk->requires( which_dw, _scaled_weight_varlabel, gn, 0 );
  tsk->requires( which_dw, d_fieldLabels->d_CCVelocityLabel, gn, 0 );
  tsk->requires( which_dw, d_fieldLabels->d_densityCPLabel, gn, 0 );
  tsk->requires( Task::NewDW, _RHS_source_varlabel, gn, 0 );
  tsk->requires( Task::NewDW, _RHS_weight_varlabel, gn, 0 );
  if ( _birth_label != nullptr )
    tsk->requires( Task::NewDW, _birth_label, gn, 0 );

  // require particle velocity
  ArchesLabel::PartVelMap::const_iterator i = d_fieldLabels->partVel.find(d_quadNode);
  tsk->requires( Task::NewDW, i->second, gn, 0 );

  // get time step size for model clipping
  tsk->requires( Task::OldDW,d_fieldLabels->d_delTLabel, Ghost::None, 0);

  sched->addTask(tsk, level->eachPatch(), d_materialManager->allMaterials( "Arches" ));

}
//---------------------------------------------------------------------------
// Method: Actually compute the source term
//---------------------------------------------------------------------------
void
DragModel::computeModel( const ProcessorGroup* pc,
                   const PatchSubset* patches,
                   const MaterialSubset* matls,
                   DataWarehouse* old_dw,
                   DataWarehouse* new_dw,
                   const int timeSubStep )
{
  //patch loop
  for (int p=0; p < patches->size(); p++){

    Ghost::GhostType  gn  = Ghost::None;

    const Patch* patch = patches->get(p);
    int archIndex = 0;
    int matlIndex = d_fieldLabels->d_materialManager->getMaterial( "Arches", archIndex)->getDWIndex();

    Vector Dx = patch->dCell();
    const double vol = Dx.x()* Dx.y()* Dx.z();

    CCVariable<double> model;
    CCVariable<double> gas_source;
    DataWarehouse* which_dw;

    if ( timeSubStep == 0 ){
      which_dw = old_dw;
      new_dw->allocateAndPut( model, d_modelLabel, matlIndex, patch );
      model.initialize(0.0);
      new_dw->allocateAndPut( gas_source, d_gasLabel, matlIndex, patch );
      gas_source.initialize(0.0);
    } else {
      which_dw = new_dw;
      new_dw->getModifiable( model, d_modelLabel, matlIndex, patch );
      new_dw->getModifiable( gas_source, d_gasLabel, matlIndex, patch );
      model.initialize(0.0);
      gas_source.initialize(0.0);
    }

    constCCVariable<double> weight_p_vel;
    if ( _dir == 0 ){
      std::string name = ArchesCore::append_qn_env("ux", d_quadNode );
      const VarLabel* label = VarLabel::find(name);
      which_dw->get( weight_p_vel, label, matlIndex, patch, gn, 0 );
    } else if ( _dir == 1 ){
      std::string name = ArchesCore::append_qn_env("uy", d_quadNode );
      const VarLabel* label = VarLabel::find(name);
      which_dw->get( weight_p_vel, label, matlIndex, patch, gn, 0 );
    } else {
      std::string name = ArchesCore::append_qn_env("uz", d_quadNode );
      const VarLabel* label = VarLabel::find(name);
      which_dw->get( weight_p_vel, label, matlIndex, patch, gn, 0 );
    }
    constCCVariable<Vector> gasVel;
    which_dw->get( gasVel, d_fieldLabels->d_CCVelocityLabel, matlIndex, patch, gn, 0 );
    constCCVariable<double> den;
    which_dw->get(den, d_fieldLabels->d_densityCPLabel, matlIndex, patch, gn, 0 );
    constCCVariable<double> rho_p;
    which_dw->get( rho_p  , _rhop_varlabel   , matlIndex , patch , gn , 0 );
    constCCVariable<double> l_p;
    which_dw->get( l_p    , _length_varlabel , matlIndex , patch , gn , 0 );
    constCCVariable<double> weight;
    which_dw->get( weight , _weight_varlabel , matlIndex , patch , gn , 0 );
    constCCVariable<double> scaled_weight;
    which_dw->get( scaled_weight, _scaled_weight_varlabel, matlIndex, patch, gn, 0 );
    constCCVariable<Vector> partVel;
    ArchesLabel::PartVelMap::const_iterator iter = d_fieldLabels->partVel.find(d_quadNode);
    new_dw->get(partVel, iter->second, matlIndex, patch, gn, 0);
    constCCVariable<double> RHS_source;
    new_dw->get( RHS_source , _RHS_source_varlabel , matlIndex , patch , gn , 0 );
    constCCVariable<double> RHS_weight;
    new_dw->get( RHS_weight , _RHS_weight_varlabel , matlIndex , patch , gn , 0 );
    constCCVariable<double> birth;
    bool add_birth = false;
    if ( _birth_label != nullptr ){
      new_dw->get( birth, _birth_label, matlIndex, patch, gn, 0 );
      add_birth = true;
    }

    delt_vartype DT;
    old_dw->get(DT, d_fieldLabels->d_delTLabel);
    const double dt = DT;

    double c_gravity=_gravity[_dir];

    std::function<double  ( int i,  int j, int k)> lambdaBirth;

    if ( add_birth ){
      lambdaBirth = [&]( int i, int j, int k)-> double   { return  birth(i,j,k);};
    }else{
      lambdaBirth = [&]( int i, int j, int k)-> double   { return 0.0;};
    }

    Uintah::BlockRange range(patch->getCellLowIndex(),patch->getCellHighIndex());
    Uintah::parallel_for(range,  [&]( int i,  int j, int k){

        if (scaled_weight(i,j,k) > _weight_small) {

        Vector gas_vel = gasVel(i,j,k);
        Vector part_vel = partVel(i,j,k);
        double denph=den(i,j,k);
        double rho_pph=rho_p(i,j,k);
        double l_pph=l_p(i,j,k);
        double weightph=weight(i,j,k);

        // Verification
        //denph=0.394622;
        //rho_pph=1300;
        //l_pph=2e-05;
        //weightph=1.40781e+09;
        //gas_vel[0]=7.56321;
        //gas_vel[1]=0.663992;
        //gas_vel[2]=0.654003;
        //part_vel[0]=6.54863;
        //part_vel[1]=0.339306;
        //part_vel[2]=0.334942;

        double diff = (gas_vel.x() - part_vel.x()) * (gas_vel.x() - part_vel.x())
                      + (gas_vel.y() - part_vel.y()) * (gas_vel.y() - part_vel.y())
                      + (gas_vel.z() - part_vel.z()) * (gas_vel.z() - part_vel.z());
        diff = pow(diff,0.5);
        double Re  = diff * l_pph / ( _kvisc / denph );

        double fDrag = Re <994 ? 1.0 + 0.15*pow(Re, 0.687) : 0.0183*Re;

        double t_p = ( rho_pph * l_pph * l_pph )/( 18.0 * _kvisc );
        double tau=t_p/fDrag;

        if (tau > dt ){ 
          model(i,j,k) = scaled_weight(i,j,k) * ( fDrag / t_p * (gas_vel[_dir]-part_vel[_dir])+c_gravity) / (_vel_scaling_constant);
          gas_source(i,j,k) =-weightph * rho_pph / 6.0 * M_PI * fDrag / t_p * ( gas_vel[_dir]-part_vel[_dir] ) * pow(l_pph,3.0);
        }else{  // rate clip, if we aren't resolving timescale
          double updated_weight = std::max(scaled_weight(i,j,k) + dt / vol * ( RHS_weight(i,j,k) ) , 1e-15);
          model(i,j,k) = 1. / _vel_scaling_constant * ( updated_weight * gas_vel[_dir] - weight_p_vel(i,j,k) ) / dt - ( RHS_source(i,j,k) / vol + lambdaBirth(i,j,k));
        } // end timescale if
      }  // end low-weight if
    }); // end lambda
  }
}
