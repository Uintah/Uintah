#include <CCA/Components/Arches/CoalModels/DragModel.h>
#include <CCA/Components/Arches/TransportEqns/EqnFactory.h>
#include <CCA/Components/Arches/TransportEqns/EqnBase.h>
#include <CCA/Components/Arches/TransportEqns/DQMOMEqn.h>
#include <CCA/Components/Arches/TransportEqns/DQMOMEqnFactory.h>
#include <CCA/Components/Arches/CoalModels/PartVel.h>
#include <CCA/Components/Arches/ParticleModels/ParticleHelper.h>
#include <CCA/Components/Arches/ArchesLabel.h>
#include <Core/Exceptions/ProblemSetupException.h>

#include <Core/ProblemSpec/ProblemSpec.h>
#include <CCA/Ports/Scheduler.h>
#include <Core/Grid/SimulationState.h>
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
                                    SimulationStateP          & sharedState,
                                    int qn ) :
  ModelBuilder( modelName, reqICLabelNames, reqScalarLabelNames, fieldLabels, sharedState, qn )
{}

DragModelBuilder::~DragModelBuilder(){}

ModelBase* DragModelBuilder::build(){
  return new DragModel( d_modelName, d_sharedState, d_fieldLabels, d_icLabels, d_scalarLabels, d_quadNode );
}

// End Builder
//---------------------------------------------------------------------------

DragModel::DragModel( std::string modelName, 
                      SimulationStateP& sharedState,
                      ArchesLabel* fieldLabels,
                      vector<std::string> icLabelNames, 
                      vector<std::string> scalarLabelNames,
                      int qn ) 
: ModelBase(modelName, sharedState, fieldLabels, icLabelNames, scalarLabelNames, qn)
{
  // Create a label for this model
  d_modelLabel = VarLabel::create( modelName, CCVariable<double>::getTypeDescription() );

  // Create the gas phase source term associated with this model
  std::string gasSourceName = modelName + "_gasSource";
  d_gasLabel = VarLabel::create( gasSourceName, CCVariable<double>::getTypeDescription() );

  //constants
  _pi = acos(-1.0);

  //initialize
  _birth_label = NULL; 

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
  std::string length_root = ParticleHelper::parse_for_role_to_label(db, "size"); 
  std::string length_name = ParticleHelper::append_env( length_root, d_quadNode ); 
  _length_varlabel = VarLabel::find(length_name); 

  // Need a density
  std::string density_root = ParticleHelper::parse_for_role_to_label(db, "density"); 
  _density_name = ParticleHelper::append_env( density_root, d_quadNode ); 

  // Need velocity scaling constant
  std::string vel_root; 
  if ( _dir == 0 ){ 
    vel_root = ParticleHelper::parse_for_role_to_label(db, "uvel"); 
  } else if ( _dir == 1){ 
    vel_root = ParticleHelper::parse_for_role_to_label(db, "vvel"); 
  } else { 
    vel_root = ParticleHelper::parse_for_role_to_label(db, "wvel"); 
  }
  
  vel_root = ParticleHelper::append_qn_env( vel_root, d_quadNode ); 
  EqnBase& temp_current_eqn = dqmom_eqn_factory.retrieve_scalar_eqn(vel_root);
  DQMOMEqn& current_eqn = dynamic_cast<DQMOMEqn&>(temp_current_eqn);
  _vel_scaling_constant = current_eqn.getScalingConstant(d_quadNode);
  std::string ic_RHS = vel_root+"_RHS";
  _RHS_source_varlabel = VarLabel::find(ic_RHS);

  //get the birth term if any: 
  const std::string birth_name = current_eqn.get_model_by_type( "SimpleBirth" ); 
  std::string birth_qn_name = ParticleHelper::append_qn_env(birth_name, d_quadNode); 
  if ( birth_name != "NULLSTRING" ){ 
    _birth_label = VarLabel::find( birth_qn_name ); 
  }

  // Need weight name and scaling constant
  std::string weight_name = ParticleHelper::append_env("w", d_quadNode); 
  _weight_varlabel = VarLabel::find(weight_name); 
  std::string scaled_weight_name = ParticleHelper::append_qn_env("w", d_quadNode); 
  _scaled_weight_varlabel = VarLabel::find(scaled_weight_name); 
  std::string weightqn_name = ParticleHelper::append_qn_env("w", d_quadNode); 
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
  Task* tsk = new Task(taskname, this, &DragModel::initVars);

  tsk->computes(d_modelLabel);
  tsk->computes(d_gasLabel);

  sched->addTask(tsk, level->eachPatch(), d_sharedState->allArchesMaterials());
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
    int matlIndex = d_sharedState->getArchesMaterial(archIndex)->getDWIndex(); 

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
  Task* tsk = new Task( taskname, this, &DragModel::computeModel, timeSubStep );

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

  tsk->requires( Task::NewDW, _rhop_varlabel, gn, 0 ); 
  tsk->requires( which_dw, _length_varlabel, gn, 0 );
  tsk->requires( which_dw, _weight_varlabel, gn, 0 ); 
  tsk->requires( which_dw, _scaled_weight_varlabel, gn, 0 ); 
  tsk->requires( which_dw, d_fieldLabels->d_CCVelocityLabel, gn, 0 );
  tsk->requires( which_dw, d_fieldLabels->d_densityCPLabel, gn, 0 );
  tsk->requires( Task::NewDW, _RHS_source_varlabel, gn, 0 ); 
  if ( _birth_label != NULL )
    tsk->requires( Task::NewDW, _birth_label, gn, 0 ); 

  // require particle velocity
  ArchesLabel::PartVelMap::const_iterator i = d_fieldLabels->partVel.find(d_quadNode);
  tsk->requires( Task::NewDW, i->second, gn, 0 );

  // get time step size for model clipping
  tsk->requires( Task::OldDW,d_fieldLabels->d_sharedState->get_delt_label(), Ghost::None, 0);  

  sched->addTask(tsk, level->eachPatch(), d_sharedState->allArchesMaterials()); 

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
    int matlIndex = d_fieldLabels->d_sharedState->getArchesMaterial(archIndex)->getDWIndex(); 

    Vector Dx = patch->dCell(); 
    double vol = Dx.x()* Dx.y()* Dx.z(); 
    
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
    }

    constCCVariable<Vector> gasVel;
    which_dw->get( gasVel, d_fieldLabels->d_CCVelocityLabel, matlIndex, patch, gn, 0 );
    constCCVariable<double> den;
    which_dw->get(den, d_fieldLabels->d_densityCPLabel, matlIndex, patch, gn, 0 );
    constCCVariable<double> rho_p; 
    new_dw->get( rho_p  , _rhop_varlabel   , matlIndex , patch , gn , 0 );
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
    constCCVariable<double> birth;
    bool add_birth = false; 
    if ( _birth_label != NULL ){
      new_dw->get( birth, _birth_label, matlIndex, patch, gn, 0 ); 
      add_birth = true; 
    } 

    delt_vartype DT;    
    old_dw->get(DT, d_fieldLabels->d_sharedState->get_delt_label());
    double dt = DT;  

    for (CellIterator iter=patch->getCellIterator(); !iter.done(); iter++){

      IntVector c = *iter;

      if (scaled_weight[c] > _weight_small) {
 
        Vector gas_vel = gasVel[c];
        Vector part_vel = partVel[c];
        double denph=den[c];
        double rho_pph=rho_p[c]; 
        double l_pph=l_p[c]; 
        double weightph=weight[c];
        double RHS_sourceph=RHS_source[c];

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

 
        double gasMag = gas_vel.x() * gas_vel.x() + 
                        gas_vel.y() * gas_vel.y() + 
                        gas_vel.z() * gas_vel.z();

        gasMag = pow(gasMag,0.5);

        double partMag = part_vel.x() * part_vel.x() + 
                       part_vel.y() * part_vel.y() + 
                       part_vel.z() * part_vel.z(); 

        partMag = pow(partMag,0.5); 

        double diff = std::abs(partMag - gasMag); 
        double Re  = diff * l_pph / ( _kvisc / denph );  
        double f;

        if(Re < 994.0) {
          f = 1.0 + 0.15*pow(Re, 0.687);
        } else {
          f = 0.0183*Re;
        }

        double t_p = ( rho_pph * l_pph * l_pph )/( 18.0 * _kvisc );
        double tau=t_p/f;
        // add rate clipping if drag time scale is smaller than dt..
        if (tau > dt ){
          model[c] = scaled_weight[c] * ( f / t_p * (gas_vel[_dir]-part_vel[_dir])+_gravity[_dir]) / (_vel_scaling_constant);
          gas_source[c] = -weightph * rho_pph / 6.0 * _pi * f / t_p * ( gas_vel[_dir]-part_vel[_dir] ) * pow(l_pph,3.0);
        } else {
          //model[c] = (weightph/(_vel_scaling_constant*_weight_scaling_constant)) * (gas_vel[_dir]-part_vel[_dir]) / dt -  RHS_sourceph/vol; 
          if ( add_birth ){ 
            model[c] = scaled_weight[c] / _vel_scaling_constant * ( gas_vel[_dir] - part_vel[_dir] ) / dt - ( RHS_sourceph / vol + birth[c] );
          } else { 
            model[c] = scaled_weight[c] / _vel_scaling_constant * ( gas_vel[_dir] - part_vel[_dir] ) / dt - ( RHS_sourceph / vol );
          }
          gas_source[c] = 0.0;
        }

      } else {
 
          model[c] = 0.0;
          gas_source[c] = 0.0;

      }
    }
  }
}
