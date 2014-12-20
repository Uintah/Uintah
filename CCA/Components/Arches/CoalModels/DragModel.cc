#include <CCA/Components/Arches/CoalModels/DragModel.h>
#include <CCA/Components/Arches/TransportEqns/EqnFactory.h>
#include <CCA/Components/Arches/TransportEqns/EqnBase.h>
#include <CCA/Components/Arches/TransportEqns/DQMOMEqn.h>
#include <CCA/Components/Arches/TransportEqns/DQMOMEqnFactory.h>
#include <CCA/Components/Arches/CoalModels/PartVel.h>
#include <CCA/Components/Arches/PropertyModelsV2/PropertyHelper.h>
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
  return scinew DragModel( d_modelName, d_sharedState, d_fieldLabels, d_icLabels, d_scalarLabels, d_quadNode );
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
  std::string length_root = PropertyHelper::parse_for_role_to_label(db, "size"); 
  _length_name = PropertyHelper::append_env( length_root, d_quadNode ); 

  // Need a density
  std::string density_root = PropertyHelper::parse_for_role_to_label(db, "density"); 
  _density_name = PropertyHelper::append_env( density_root, d_quadNode ); 

  // Need velocity scaling constant
  std::string vel_root; 
  if ( _dir == 0 ){ 
    vel_root = PropertyHelper::parse_for_role_to_label(db, "uvel"); 
  } else if ( _dir == 1){ 
    vel_root = PropertyHelper::parse_for_role_to_label(db, "vvel"); 
  } else { 
    vel_root = PropertyHelper::parse_for_role_to_label(db, "wvel"); 
  }
  
  vel_root = PropertyHelper::append_qn_env( vel_root, d_quadNode ); 
  EqnBase& temp_current_eqn = dqmom_eqn_factory.retrieve_scalar_eqn(vel_root);
  DQMOMEqn& current_eqn = dynamic_cast<DQMOMEqn&>(temp_current_eqn);
  _vel_scaling_const = current_eqn.getScalingConstant();

  // Need weight name and scaling constant
  _weight_name = PropertyHelper::append_qn_env("w", d_quadNode); 
  EqnBase& temp_current_eqn2 = dqmom_eqn_factory.retrieve_scalar_eqn(_weight_name);
  DQMOMEqn& current_eqn2 = dynamic_cast<DQMOMEqn&>(temp_current_eqn2);
  _w_scaling_const = current_eqn2.getScalingConstant();

}

//---------------------------------------------------------------------------
// Method: Schedule the initialization of special variables unique to model
//---------------------------------------------------------------------------
void 
DragModel::sched_initVars( const LevelP& level, SchedulerP& sched )
{
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

  if (timeSubStep == 0 ) { 
    tsk->computes(d_modelLabel);
    tsk->computes(d_gasLabel);
    which_dw = Task::OldDW; 
  } else {
    tsk->modifies(d_modelLabel); 
    tsk->modifies(d_gasLabel); 
    which_dw = Task::NewDW; 
  }

  //density: 
  const VarLabel* rhop_label = VarLabel::find(_density_name); 
  const VarLabel* length_label = VarLabel::find(_length_name); 
  const VarLabel* weight_label = VarLabel::find(_weight_name); 
  tsk->requires( which_dw, rhop_label, gn, 0 ); 
  tsk->requires( which_dw, length_label, gn, 0 );
  tsk->requires( which_dw, weight_label, gn, 0 ); 

  //EqnFactory& eqn_factory = EqnFactory::self();
  DQMOMEqnFactory& dqmom_eqn_factory = DQMOMEqnFactory::self();

  tsk->requires(which_dw, d_fieldLabels->d_CCVelocityLabel, gn, 0 );
  tsk->requires(which_dw, d_fieldLabels->d_densityCPLabel, Ghost::None, 0);

  // require particle velocity
  ArchesLabel::PartVelMap::const_iterator i = d_fieldLabels->partVel.find(d_quadNode);
  tsk->requires( Task::NewDW, i->second, gn, 0 );


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
    constCCVariable<double> l_p; 
    constCCVariable<double> weight; 

    const VarLabel* rhop_label   = VarLabel::find(_density_name);
    const VarLabel* length_label = VarLabel::find(_length_name);
    const VarLabel* weight_label = VarLabel::find(_weight_name);

    which_dw->get( rho_p  , rhop_label   , matlIndex , patch , gn , 0 );
    which_dw->get( l_p    , length_label , matlIndex , patch , gn , 0 );
    which_dw->get( weight , weight_label , matlIndex , patch , gn , 0 );

    constCCVariable<Vector> partVel;
    ArchesLabel::PartVelMap::const_iterator iter = d_fieldLabels->partVel.find(d_quadNode);
    new_dw->get(partVel, iter->second, matlIndex, patch, gn, 0);

    for (CellIterator iter=patch->getCellIterator(); !iter.done(); iter++){

      IntVector c = *iter;

      double gasMag = gasVel[c].x() * gasVel[c].x() + 
                      gasVel[c].y() * gasVel[c].y() + 
                      gasVel[c].z() * gasVel[c].z();

      gasMag = pow(gasMag,0.5);

      double partMag = partVel[c].x() * partVel[c].x() + 
                       partVel[c].y() * partVel[c].y() + 
                       partVel[c].z() * partVel[c].z(); 

      partMag = pow(partMag,0.5); 

      double diff = std::abs(partMag - gasMag); 
      double Re  = diff * l_p[c] / ( _kvisc / den[c] );  
      double f;

      if(Re < 1.0) {

        f = 1.0;

      } else if(Re>1000.0) {

        f = 0.0183*Re;

      } else {

        f = 1. + .15*pow(Re, 0.687);

      }

      double t_p = ( rho_p[c] * l_p[c] * l_p[c] )/( 18.0 * _kvisc );

      if ( t_p > 0 ){ 

        model[c] = weight[c] * ( f / t_p * (gasVel[c][_dir]-partVel[c][_dir])+_gravity[_dir]) / _vel_scaling_const;
        gas_source[c] = -weight[c] * _w_scaling_const * rho_p[c] / 6.0 * _pi * f / t_p * ( gasVel[c][_dir]-partVel[c][_dir] ) * pow(l_p[c],3.0);

      } else { 

        model[c] = 0.0;
        gas_source[c] = 0.0;

      }


    }
  }
}
