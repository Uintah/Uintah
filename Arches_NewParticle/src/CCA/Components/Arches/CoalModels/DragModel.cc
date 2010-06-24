#include <CCA/Components/Arches/CoalModels/DragModel.h>
#include <CCA/Components/Arches/CoalModels/ParticleVelocity.h>
#include <CCA/Components/Arches/TransportEqns/EqnFactory.h>
#include <CCA/Components/Arches/TransportEqns/EqnBase.h>
#include <CCA/Components/Arches/TransportEqns/DQMOMEqn.h>
#include <CCA/Components/Arches/ArchesLabel.h>
#include <CCA/Components/Arches/Directives.h>
#include <CCA/Components/Arches/BoundaryCond_new.h>

#include <Core/ProblemSpec/ProblemSpec.h>
#include <CCA/Ports/Scheduler.h>
#include <Core/Grid/SimulationState.h>
#include <Core/Grid/Variables/VarTypes.h>
#include <Core/Grid/Variables/CCVariable.h>
#include <Core/Exceptions/InvalidValue.h>
#include <Core/Parallel/Parallel.h>

//===========================================================================

using namespace std;
using namespace Uintah; 

//---------------------------------------------------------------------------
// Builder:
DragModelBuilder::DragModelBuilder( const std::string         & modelName, 
                                    const vector<std::string> & reqICLabelNames,
                                    const vector<std::string> & reqScalarLabelNames,
                                    const ArchesLabel         * fieldLabels,
                                    SimulationStateP          & sharedState,
                                    int qn ) :
  ModelBuilder( modelName, reqICLabelNames, reqScalarLabelNames, fieldLabels, sharedState, qn )
{
}

DragModelBuilder::~DragModelBuilder(){}

ModelBase* DragModelBuilder::build(){
  return scinew DragModel( d_modelName, d_sharedState, d_fieldLabels, d_icLabels, d_scalarLabels, d_quadNode );
}
// End Builder
//---------------------------------------------------------------------------

DragModel::DragModel( std::string modelName, 
                      SimulationStateP& sharedState,
                      const ArchesLabel* fieldLabels,
                      vector<std::string> icLabelNames, 
                      vector<std::string> scalarLabelNames,
                      int qn ) 
: ParticleVelocity(modelName, sharedState, fieldLabels, icLabelNames, scalarLabelNames, qn)
{
  pi = 3.141592653589793;
  d_quadNode = qn;

  // particle velocity label is created in parent class
}

DragModel::~DragModel()
{}

//---------------------------------------------------------------------------
// Method: Problem Setup
//---------------------------------------------------------------------------
void 
DragModel::problemSetup(const ProblemSpecP& params)
{
  // call parent's method first
  ParticleVelocity::problemSetup(params);

  ProblemSpecP db = params; 

  string label_name;
  string role_name;
  string temp_label_name;
  string node;
  std::stringstream out;

  // Look for required internal coordinates
  ProblemSpecP db_icvars = params->findBlock("ICVars");
  for (ProblemSpecP variable = db_icvars->findBlock("variable"); variable != 0; variable = variable->findNextBlock("variable") ) {

    variable->getAttribute("label",label_name);
    variable->getAttribute("role", role_name);

    temp_label_name = label_name;
    
    out << d_quadNode;
    node = out.str();
    temp_label_name += "_qn";
    temp_label_name += node;

    // user specifies "role" of each internal coordinate
    // if it isn't an internal coordinate or a scalar, it's required explicitly
    // ( see comments in Arches::registerModels() for details )
    if (role_name == "length") {
      LabelToRoleMap[temp_label_name] = role_name;
      d_length_set = true;
    } else if ( role_name == "u_velocity" ) {
      LabelToRoleMap[temp_label_name] = role_name;
      d_uvel_set = true;
    } else if ( role_name == "v_velocity" ) {
      LabelToRoleMap[temp_label_name] = role_name;
      d_vvel_set = true;
    } else if ( role_name == "w_velocity" ) {
      LabelToRoleMap[temp_label_name] = role_name;
      d_wvel_set = true;
    } else {
      std::string errmsg;
      errmsg = "ERROR: DragModel: problemSetup: Invalid variable role for Drag model: must be \"length\", \"u_velocity\", \"v_velocity\", or \"w_velocity\", you specified \"" + role_name + "\".";
      throw InvalidValue(errmsg,__FILE__,__LINE__);
    }
  }

  if ( !d_length_set  || !d_uvel_set
       || !d_vvel_set || !d_wvel_set ) {
    std::string errmsg;
    std::string whichic;
    if(!d_length_set) {
      whichic = "\"length\"";
    } else if (!d_uvel_set) {
      whichic = "\"u_velocity\"";
    } else if (!d_vvel_set) {
      whichic = "\"v_velocity\"";
    } else if (!d_wvel_set) {
      whichic = "\"w_velocity\"";
    }
    errmsg = "ERROR: DragModel: problemSetup: missing internal coordinate "+whichic+", add this to the list of internal coordinates required by the model in your input file.\n";
    throw InvalidValue(errmsg,__FILE__,__LINE__);
  }

  string temp_ic_name;
  string temp_ic_name_full;

  // fix the d_icLabels to point to the correct quadrature node (since there is 1 model per quad node)
  for ( vector<std::string>::iterator iString = d_icLabels.begin(); 
        iString != d_icLabels.end(); ++iString) {
    temp_ic_name      = (*iString);
    temp_ic_name_full = temp_ic_name;

    out << d_quadNode;
    node = out.str();
    temp_ic_name_full += "_qn";
    temp_ic_name_full += node;

    std::replace( d_icLabels.begin(), d_icLabels.end(), temp_ic_name, temp_ic_name_full);
  }


  DQMOMEqnFactory& dqmom_eqn_factory = DQMOMEqnFactory::self();

  // construct the weight label corresponding to this quad node
  std::string temp_weight_name = "w_qn";
  out << d_quadNode;
  node = out.str();
  temp_weight_name += node;
  EqnBase& t_weight_eqn = dqmom_eqn_factory.retrieve_scalar_eqn( temp_weight_name );
  DQMOMEqn& weight_eqn = dynamic_cast<DQMOMEqn&>(t_weight_eqn);
  d_weight_label = weight_eqn.getTransportEqnLabel();
  d_w_small = weight_eqn.getSmallClip();
  d_w_scaling_factor = weight_eqn.getScalingConstant();
  
  // Fill d_velocityLabels with particle velocity internal coordinate labels
  for( vector<std::string>::iterator iter = d_icLabels.begin();
       iter != d_icLabels.end(); ++iter ) {

    map<string, string>::iterator iMap = LabelToRoleMap.find(*iter);

    if( iMap != LabelToRoleMap.end() ) {

      if( iMap->second == "u_velocity" ) {
        if( dqmom_eqn_factory.find_scalar_eqn(*iter) ) {
          EqnBase& t_current_eqn = dqmom_eqn_factory.retrieve_scalar_eqn(*iter);
          DQMOMEqn& current_eqn = dynamic_cast<DQMOMEqn&>(t_current_eqn);
          d_uvel_label = current_eqn.getTransportEqnLabel();
          d_uvel_scaling_factor = current_eqn.getScalingConstant();
        } else {
          std::string errmsg = "ARCHES: DragModel: Invalid variable given in <variable> tag for Drag model";
          errmsg += "\nCould not find given particle u-velocity variable \"";
          errmsg += *iter;
          errmsg += "\" in EqnFactory or in DQMOMEqnFactory.";
          throw InvalidValue(errmsg,__FILE__,__LINE__);
        }
  
      } else if ( iMap->second == "v_velocity" ) {
        if( dqmom_eqn_factory.find_scalar_eqn(*iter) ) {
          EqnBase& t_current_eqn = dqmom_eqn_factory.retrieve_scalar_eqn(*iter);
          DQMOMEqn& current_eqn = dynamic_cast<DQMOMEqn&>(t_current_eqn);
          d_vvel_label = current_eqn.getTransportEqnLabel();
          d_vvel_scaling_factor = current_eqn.getScalingConstant();
        } else {
          std::string errmsg = "ARCHES: DragModel: Invalid variable given in <variable> tag for Drag model";
          errmsg += "\nCould not find given particle v-velocity variable \"";
          errmsg += *iter;
          errmsg += "\" in EqnFactory or in DQMOMEqnFactory.";
          throw InvalidValue(errmsg,__FILE__,__LINE__);
        }
    
      } else if ( iMap->second == "w_velocity" ) {
        if( dqmom_eqn_factory.find_scalar_eqn(*iter) ) {
          EqnBase& t_current_eqn = dqmom_eqn_factory.retrieve_scalar_eqn(*iter);
          DQMOMEqn& current_eqn = dynamic_cast<DQMOMEqn&>(t_current_eqn);
          d_wvel_label = current_eqn.getTransportEqnLabel();
          d_wvel_scaling_factor = current_eqn.getScalingConstant();
        } else {
          std::string errmsg = "ARCHES: DragModel: Invalid variable given in <variable> tag for Drag model";
          errmsg += "\nCould not find given particle w-velocity variable \"";
          errmsg += *iter;
          errmsg += "\" in EqnFactory or in DQMOMEqnFactory.";
          throw InvalidValue(errmsg,__FILE__,__LINE__);
        }
    
      } else if ( iMap->second == "particle_length" ) {
        if (dqmom_eqn_factory.find_scalar_eqn(*iter) ) {
          EqnBase& t_current_eqn = dqmom_eqn_factory.retrieve_scalar_eqn(*iter);
          DQMOMEqn& current_eqn = dynamic_cast<DQMOMEqn&>(t_current_eqn);
          d_length_label = current_eqn.getTransportEqnLabel();
          d_length_scaling_factor = current_eqn.getScalingConstant();
        } else {
          std::string errmsg = "ARCHES: DragModel: Invalid variable given in <variable> tag for Drag model";
          errmsg += "\nCould not find given particle length variable \"";
          errmsg += *iter;
          errmsg += "\" in EqnFactory or in DQMOMEqnFactory.";
          throw InvalidValue(errmsg,__FILE__,__LINE__);
        }
      } //else... we don't need that variable!

    } else {
      // can't find this required variable in the labels-to-roles map!
      std::string errmsg = "ARCHES: DragModel: You specified that the variable \"" + *iter + 
                           "\" was required, but you did not specify a role for it!\n";
      throw InvalidValue(errmsg,__FILE__,__LINE__);
    }
  }
}


//---------------------------------------------------------------------------
// Method: Schedule the calculation of the Model 
//---------------------------------------------------------------------------
void 
DragModel::sched_computeModel( const LevelP& level, 
                               SchedulerP& sched, 
                               int timeSubStep )
{
  // calculate model source term for dqmom velocity internal coodinate
  std::string taskname = "DragModel::computeModel";
  Task* tsk = scinew Task(taskname, this, &DragModel::computeModel, timeSubStep);

  CoalModelFactory& coal_model_factory = CoalModelFactory::self();

  Ghost::GhostType  gn  = Ghost::None;

  d_timeSubStep = timeSubStep; 

  if (d_timeSubStep == 0 && !d_labelSchedInit) {
    // Every model term needs to set this flag after the varLabel is computed. 
    // transportEqn.cleanUp should reinitialize this flag at the end of the time step. 
    d_labelSchedInit = true;

    tsk->computes(d_modelLabel);
    tsk->computes(d_gasLabel);
  } else {
    tsk->modifies(d_modelLabel); 
    tsk->modifies(d_gasLabel);
  }
  
  // require gas density
  if ( timeSubStep == 0 ) {
    tsk->requires(Task::OldDW, d_fieldLabels->d_densityCPLabel,   gn, 0);
  } else {
    tsk->requires(Task::NewDW, d_fieldLabels->d_densityTempLabel, gn, 0);
  }

  // require gas velocity
  tsk->requires( Task::OldDW, d_fieldLabels->d_newCCVelocityLabel, gn, 0 );

  // require particle velocity (this quantity is calculated from the internal coordinate values)
  tsk->requires( Task::OldDW, d_velocity_label, gn, 0 );

  // require particle density
  tsk->requires( Task::OldDW, coal_model_factory.getParticleDensityLabel( d_quadNode ), gn, 0);

  // require weights
  tsk->requires( Task::OldDW, d_weight_label, gn, 0);

  // reqiure internal coordiantes
  //tsk->requires(Task::OldDW, d_uvel_label, gn, 0);  // <-- These are available via d_velocity_label
  //tsk->requires(Task::OldDW, d_vvel_label, gn, 0);  //     (that quantity is calculated from the internal coordinate values
  //tsk->requires(Task::OldDW, d_wvel_label, gn, 0);  //      in DragModel::computeParticleVelocity() )
  tsk->requires(Task::OldDW, d_length_label, gn, 0);

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
                         int timeSubStep )
{
  for (int p=0; p < patches->size(); p++){

    Ghost::GhostType  gn  = Ghost::None;

    const Patch* patch = patches->get(p);
    int archIndex = 0;
    int matlIndex = d_fieldLabels->d_sharedState->getArchesMaterial(archIndex)->getDWIndex();

    CoalModelFactory& coal_model_factory = CoalModelFactory::self();

    CCVariable<Vector> drag_particle;
    if (new_dw->exists( d_modelLabel, matlIndex, patch )){
      new_dw->getModifiable( drag_particle, d_modelLabel, matlIndex, patch ); 
    } else {
      new_dw->allocateAndPut( drag_particle, d_modelLabel, matlIndex, patch );
      drag_particle.initialize(Vector(0.,0.,0.));
    }

    CCVariable<Vector> drag_gas; 
    if (new_dw->exists( d_gasLabel, matlIndex, patch )){
      new_dw->getModifiable( drag_gas, d_gasLabel, matlIndex, patch ); 
    } else {
      new_dw->allocateAndPut( drag_gas, d_gasLabel, matlIndex, patch );
      drag_gas.initialize(Vector(0.,0.,0.));
    }

    constCCVariable<double> gas_density;
    if( timeSubStep == 0 ) {
      old_dw->get( gas_density, d_fieldLabels->d_densityCPLabel, matlIndex, patch, gn, 0);
    } else {
      new_dw->get( gas_density, d_fieldLabels->d_densityTempLabel, matlIndex, patch, gn, 0);
    }

    constCCVariable<double> particle_density;
    old_dw->get( particle_density, coal_model_factory.getParticleDensityLabel( d_quadNode ), matlIndex, patch, gn, 0);

    constCCVariable<Vector> gas_velocity;
    old_dw->get( gas_velocity, d_fieldLabels->d_newCCVelocityLabel, matlIndex, patch, gn, 0);

    constCCVariable<Vector> particle_velocity;
    old_dw->get( particle_velocity, d_velocity_label, matlIndex, patch, gn, 0);

    constCCVariable<double> w_particle_length; 
    old_dw->get( w_particle_length, d_length_label, matlIndex, patch, gn, 0 );

    constCCVariable<double> weight;
    old_dw->get( weight, d_weight_label, matlIndex, patch, gn, 0 );
    
    for (CellIterator iter=patch->getCellIterator(); !iter.done(); iter++){
      IntVector c = *iter; 

      if( weight[c] < d_w_small ) {
        drag_particle[c] = Vector(0.,0.,0.);
        drag_gas[c] = Vector(0.,0.,0.);
      } else {
        double length = w_particle_length[c]/weight[c]*d_length_scaling_factor;   

        Vector gasVel = gas_velocity[c];
        Vector partVel = particle_velocity[c];

        double rhop = particle_density[c];

        double diff = fabs(gasVel.length() - partVel.length());
        double Re  = (diff*length)/d_visc;

        double phi;
        if(Re < 1) {
          phi = 1;
        } else {
          phi = 1.0 + 0.15*pow(Re, 0.687);
        }

        double t_p = particle_density[c]/(18*d_visc)*pow(length,2); 

        double part_src_mag = phi/t_p*diff;
        drag_particle[c] = ( partVel.safe_normalize() )*(part_src_mag);
        
        double gas_src_mag = -(weight[c]*d_w_scaling_factor)*rhop*(4/3)*pi*(phi/t_p)*diff*pow(length,3);
        drag_gas[c] = ( gasVel.safe_normalize() )*(gas_src_mag);

       }  
    }
  }
}

void
DragModel::sched_computeParticleVelocity( const LevelP& level,
                                          SchedulerP&   sched,
                                          const int timeSubStep )
{
  string taskname = "DragModel::computeParticleVelocity";
  Task* tsk = scinew Task(taskname, this, &DragModel::computeParticleVelocity);

  Ghost::GhostType gn = Ghost::None;

  //DQMOMEqnFactory& dqmom_eqn_factory = DQMOMEqnFactory::self();
  //CoalModelFactory& coal_model_factory = CoalModelFactory::self();

  // setting particle velocity (Vector)
  if( timeSubStep == 0 ) {
    tsk->computes( d_velocity_label );
  } else {
    tsk->modifies( d_velocity_label );
  }

  // require particle velocity internal coordinate labels (assembling these into a Vector)
  tsk->requires( Task::OldDW, d_uvel_label, gn, 0);
  tsk->requires( Task::OldDW, d_vvel_label, gn, 0);
  tsk->requires( Task::OldDW, d_wvel_label, gn, 0);

  // require weights 
  tsk->requires( Task::OldDW, d_weight_label, gn, 0);

  // gas velocity
  tsk->requires( Task::OldDW, d_fieldLabels->d_newCCVelocityLabel, gn, 0);

  // NOTE: don't need anything else, because the particle velocity
  //       has already been calculated (via DQMOM method).
  //       We're simply setting the particle velocity vector components 
  //       equal to these DQMOM scalars.
}

void 
DragModel::computeParticleVelocity( const ProcessorGroup* pc,
                                    const PatchSubset*    patches,
                                    const MaterialSubset* matls,
                                    DataWarehouse*        old_dw,
                                    DataWarehouse*        new_dw )
{
  for( int p=0; p<patches->size(); ++p ) {
    Ghost::GhostType gn = Ghost::None;

    const Patch* patch = patches->get(p);
    int archIndex = 0;
    int matlIndex = d_fieldLabels->d_sharedState->getArchesMaterial(archIndex)->getDWIndex();

    CCVariable<Vector> particle_velocity;
    if( new_dw->exists( d_velocity_label, matlIndex, patch ) ) {
      new_dw->getModifiable( particle_velocity, d_velocity_label, matlIndex, patch );
    } else {
      new_dw->allocateAndPut( particle_velocity, d_velocity_label, matlIndex, patch );
      particle_velocity.initialize( Vector(0.0, 0.0, 0.0) );
    }

    constCCVariable<double> weight;
    old_dw->get( weight, d_weight_label, matlIndex, patch, gn, 0 );

    constCCVariable<double> wtd_particle_uvel;
    old_dw->get( wtd_particle_uvel, d_uvel_label, matlIndex, patch, gn, 0 );

    constCCVariable<double> wtd_particle_vvel;
    old_dw->get( wtd_particle_vvel, d_vvel_label, matlIndex, patch, gn, 0 );

    constCCVariable<double> wtd_particle_wvel;
    old_dw->get( wtd_particle_wvel, d_wvel_label, matlIndex, patch, gn, 0 );

    constCCVariable<Vector> gas_velocity;
    old_dw->get( gas_velocity, d_fieldLabels->d_newCCVelocityLabel, matlIndex, patch, gn, 0);

    for (CellIterator iter=patch->getCellIterator(); !iter.done(); iter++){
      IntVector c = *iter; 

      double U = (wtd_particle_uvel[c]/weight[c])*d_uvel_scaling_factor;
      double V = (wtd_particle_vvel[c]/weight[c])*d_vvel_scaling_factor;
      double W = (wtd_particle_wvel[c]/weight[c])*d_wvel_scaling_factor;
      particle_velocity[c] = Vector(U,V,W);
    }

    // NOTE: When tracking particle velocity as an internal coordinate,
    //       it is broken up into three separate scalars.
    //       HOWEVER, we still have to treat it as a Vector in the code,
    //       so when the boundary conditions for particle velocity are set,
    //       they are set as a Vector
    //
    //       Velocity boundary conditions and velocity internal coordinate
    //       boundary values are set via "vel_qn" vectors at each boundary,
    //       NOT through the DQMOM scalars!!!
    //
    // e.g., if the input file contains
    // 
    //      <BCType id = "0" label = "vel_qn0" var = "Dirichlet">
    //        <value>[5.0,1.0,3.0]</value>
    //      </BCType>
    //
    // Then "vel_qn0" will be set to Vector(5.0, 1.0, 3.0),
    // and "uvel_internal_coordinate" will be set to 5.0
    //     "vvel_internal_coordinate" will be set to 1.0
    //     "wvel_internal_coordinate" will be set to 3.0
    // (no boundary condition is explicitly set for the internal coordinates)
    //
    // This keeps from having to specify the boundary condition twice
    // (and potential mistakes from using two different values)

    // now that vel field is set, apply boundary conditions
    string name = "vel_qn";
    string node;
    std::stringstream out; 
    out << d_quadNode; 
    node = out.str(); 
    name += node; 
    if (d_gasBC) {
      // assume particle velocity = gas velocity at boundary
      // DON'T DO THIS, IT'S WRONG!
      d_boundaryCond->setVectorValueBC( 0, patch, particle_velocity, gas_velocity, name );
    } else {
      // Particle velocity at boundary is set by user
      d_boundaryCond->setVectorValueBC( 0, patch, particle_velocity, name);
    }

  }//end patch loop
}


void
DragModel::computeModel( const ProcessorGroup * pc, 
                         const PatchSubset    * patches, 
                         const MaterialSubset * matls, 
                         DataWarehouse        * old_dw, 
                         DataWarehouse        * new_dw )
{
}



