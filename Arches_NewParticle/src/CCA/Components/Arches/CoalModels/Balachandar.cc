#include <CCA/Components/Arches/CoalModels/Balachandar.h>
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
BalachandarBuilder::BalachandarBuilder( const std::string         & modelName,
                                        const vector<std::string> & reqICLabelNames,
                                        const vector<std::string> & reqScalarLabelNames,
                                        const ArchesLabel         * fieldLabels,
                                        SimulationStateP          & sharedState,
                                        int qn ) :
  ModelBuilder( modelName, reqICLabelNames, reqScalarLabelNames, fieldLabels, sharedState, qn )
{
}

BalachandarBuilder::~BalachandarBuilder(){}

ModelBase* BalachandarBuilder::build() {
  return scinew Balachandar( d_modelName, d_sharedState, d_fieldLabels, d_icLabels, d_scalarLabels, d_quadNode );
}
// End Builder
//---------------------------------------------------------------------------

Balachandar::Balachandar( std::string modelName, 
                          SimulationStateP& sharedState,
                          const ArchesLabel* fieldLabels,
                          vector<std::string> icLabelNames, 
                          vector<std::string> scalarLabelNames,
                          int qn ) 
: ParticleVelocity(modelName, sharedState, fieldLabels, icLabelNames, scalarLabelNames, qn)
{
  // NOTE: there is one Balachandar object for each quad node
  // so, no need to have a vector of VarLabels

  // particle velocity label is created in parent class
}

Balachandar::~Balachandar()
{}

//---------------------------------------------------------------------------
// Method: Problem Setup
//---------------------------------------------------------------------------
  void 
Balachandar::problemSetup(const ProblemSpecP& params)
{
  // call parent's method first
  ParticleVelocity::problemSetup(params);

  ProblemSpecP db = params; 

  // no longer using <kinematic_viscosity> b/c grabbing kinematic viscosity from physical properties section
  // no longer using <rho_ratio> b/c density obtained dynamically
  // no longer using <beta> because density (and beta) will be obtained dynamically
  db->getWithDefault("iter", d_totIter, 15);
  db->getWithDefault("tol", d_tol, 1e-15);
  db->getWithDefault("L", d_L, 1.0);
  db->getWithDefault("eta", d_eta, 1e-5);
  db->getWithDefault("regime", regime, 1);
  db->getWithDefault("min_vel_ratio", d_min_vel_ratio, 0.1);

  if (regime==1) {
    d_power = 1;
  } else if (regime == 2) {
    d_power = 0.5;
  } else if (regime == 3) {
    d_power = 1.0/3.0;
  }

  string label_name;
  string role_name;
  string temp_label_name;
  
  string temp_ic_name;
  string temp_ic_name_full;

  // -----------------------------------------------------------------
  // Look for required internal coordinates
  ProblemSpecP db_icvars = params->findBlock("ICVars");
  if (db_icvars) {
    for (ProblemSpecP variable = db_icvars->findBlock("variable"); variable != 0; variable = variable->findNextBlock("variable") ) {

      variable->getAttribute("label",label_name);
      variable->getAttribute("role", role_name);

      temp_label_name = label_name;
      
      std::stringstream out;
      out << d_quadNode;
      string node = out.str();
      temp_label_name += "_qn";
      temp_label_name += node;

      // user specifies "role" of each internal coordinate
      if ( role_name == "length" ) {
        LabelToRoleMap[temp_label_name] = role_name;
      } else {
        std::string errmsg;
        errmsg = "Invalid variable role for Balachandar particle velocity model: must be \"length\", \"u_velocity\", \"v_velocity\", or \"w_velocity\", you specified \"" + role_name + "\".";
        throw InvalidValue(errmsg,__FILE__,__LINE__);
      }
    }
  }

  // fix the d_icLabels to point to the correct quadrature node (since there is 1 model per quad node)
  for ( vector<std::string>::iterator iString = d_icLabels.begin(); 
        iString != d_icLabels.end(); ++iString) {

    temp_ic_name        = *iString;
    temp_ic_name_full   = temp_ic_name;

    std::stringstream out;
    out << d_quadNode;
    string node = out.str();
    temp_ic_name_full += "_qn";
    temp_ic_name_full += node;

    std::replace( d_icLabels.begin(), d_icLabels.end(), temp_ic_name, temp_ic_name_full);
  }
 
  DQMOMEqnFactory& dqmom_eqn_factory = DQMOMEqnFactory::self();

  // assign labels for each internal coordinate
  for (vector<std::string>::iterator iter = d_icLabels.begin(); 
      iter != d_icLabels.end(); iter++) { 

    map<string, string>::iterator iMap = LabelToRoleMap.find(*iter);
    
    if ( iMap != LabelToRoleMap.end() ) {
      if ( iMap->second == "length") {
        if (dqmom_eqn_factory.find_scalar_eqn(*iter) ) {
          EqnBase& t_current_eqn = dqmom_eqn_factory.retrieve_scalar_eqn(*iter);
          DQMOMEqn& current_eqn = dynamic_cast<DQMOMEqn&>(t_current_eqn);
          d_length_label = current_eqn.getTransportEqnLabel();
          d_length_scaling_factor = current_eqn.getScalingConstant();
        } else {
          std::string errmsg = "ARCHES: Balachandar: Invalid variable given in <variable> tag for Balachandar model";
          errmsg += "\nCould not find given particle length variable \"";
          errmsg += *iter;
          errmsg += "\" in EqnFactory or in DQMOMEqnFactory.";
          throw InvalidValue(errmsg,__FILE__,__LINE__);
        }
      }
    } else {
      // can't find this required variable in the labels-to-roles map!
      std::string errmsg = "ARCHES: Balachandar: You specified that the variable \"" + *iter + 
                           "\" was required, but you did not specify a role for it!\n";
      throw InvalidValue( errmsg, __FILE__, __LINE__);
    }
  }//end for ic labels

  // FIXME
  // set the x-, y-, and z-velocity VarLabels
  
  // // set model clipping (not used)
  // db->getWithDefault( "low_clip", d_lowModelClip,   1.0e-6 );
  // db->getWithDefault( "high_clip", d_highModelClip, 999999 );  


}


//-------------------------------------------------------------------------
// Method: Actually do the dummy initialization
//-------------------------------------------------------------------------
/** @details
This method intentionally left blank. 
@seealso ParticleVelocity::dummyInit
*/
void
Balachandar::dummyInit( const ProcessorGroup* pc,
                        const PatchSubset* patches, 
                        const MaterialSubset* matls, 
                        DataWarehouse* old_dw, 
                        DataWarehouse* new_dw )
{
}

//---------------------------------------------------------------------------
// Method: Schedule the initialization of special variables unique to model
//---------------------------------------------------------------------------
/** @details
This method intentionally left blank. 
@seealso ParticleVelocity::sched_initVars
*/
void 
Balachandar::sched_initVars( const LevelP& level, SchedulerP& sched )
{
}

//-------------------------------------------------------------------------
// Method: Initialize special variables unique to the model
//-------------------------------------------------------------------------
/** @details
This method intentionally left blank. 
@seealso ParticleVelocity::initVars
*/
void
Balachandar::initVars( const ProcessorGroup * pc, 
                       const PatchSubset    * patches, 
                       const MaterialSubset * matls, 
                       DataWarehouse        * old_dw, 
                       DataWarehouse        * new_dw )
{
}

//---------------------------------------------------------------------------
// Method: Schedule the calculation of the Model 
//---------------------------------------------------------------------------
void 
Balachandar::sched_computeModel( const LevelP& level, SchedulerP& sched, int timeSubStep )
{
  std::string taskname = "Balachandar::computeModel";
  Task* tsk = scinew Task(taskname, this, &Balachandar::computeModel);

  //Ghost::GhostType gn = Ghost::None;

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

  // The Balachandar model is not associated with any internal coordinates
  // So, it doesn't have any model term to calculate

  sched->addTask(tsk, level->eachPatch(), d_sharedState->allArchesMaterials()); 

}

//---------------------------------------------------------------------------
// Method: Actually compute the source term 
//---------------------------------------------------------------------------
/**
@details
The Balachandar model doesn't have any model term G associated with
 it (since it is not a source term for an internal coordinate).
Therefore, the computeModel() method simply sets this model term
 equal to zero.
*/
void
Balachandar::computeModel( const ProcessorGroup * pc, 
                           const PatchSubset    * patches, 
                           const MaterialSubset * matls, 
                           DataWarehouse        * old_dw, 
                           DataWarehouse        * new_dw )
{
  for( int p=0; p < patches->size(); p++ ) {

    const Patch* patch = patches->get(p);
    int archIndex = 0;
    int matlIndex = d_fieldLabels->d_sharedState->getArchesMaterial(archIndex)->getDWIndex(); 

    CCVariable<Vector> model;
    if( new_dw->exists( d_modelLabel, matlIndex, patch ) ) {
      new_dw->getModifiable( model, d_modelLabel, matlIndex, patch ); 
    } else {
      new_dw->allocateAndPut( model, d_modelLabel, matlIndex, patch );
      model.initialize(Vector(0.0,0.0,0.0));
    }

    CCVariable<Vector> model_gasSource;
    if (new_dw->exists( d_gasLabel, matlIndex, patch )){
      new_dw->getModifiable( model_gasSource, d_gasLabel, matlIndex, patch ); 
    } else {
      new_dw->allocateAndPut( model_gasSource, d_gasLabel, matlIndex, patch ); 
      model_gasSource.initialize(Vector(0.0,0.0,0.0));
    }

  }//end patch loop
}

void
Balachandar::sched_computeParticleVelocity( const LevelP& level,
                                            SchedulerP&   sched,
                                            const int timeSubStep )
{
  string taskname = "Balachandar::computeParticleVelocity";
  Task* tsk = scinew Task(taskname, this, &Balachandar::computeParticleVelocity);

  Ghost::GhostType gn = Ghost::None;

  CoalModelFactory& coal_model_factory = CoalModelFactory::self();

  if( timeSubStep == 0 ) {
    tsk->computes( d_velocity_label );
  } else {
    tsk->modifies( d_velocity_label );
  }

  // use the old particle velocity
  tsk->requires( Task::OldDW, d_velocity_label, gn, 0);

  // gas velocity
  tsk->requires(Task::OldDW, d_fieldLabels->d_newCCVelocityLabel, Ghost::None, 0);

  // gas density
  tsk->requires(Task::OldDW, d_fieldLabels->d_densityCPLabel, Ghost::None, 0);

  // density label for this environment/quad node
  const VarLabel* density_label = coal_model_factory.getParticleDensityLabel(d_quadNode);
  tsk->requires( Task::OldDW, density_label, gn, 0);

  // require weight label
  tsk->requires(Task::OldDW, d_weight_label, gn, 0);

  // require internal coordinates
  tsk->requires(Task::OldDW, d_length_label, Ghost::None, 0);
}

void
Balachandar::computeParticleVelocity( const ProcessorGroup* pc,
                                      const PatchSubset*    patches,
                                      const MaterialSubset* matls,
                                      DataWarehouse*        old_dw,
                                      DataWarehouse*        new_dw )
{
  for( int p=0; p < patches->size(); p++ ) {  // Patch loop

    //Ghost::GhostType  gaf = Ghost::AroundFaces;
    //Ghost::GhostType  gac = Ghost::AroundCells;
    Ghost::GhostType  gn  = Ghost::None;

    const Patch* patch = patches->get(p);
    int archIndex = 0;
    int matlIndex = d_fieldLabels->d_sharedState->getArchesMaterial(archIndex)->getDWIndex(); 

    CoalModelFactory& coal_model_factory = CoalModelFactory::self();

    //CCVariable<Vector> particle_velocity;
    //if( new_dw->exists( d_velocity_label, matlIndex, patch) ) {
    //  new_dw->getModifiable( particle_velocity, d_velocity_label, matlIndex, patch );
    //} else {
    //  new_dw->allocateAndPut( particle_velocity, d_velocity_label, matlIndex, patch );
    //  particle_velocity.initialize(Vector(0.0,0.0,0.0));
    //}

    constCCVariable<double> weight;
    old_dw->get(weight, d_weight_label, matlIndex, patch, gn, 0);

    constCCVariable<double> wtd_length;
    old_dw->get(wtd_length, d_length_label, matlIndex, patch, gn, 0);

    CCVariable<Vector> particle_velocity;
    if( new_dw->exists( d_velocity_label, matlIndex, patch) ) {
      new_dw->getModifiable( particle_velocity, d_velocity_label, matlIndex, patch );
    } else {
      new_dw->allocateAndPut( particle_velocity, d_velocity_label, matlIndex, patch );
      particle_velocity.initialize( Vector(0.0,0.0,0.0) );
    }

    /*
    CCVariable<double> particle_u_velocity;
    if( new_dw->exists( d_uvel_label, matlIndex, patch) ) {
      new_dw->getModifiable( particle_u_velocity, d_uvel_label, matlIndex, patch );
    } else {
      new_dw->allocateAndPut( particle_u_velocity, d_uvel_label, matlIndex, patch );
      particle_u_velocity.initialize(0.0);
    }

    CCVariable<double> particle_v_velocity;
    if( new_dw->exists( d_vvel_label, matlIndex, patch) ) {
      new_dw->getModifiable( particle_v_velocity, d_vvel_label, matlIndex, patch );
    } else {
      new_dw->allocateAndPut( particle_v_velocity, d_vvel_label, matlIndex, patch );
      particle_v_velocity.initialize(0.0);
    }

    CCVariable<double> particle_w_velocity;
    if( new_dw->exists( d_wvel_label, matlIndex, patch) ) {
      new_dw->getModifiable( particle_w_velocity, d_wvel_label, matlIndex, patch );
    } else {
      new_dw->allocateAndPut( particle_w_velocity, d_wvel_label, matlIndex, patch );
      particle_w_velocity.initialize(0.0);
    }
    */

    constCCVariable<Vector> old_particle_velocity;
    old_dw->get( old_particle_velocity, d_velocity_label, matlIndex, patch, gn, 0 );

    /*
    constCCVariable<double> old_particle_u_velocity;
    old_dw->get( old_particle_u_velocity, d_uvel_label, matlIndex, patch, gn, 0 );

    constCCVariable<double> old_particle_v_velocity;
    old_dw->get( old_particle_v_velocity, d_vvel_label, matlIndex, patch, gn, 0 );

    constCCVariable<double> old_particle_w_velocity;
    old_dw->get( old_particle_w_velocity, d_wvel_label, matlIndex, patch, gn, 0 );
    */

    constCCVariable<Vector> gas_velocity;
    old_dw->get( gas_velocity, d_fieldLabels->d_newCCVelocityLabel, matlIndex, patch, gn, 0 );

    constCCVariable<double> gas_density;
    old_dw->get( gas_density, d_fieldLabels->d_densityCPLabel, matlIndex, patch, gn, 0 );

    constCCVariable<double> particle_density;
    old_dw->get( particle_density, coal_model_factory.getParticleDensityLabel(d_quadNode), matlIndex, patch, gn, 0 );

    for (CellIterator iter=patch->getCellIterator(); !iter.done(); iter++){
      IntVector c = *iter; 

      Vector gasvel = gas_velocity[c];
      double length;

      if( weight[c] < d_w_small ) {
        particle_velocity[c] = gas_velocity[c];
      } else {
        length = (wtd_length[c]/weight[c])*d_length_scaling_factor;

        Vector gasVel = gas_velocity[c];
        Vector oldPartVel = old_particle_velocity[c];
        Vector newPartVel = Vector(0.0, 0.0, 0.0);

        // do a loop over each velocity component
        for( int j=0; j<3; ++j) {

          double length_ratio = 0.0;
          epsilon = pow(gasVel[j], 3);
          epsilon /= d_L;

          length_ratio = length / d_eta;
          double uk = 0.0;

          if (length > 0.0) {
            uk = pow(d_eta/d_L, 1.0/3.0);
            uk *= gasVel[j];
          }

          double diff = 0.0;
          double prev_diff = 0.0;

          // now iterate to convergence
          for (int iter=0; iter<d_totIter; ++iter) {
            prev_diff = diff;
            double Re = fabs(diff)*length / kvisc; // do we really want a componentwise Re?
            double phi = 1.0 + 0.15*pow(Re, 0.687);
            double t_p_by_t_k = ( (2*rhoRatio + 1)/36.0 )*( 1/phi )*( pow(length_ratio,2) );

            diff = uk*(1-beta)*pow(t_p_by_t_k, d_power);
            double error = abs(diff - prev_diff)/diff;

            if( abs(diff) < 1e-16 ) {
              error = 0.0;
            }
            if( abs(error) < d_tol ) {
              break;
            }
          }

          double newPartMag = gasVel[j] - diff;

          newPartVel[j] = newPartMag;
        }

        particle_velocity[c] = newPartVel;

      }//end if weight is small

    }//end cell iterator

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

