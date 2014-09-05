#include <CCA/Components/Arches/CoalModels/ConstantSizeInert.h>
#include <CCA/Components/Arches/CoalModels/ParticleDensity.h>
#include <CCA/Components/Arches/TransportEqns/EqnFactory.h>
#include <CCA/Components/Arches/TransportEqns/EqnBase.h>
#include <CCA/Components/Arches/TransportEqns/DQMOMEqn.h>
#include <CCA/Components/Arches/ArchesLabel.h>
#include <CCA/Components/Arches/Directives.h>

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
ConstantSizeInertBuilder::ConstantSizeInertBuilder( const std::string         & modelName,
                                                    const vector<std::string> & reqICLabelNames,
                                                    const vector<std::string> & reqScalarLabelNames,
                                                    ArchesLabel         * fieldLabels,
                                                    SimulationStateP          & sharedState,
                                                    int qn ) :
  ModelBuilder( modelName, reqICLabelNames, reqScalarLabelNames, fieldLabels, sharedState, qn )
{
}

ConstantSizeInertBuilder::~ConstantSizeInertBuilder(){}

ModelBase* ConstantSizeInertBuilder::build() {
  return scinew ConstantSizeInert( d_modelName, d_sharedState, d_fieldLabels, d_icLabels, d_scalarLabels, d_quadNode );
}
// End Builder
//---------------------------------------------------------------------------

ConstantSizeInert::ConstantSizeInert( std::string modelName, 
                                      SimulationStateP& sharedState,
                                      ArchesLabel* fieldLabels,
                                      vector<std::string> icLabelNames, 
                                      vector<std::string> scalarLabelNames,
                                      int qn ) 
: ParticleDensity(modelName, sharedState, fieldLabels, icLabelNames, scalarLabelNames, qn)
{
  d_useLength = false;
  d_useMass = false;
}

ConstantSizeInert::~ConstantSizeInert()
{}

/**
-----------------------------------------------------------------------------
Problem Setup
-----------------------------------------------------------------------------
*/
void 
ConstantSizeInert::problemSetup(const ProblemSpecP& params)
{
  // call parent's method first
  ParticleDensity::problemSetup(params);

  ProblemSpecP db = params; 

  string label_name;
  string role_name;
  string temp_label_name;
  
  string temp_ic_name;
  string temp_ic_name_full;

  std::stringstream out;
  out << d_quadNode; 
  string node = out.str();

  // -----------------------------------------------------------------
  // Look for required internal coordinates
  ProblemSpecP db_icvars = params->findBlock("ICVars");
  if (db_icvars) {
    for (ProblemSpecP variable = db_icvars->findBlock("variable"); variable != 0; variable = variable->findNextBlock("variable") ) {

      variable->getAttribute("label",label_name);
      variable->getAttribute("role", role_name);

      temp_label_name = label_name;
      
      temp_label_name += "_qn";
      temp_label_name += node;

      // user specifies "role" of each internal coordinate
      if( role_name == "particle_length" ) {
        LabelToRoleMap[temp_label_name] = role_name;
        d_useLength = true;
      } else if( role_name == "particle_mass" ) {
        LabelToRoleMap[temp_label_name] = role_name;
        d_useMass = true;
      } else {
        string errmsg = "ERROR: Arches: ConstantSizeInert: Invalid variable role: must be \"particle_length\" or \"particle_mass\", you specified \"" + role_name + "\".";
        throw InvalidValue(errmsg,__FILE__,__LINE__);
      }
    }//end for dqmom variables
  }

  // fix the d_icLabels to point to the correct quadrature node (since there is 1 model per quad node)
  for ( vector<std::string>::iterator iString = d_icLabels.begin(); 
        iString != d_icLabels.end(); ++iString) {

    temp_ic_name        = (*iString);
    temp_ic_name_full   = temp_ic_name;

    temp_ic_name_full += "_qn";
    temp_ic_name_full += node;

    std::replace( d_icLabels.begin(), d_icLabels.end(), temp_ic_name, temp_ic_name_full);
  }

  /*
  // -----------------------------------------------------------------
  // Look for required scalars
  ProblemSpecP db_scalarvars = params->findBlock("scalarVars");
  if (db_scalarvars) {
    for( ProblemSpecP variable = db_scalarvars->findBlock("variable");
         variable != 0; variable = variable->findNextBlock("variable") ) {

      variable->getAttribute("label", label_name);
      variable->getAttribute("role",  role_name);

      // user specifies "role" of each scalar
      if( role_name == "---" ) {
        LabelToRoleMap[temp_label_name] = role_name;
      } else {
        string errmsg = "ERROR: Arches: CoalParticleHeatTransfer: Invalid scalar variable role for Simple Heat Transfer model: must be \"particle_temperature\" or \"gas_temperature\", you specified \"" + role_name + "\".";
        throw InvalidValue(errmsg,__FILE__,__LINE__);
      }
    }
  }
  */

  if( !d_useMass ) {
    string errmsg = "ERROR: Arches: ConstantSizeInert: You must specify a particle mass internal coordinate.\n";
    throw ProblemSetupException(errmsg,__FILE__,__LINE__);
  }

  if( !d_useLength ) {
    string errmsg = "ERROR: Arches: ConstantSizeInert: You must specify a particle size internal coordinate.\n";
    throw ProblemSetupException(errmsg,__FILE__,__LINE__);
  }


  ///////////////////////////////////////////


  DQMOMEqnFactory& dqmom_eqn_Factory = DQMOMEqnFactory::self();
  EqnFactory& eqn_Factory = EqnFactory::self();

  // assign labels for each required internal coordinate
  for( map<string,string>::iterator iter = LabelToRoleMap.begin();
       iter != LabelToRoleMap.end(); ++iter ) {

    EqnBase* current_eqn;
    if( dqmom_eqn_Factory.find_scalar_eqn(iter->first) ) {
      current_eqn = &(dqmom_eqn_Factory.retrieve_scalar_eqn(iter->first));
    } else if( eqn_Factory.find_scalar_eqn(iter->first) ) {
      current_eqn = &(eqn_Factory.retrieve_scalar_eqn(iter->first));
    } else {
      string errmsg = "ERROR: Arches: ConstantSizeInert: Invalid variable \"" + iter->first + "\" given for \""+iter->second+"\" role, could not find in EqnFactory or DQMOMEqnFactory!";
      throw ProblemSetupException(errmsg,__FILE__,__LINE__);
    }

    if( iter->second == "particle_length" ) {
      d_length_label = current_eqn->getTransportEqnLabel();
      d_length_scaling_constant = current_eqn->getScalingConstant();
    } else if( iter->second == "particle_mass" ) {
      d_particle_mass_label = current_eqn->getTransportEqnLabel();
      d_mass_scaling_constant = current_eqn->getScalingConstant();
    } else {
      string errmsg = "ERROR: Arches: ConstantSizeInert: Could not identify specified variable role \""+iter->second+"\".";
      throw ProblemSetupException(errmsg,__FILE__,__LINE__);
    }
  
  }

  //// set model clipping
  //db->getWithDefault( "low_clip",  d_lowModelClip,  1.0e-6 );
  //db->getWithDefault( "high_clip", d_highModelClip, 999999 );

}


//---------------------------------------------------------------------------
//Schedule computation of the particle density
//---------------------------------------------------------------------------
/**
@details
The particle density calculation is scheduled before the other model term calculations are scheduled.
*/
void
ConstantSizeInert::sched_computeParticleDensity( const LevelP& level,
                                                 SchedulerP& sched,
                                                 int timeSubStep )
{
  std::string taskname = "ConstantSizeInert::computeParticleDensity";
  Task* tsk = scinew Task(taskname, this, &ConstantSizeInert::computeParticleDensity, timeSubStep);

  Ghost::GhostType gn = Ghost::None;

  d_timeSubStep = timeSubStep;

  if( timeSubStep == 0 ) {
    tsk->computes( d_density_label );
  } else {
    tsk->modifies( d_density_label );
  }

  if( d_timeSubStep == 0 ) {
    tsk->requires(Task::OldDW, d_weight_label, gn, 0 );
    tsk->requires(Task::OldDW, d_length_label, gn, 0);
    tsk->requires(Task::OldDW, d_particle_mass_label, gn, 0 );

  } else {
    tsk->requires(Task::NewDW, d_weight_label, gn, 0 );
    tsk->requires(Task::NewDW, d_length_label, gn, 0 );
    tsk->requires(Task::NewDW, d_particle_mass_label, gn, 0 );

  }

  sched->addTask(tsk, level->eachPatch(), d_sharedState->allArchesMaterials()); 
}

//---------------------------------------------------------------------------
// Schedule computation of the particle density
//---------------------------------------------------------------------------
/**
@details

*/
void
ConstantSizeInert::computeParticleDensity( const ProcessorGroup* pc,
                                          const PatchSubset* patches,
                                          const MaterialSubset* matls,
                                          DataWarehouse* old_dw,
                                          DataWarehouse* new_dw,
                                          int timeSubStep )
{
  for( int p=0; p < patches->size(); p++ ) {  // Patch loop

    Ghost::GhostType  gn  = Ghost::None;

    const Patch* patch = patches->get(p);
    int archIndex = 0;
    int matlIndex = d_fieldLabels->d_sharedState->getArchesMaterial(archIndex)->getDWIndex(); 
    
    CCVariable<double> density;

    constCCVariable<double> weight;
    constCCVariable<double> length;
    constCCVariable<double> particle_mass;

    if( timeSubStep == 0 ) {
      
      new_dw->allocateAndPut( density, d_density_label, matlIndex, patch );
      density.initialize(0.0);

      old_dw->get( weight, d_weight_label, matlIndex, patch, gn, 0 );
      old_dw->get( length, d_length_label, matlIndex, patch, gn, 0 );
      old_dw->get( particle_mass, d_particle_mass_label, matlIndex, patch, gn, 0 );

    } else {

      new_dw->getModifiable( density, d_density_label, matlIndex, patch );

      new_dw->get( weight, d_weight_label, matlIndex, patch, gn, 0 );
      new_dw->get( length, d_length_label, matlIndex, patch, gn, 0 );
      new_dw->get( particle_mass, d_particle_mass_label, matlIndex, patch, gn, 0 );

    }

    for (CellIterator iter=patch->getCellIterator(); !iter.done(); iter++){

      IntVector c = *iter; 

      // weight - check if small
      bool weight_is_small = (weight[c] < d_w_small) || (weight[c] == 0.0);

      double unscaled_weight;
      double unscaled_length;
      double unscaled_mass;

      if(!d_unweighted && weight_is_small) {
        unscaled_weight = 0.0;
        unscaled_length = 0.0;
        unscaled_mass = 0.0;

      } else {
        unscaled_weight = weight[c]*d_w_scaling_constant;
        
        if( d_unweighted ) {
          unscaled_length = length[c]*d_length_scaling_constant;
        } else {
          unscaled_length = (length[c]*d_length_scaling_constant)/weight[c];
        }

        if( d_unweighted ) {
          unscaled_mass = particle_mass[c]*d_mass_scaling_constant;
        } else {
          unscaled_mass = (particle_mass[c]*d_mass_scaling_constant)/weight[c];
        }

      }

      density[c] = (6.0*unscaled_mass)/(pow(unscaled_length,3)*pi);

    }//end for cells

  }//end for patches

}


//---------------------------------------------------------------------------
//Schedule computation of the model
//---------------------------------------------------------------------------
/**
@details
The constant size model assumes that any length internal coordinate
associated with the ConstantSizeInert model does not change, so the
model term is always 0
*/
void 
ConstantSizeInert::sched_computeModel( const LevelP& level, SchedulerP& sched, int timeSubStep )
{
  // For constant particle size, model term is 0
  std::string taskname = "ConstantSizeInert::computeModel";
  Task* tsk = scinew Task(taskname, this, &ConstantSizeInert::computeModel, timeSubStep );

  if( timeSubStep == 0 && !d_labelSchedInit) {
    // Every model term needs to set this flag after the varLabel is computed. 
    // transportEqn.cleanUp should reinitialize this flag at the end of the time step. 
    d_labelSchedInit = true;
  }

  if( timeSubStep == 0 ) {
    tsk->computes(d_modelLabel);
    tsk->computes(d_gasLabel); 
  } else {
    tsk->modifies(d_modelLabel);
    tsk->modifies(d_gasLabel);  
  }

  sched->addTask(tsk, level->eachPatch(), d_sharedState->allArchesMaterials()); 

}

//---------------------------------------------------------------------------
// Method: Actually compute the source term 
//---------------------------------------------------------------------------
void
ConstantSizeInert::computeModel( const ProcessorGroup * pc, 
                                 const PatchSubset    * patches, 
                                 const MaterialSubset * matls, 
                                 DataWarehouse        * old_dw, 
                                 DataWarehouse        * new_dw,
                                 int timeSubStep )
{
  for( int p=0; p < patches->size(); p++ ) {  // Patch loop

    const Patch* patch = patches->get(p);
    int archIndex = 0;
    int matlIndex = d_fieldLabels->d_sharedState->getArchesMaterial(archIndex)->getDWIndex();

    CCVariable<double> constant_size_source;
    CCVariable<double> constant_size_gasSource;

    if( timeSubStep == 0 ) {

      new_dw->allocateAndPut( constant_size_source, d_modelLabel, matlIndex, patch );
      new_dw->allocateAndPut( constant_size_gasSource, d_gasLabel, matlIndex, patch ); 

    } else {

      new_dw->getModifiable( constant_size_source, d_modelLabel, matlIndex, patch ); 
      new_dw->getModifiable( constant_size_gasSource, d_gasLabel, matlIndex, patch ); 

    }

    constant_size_source.initialize(0.0);
    constant_size_gasSource.initialize(0.0);

    // ConstantSizeInert model does not change particle length

  }//end patch loop
}


