#include <CCA/Components/Arches/CoalModels/ConstantSizeCoal.h>
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
ConstantSizeCoalBuilder::ConstantSizeCoalBuilder( const std::string         & modelName,
                                                  const vector<std::string> & reqICLabelNames,
                                                  const vector<std::string> & reqScalarLabelNames,
                                                  const ArchesLabel         * fieldLabels,
                                                  SimulationStateP          & sharedState,
                                                  int qn ) :
  ModelBuilder( modelName, reqICLabelNames, reqScalarLabelNames, fieldLabels, sharedState, qn )
{
}

ConstantSizeCoalBuilder::~ConstantSizeCoalBuilder(){}

ModelBase* ConstantSizeCoalBuilder::build() {
  return scinew ConstantSizeCoal( d_modelName, d_sharedState, d_fieldLabels, d_icLabels, d_scalarLabels, d_quadNode );
}
// End Builder
//---------------------------------------------------------------------------

ConstantSizeCoal::ConstantSizeCoal( std::string modelName, 
                                    SimulationStateP& sharedState,
                                    const ArchesLabel* fieldLabels,
                                    vector<std::string> icLabelNames, 
                                    vector<std::string> scalarLabelNames,
                                    int qn ) 
: ParticleDensity(modelName, sharedState, fieldLabels, icLabelNames, scalarLabelNames, qn)
{
  d_useLength = false;
  d_useRawCoal = false;
  d_useChar = false;
  d_useMoisture = false;
}

ConstantSizeCoal::~ConstantSizeCoal()
{}

/**
-----------------------------------------------------------------------------
Problem Setup
-----------------------------------------------------------------------------
*/
void 
ConstantSizeCoal::problemSetup(const ProblemSpecP& params)
{
  // call parent's method first
  ParticleDensity::problemSetup(params);

  ProblemSpecP db = params; 
  const ProblemSpecP params_root = db->getRootNode();

  // get ash mass
  if( params_root->findBlock("CFD") ) {
    if( params_root->findBlock("CFD")->findBlock("ARCHES") ) {
      if( params_root->findBlock("CFD")->findBlock("ARCHES")->findBlock("Coal_Properties") ) {
        ProblemSpecP db_coal = params_root->findBlock("CFD")->findBlock("ARCHES")->findBlock("Coal_Properties");
        db_coal->require("initial_ash_mass", d_ash_mass);
      } else {
        throw InvalidValue("ERROR: CoalParticleHeatTransfer: problemSetup(): Missing <Coal_Properties> section in input file. Please specify the elemental composition of the coal and the initial ash mass.",__FILE__,__LINE__);
      }
    }
  }

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
      } else if( role_name == "raw_coal_mass" ) {
        LabelToRoleMap[temp_label_name] = role_name;
        d_useRawCoal = true;
      } else if( role_name == "char_mass" ) {
        LabelToRoleMap[temp_label_name] = role_name;
        d_useChar = true;
      } else if( role_name == "moisture_mass" ) {
        LabelToRoleMap[temp_label_name] = role_name;
        d_useMoisture = true;
      } else {
        string errmsg = "ERROR: Arches: ConstantSizeCoal: Invalid variable role: must be \"particle_length\" or \"raw_coal_mass\", you specified \"" + role_name + "\".";
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

  if( !d_useRawCoal ) {
    string errmsg = "ERROR: Arches: ConstantSizeCoal: You must specify a particle mass internal coordinate.\n";
    throw ProblemSetupException(errmsg,__FILE__,__LINE__);
  }

  if( !d_useLength ) {
    string errmsg = "ERROR: Arches: ConstantSizeCoal: You must specify a particle size internal coordinate.\n";
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
      string errmsg = "ERROR: Arches: ConstantSizeCoal: Invalid variable \"" + iter->first + "\" given for \""+iter->second+"\" role, could not find in EqnFactory or DQMOMEqnFactory!";
      throw ProblemSetupException(errmsg,__FILE__,__LINE__);
    }

    if( iter->second == "particle_length" ) {
      d_length_label = current_eqn->getTransportEqnLabel();
      d_length_scaling_constant = current_eqn->getScalingConstant();
    } else if( iter->second == "raw_coal_mass" ) {
      d_raw_coal_mass_label = current_eqn->getTransportEqnLabel();
      d_rc_scaling_constant = current_eqn->getScalingConstant();
    } else if( iter->second == "char_mass" ) {
      d_char_mass_label = current_eqn->getTransportEqnLabel();
      d_char_scaling_constant = current_eqn->getScalingConstant();
    } else if( iter->second == "moisture_mass" ) {
      d_moisture_mass_label = current_eqn->getTransportEqnLabel();
      d_moisture_scaling_constant = current_eqn->getScalingConstant();
    } else {
      string errmsg = "ERROR: Arches: ConstantSizeCoal: Could not identify specified variable role \""+iter->second+"\".";
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
ConstantSizeCoal::sched_computeParticleDensity( const LevelP& level,
                                                SchedulerP& sched,
                                                int timeSubStep )
{
  std::string taskname = "ConstantSizeCoal::computeParticleDensity";
  Task* tsk = scinew Task(taskname, this, &ConstantSizeCoal::computeParticleDensity, timeSubStep );

  Ghost::GhostType gn = Ghost::None;

  d_timeSubStep = timeSubStep;

  // Density labels
  if( timeSubStep == 0 ) {
    tsk->computes( d_density_label );
  } else {
    tsk->modifies( d_density_label );
  }

  if( d_timeSubStep == 0 ) {
    tsk->requires(Task::OldDW, d_weight_label, gn, 0 );
    tsk->requires(Task::OldDW, d_length_label, gn, 0);
    tsk->requires(Task::OldDW, d_raw_coal_mass_label, gn, 0 );
    if(d_useChar) {
      tsk->requires(Task::OldDW, d_char_mass_label, gn, 0 );
    }
    if(d_useMoisture) {
      tsk->requires(Task::OldDW, d_moisture_mass_label, gn, 0 );
    }

  } else {
    tsk->requires(Task::NewDW, d_weight_label, gn, 0 );
    tsk->requires(Task::NewDW, d_length_label, gn, 0 );
    tsk->requires(Task::NewDW, d_raw_coal_mass_label, gn, 0 );
    if(d_useChar) {
      tsk->requires(Task::NewDW, d_char_mass_label, gn, 0 );
    }
    if(d_useMoisture) {
      tsk->requires(Task::NewDW, d_moisture_mass_label, gn, 0 );
    }

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
ConstantSizeCoal::computeParticleDensity( const ProcessorGroup* pc,
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
    constCCVariable<double> raw_coal_mass;
    constCCVariable<double> char_mass;
    constCCVariable<double> moisture_mass;

    if( timeSubStep == 0 ) {

      new_dw->allocateAndPut( density, d_density_label, matlIndex, patch );
      density.initialize(0.0);

      old_dw->get( weight, d_weight_label, matlIndex, patch, gn, 0 );
      old_dw->get( length, d_length_label, matlIndex, patch, gn, 0 );
      old_dw->get( raw_coal_mass, d_raw_coal_mass_label, matlIndex, patch, gn, 0 );
      if(d_useChar) {
        old_dw->get( char_mass, d_char_mass_label, matlIndex, patch, gn, 0 );
      }
      if(d_useMoisture) {
        old_dw->get( moisture_mass, d_moisture_mass_label, matlIndex, patch, gn, 0 );
      }

    } else {

      new_dw->getModifiable( density, d_density_label, matlIndex, patch );

      new_dw->get( weight, d_weight_label, matlIndex, patch, gn, 0 );
      new_dw->get( length, d_length_label, matlIndex, patch, gn, 0 );
      new_dw->get( raw_coal_mass, d_raw_coal_mass_label, matlIndex, patch, gn, 0 );
      if(d_useChar) {
        new_dw->get( char_mass, d_char_mass_label, matlIndex, patch, gn, 0 );
      }
      if(d_useMoisture) {
        new_dw->get( moisture_mass, d_moisture_mass_label, matlIndex, patch, gn, 0 );
      }

    }

    for (CellIterator iter=patch->getCellIterator(); !iter.done(); iter++){

      IntVector c = *iter; 

      // weight - check if small
      bool weight_is_small = (weight[c] < d_w_small) || (weight[c] == 0.0);

      double unscaled_weight        = 0.0;
      double unscaled_length        = 0.0;
      double unscaled_rc_mass       = 0.0;
      double unscaled_char_mass     = 0.0;
      double unscaled_moisture_mass = 0.0;

      if(!d_unweighted && weight_is_small) {

        unscaled_weight = 0.0;
        unscaled_length = 0.0;
        unscaled_rc_mass = 0.0;
        if(d_useChar) {
          unscaled_char_mass = 0.0;
        }
        if(d_useMoisture) {
          unscaled_moisture_mass = 0.0;
        }

      } else {

        unscaled_weight = weight[c]*d_w_scaling_constant;

        if( d_unweighted ) {
          unscaled_length = length[c]*d_length_scaling_constant;
        } else {
          unscaled_length = (length[c]*d_length_scaling_constant)/weight[c];
        }

        if( d_unweighted ) {
          unscaled_rc_mass = raw_coal_mass[c]*d_rc_scaling_constant;
        } else {
          unscaled_rc_mass = (raw_coal_mass[c]*d_rc_scaling_constant)/weight[c];
        }

        if(d_useChar) {
          if( d_unweighted ) {
            unscaled_char_mass = char_mass[c]*d_char_scaling_constant;
          } else {
            unscaled_char_mass = (char_mass[c]*d_char_scaling_constant)/weight[c];
          }
        }
        if(d_useMoisture) {
          if( d_unweighted ) {
            unscaled_moisture_mass = moisture_mass[c]*d_moisture_scaling_constant;
          } else {
            unscaled_moisture_mass = (moisture_mass[c]*d_moisture_scaling_constant)/weight[c];
          }
        }

      }

      double m_p;
      m_p = unscaled_rc_mass;
      m_p += d_ash_mass[d_quadNode]; 
      if(d_useChar) {
        m_p += unscaled_char_mass;
      }
      if(d_useMoisture) {
        m_p += unscaled_moisture_mass;
      }

      density[c] = (6.0*m_p)/(pow(unscaled_length,3)*pi);

#ifdef DEBUG_MODELS
      if( isnan(density[c]) ) {
        proc0cout << "something is nan! from constant size density model qn " << d_quadNode << endl;
        proc0cout << "density = " << density[c] << endl;
      }
#endif

    }//end for cells

  }//end for patches

}


//---------------------------------------------------------------------------
//Schedule computation of the model
//---------------------------------------------------------------------------
/**
@details
The constant size model assumes that any length internal coordinate
associated with the ConstantSizeCoal model does not change, so the
model term is always 0
*/
void 
ConstantSizeCoal::sched_computeModel( const LevelP& level, SchedulerP& sched, int timeSubStep )
{
  // For constant particle size, model term is 0
  std::string taskname = "ConstantSizeCoal::computeModel";
  Task* tsk = scinew Task(taskname, this, &ConstantSizeCoal::computeModel, timeSubStep);

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
ConstantSizeCoal::computeModel( const ProcessorGroup * pc, 
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

    // ConstantSizeCoal model does not change particle length

  }//end patch loop
}


