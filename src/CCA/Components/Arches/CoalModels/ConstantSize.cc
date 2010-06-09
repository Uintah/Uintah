#include <CCA/Components/Arches/CoalModels/ConstantSize.h>
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
ConstantSizeBuilder::ConstantSizeBuilder( const std::string         & modelName,
                                          const vector<std::string> & reqICLabelNames,
                                          const vector<std::string> & reqScalarLabelNames,
                                          const ArchesLabel         * fieldLabels,
                                          SimulationStateP          & sharedState,
                                          int qn ) :
  ModelBuilder( modelName, reqICLabelNames, reqScalarLabelNames, fieldLabels, sharedState, qn )
{
}

ConstantSizeBuilder::~ConstantSizeBuilder(){}

ModelBase* ConstantSizeBuilder::build() {
  return scinew ConstantSize( d_modelName, d_sharedState, d_fieldLabels, d_icLabels, d_scalarLabels, d_quadNode );
}
// End Builder
//---------------------------------------------------------------------------

ConstantSize::ConstantSize( std::string modelName, 
                                              SimulationStateP& sharedState,
                                              const ArchesLabel* fieldLabels,
                                              vector<std::string> icLabelNames, 
                                              vector<std::string> scalarLabelNames,
                                              int qn ) 
: ParticleDensity(modelName, sharedState, fieldLabels, icLabelNames, scalarLabelNames, qn)
{
  pi = 3.1415926535;
}

ConstantSize::~ConstantSize()
{}

/**
-----------------------------------------------------------------------------
Problem Setup
-----------------------------------------------------------------------------
*/
void 
ConstantSize::problemSetup(const ProblemSpecP& params)
{
  // call parent's method first
  ParticleDensity::problemSetup(params);

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
      // if it isn't an internal coordinate or a scalar, it's required explicitly
      // ( see comments in Arches::registerModels() for details )
      if (    role_name == "length"
           || role_name == "raw_coal_mass" 
           /*
           || role_name == "moisture_mass" 
           || role_name == "char_mass"
           */ ) {
        LabelToRoleMap[temp_label_name] = role_name;
      } else {
        std::string errmsg;
        errmsg = "Invalid variable role for Constant Size model: must be \"length\" or \"raw_coal_mass\", you specified \"" + role_name + "\".";
        //errmsg = "Invalid variable role for Constant Size model: must be \"length\", \"raw_coal_mass\", \"char_mass\", or \"moisture_mass\", you specified \"" + role_name + "\".";
        throw InvalidValue(errmsg,__FILE__,__LINE__);
      }
    }
  }

  // fix the d_icLabels to point to the correct quadrature node (since there is 1 model per quad node)
  for ( vector<std::string>::iterator iString = d_icLabels.begin(); 
        iString != d_icLabels.end(); ++iString) {

    temp_ic_name        = (*iString);
    temp_ic_name_full   = temp_ic_name;

    std::stringstream out;
    out << d_quadNode;
    string node = out.str();
    temp_ic_name_full += "_qn";
    temp_ic_name_full += node;

    std::replace( d_icLabels.begin(), d_icLabels.end(), temp_ic_name, temp_ic_name_full);
  }

}

/**
---------------------------------------------------------------------------
Schedule computation of the model

@details
The constant size model assumes that any length internal coordinate
associated with the ConstantSize model does not change, so the
model term is always 0
---------------------------------------------------------------------------
*/
void 
ConstantSize::sched_computeModel( const LevelP& level, SchedulerP& sched, int timeSubStep )
{
  // For constant particle size, model term is 0
  std::string taskname = "ConstantSize::computeModel";
  Task* tsk = scinew Task(taskname, this, &ConstantSize::computeModel);

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

  sched->addTask(tsk, level->eachPatch(), d_sharedState->allArchesMaterials()); 

}

//---------------------------------------------------------------------------
// Method: Actually compute the source term 
//---------------------------------------------------------------------------
void
ConstantSize::computeModel( const ProcessorGroup * pc, 
                            const PatchSubset    * patches, 
                            const MaterialSubset * matls, 
                            DataWarehouse        * old_dw, 
                            DataWarehouse        * new_dw )
{
  for( int p=0; p < patches->size(); p++ ) {  // Patch loop

    //Ghost::GhostType  gaf = Ghost::AroundFaces;
    //Ghost::GhostType  gac = Ghost::AroundCells;
    //Ghost::GhostType  gn  = Ghost::None;

    const Patch* patch = patches->get(p);
    int archIndex = 0;
    int matlIndex = d_fieldLabels->d_sharedState->getArchesMaterial(archIndex)->getDWIndex();

    CCVariable<double> constant_size_source;
    if( new_dw->exists( d_modelLabel, matlIndex, patch ) ) {
      new_dw->getModifiable( constant_size_source, d_modelLabel, matlIndex, patch ); 
    } else {
      new_dw->allocateAndPut( constant_size_source, d_modelLabel, matlIndex, patch );
    }
    constant_size_source.initialize(0.0);

    CCVariable<double> constant_size_gasSource;
    if (new_dw->exists( d_gasLabel, matlIndex, patch )){
      new_dw->getModifiable( constant_size_gasSource, d_gasLabel, matlIndex, patch ); 
    } else {
      new_dw->allocateAndPut( constant_size_gasSource, d_gasLabel, matlIndex, patch ); 
    }
    constant_size_gasSource.initialize(0.0);

    // ConstantSize model does not change particle length

  }//end patch loop
}

void
ConstantSize::sched_computeParticleDensity( const LevelP& level,
                                            SchedulerP& sched,
                                            int timeSubStep )
{
  std::string taskname = "ConstantSize::computeParticleDensity";
  Task* tsk = scinew Task(taskname, this, &ConstantSize::computeParticleDensity);

  Ghost::GhostType gn = Ghost::None;

  d_timeSubStep = timeSubStep;

  // Density labels
  if( timeSubStep == 0 ) {
    tsk->computes( d_density_label );
  } else {
    tsk->computes( d_density_label );
  }

  DQMOMEqnFactory& dqmom_eqn_factory = DQMOMEqnFactory::self();

  // construct the weight label corresponding to this quad node
  std::string temp_weight_name = "w_qn";
  std::string node;
  std::stringstream out;
  out << d_quadNode;
  node = out.str();
  temp_weight_name += node;
  EqnBase& t_weight_eqn = dqmom_eqn_factory.retrieve_scalar_eqn( temp_weight_name );
  DQMOMEqn& weight_eqn = dynamic_cast<DQMOMEqn&>(t_weight_eqn);
  d_weight_label = weight_eqn.getTransportEqnLabel();
  tsk->requires(Task::OldDW, d_weight_label, gn, 0);

  // For each required variable, determine what role it plays
  // - "length"
  // - "raw_coal_mass"
  // - "moisture_mass"
  // - "char_mass"

  // for each required internal coordinate:
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
          tsk->requires(Task::OldDW, d_length_label, Ghost::None, 0);
        } else {
          std::string errmsg = "ARCHES: ConstantSize: Invalid variable given in <variable> tag for ConstantSize model";
          errmsg += "\nCould not find given length variable \"";
          errmsg += *iter;
          errmsg += "\" in DQMOMEqnFactory.";
          throw InvalidValue(errmsg,__FILE__,__LINE__);
        }

      } else if ( iMap->second == "raw_coal_mass") {
        if (dqmom_eqn_factory.find_scalar_eqn(*iter) ) {
          EqnBase& t_current_eqn = dqmom_eqn_factory.retrieve_scalar_eqn(*iter);
          DQMOMEqn& current_eqn = dynamic_cast<DQMOMEqn&>(t_current_eqn);
          d_raw_coal_mass_label = current_eqn.getTransportEqnLabel();
          d_rc_scaling_factor = current_eqn.getScalingConstant();
          tsk->requires(Task::OldDW, d_raw_coal_mass_label, Ghost::None, 0);
        } else {
          std::string errmsg = "ARCHES: ConstantSize: Invalid variable given in <variable> tag for ConstantSize model";
          errmsg += "\nCould not find given raw coal mass variable \"";
          errmsg += *iter;
          errmsg += "\" in DQMOMEqnFactory.";
          throw InvalidValue(errmsg,__FILE__,__LINE__);
        }

      } /*else if ( iMap->second == "char_mass" ) {
        if (dqmom_eqn_factory.find_scalar_eqn(*iter) ) {
          EqnBase& t_current_eqn = dqmom_eqn_factory.retrieve_scalar_eqn(*iter);
          DQMOMEqn& current_eqn = dynamic_cast<DQMOMEqn&>(t_current_eqn);
          d_char_mass_label = current_eqn.getTransportEqnLabel();
          d_char_scaling_factor = current_eqn.getScalingConstant();
          tsk->requires(Task::OldDW, d_char_mass_label, Ghost::None, 0);
        } else {
          std::string errmsg = "ARCHES: ConstantSize: Invalid variable given in <variable> tag for ConstantSize model";
          errmsg += "\nCould not find given char mass variable \"";
          errmsg += *iter;
          errmsg += "\" in DQMOMEqnFactory.";
          throw InvalidValue(errmsg,__FILE__,__LINE__);
        }

      } else if ( iMap->second == "moisture_mass" ) {
        if (dqmom_eqn_factory.find_scalar_eqn(*iter) ) {
          EqnBase& t_current_eqn = dqmom_eqn_factory.retrieve_scalar_eqn(*iter);
          DQMOMEqn& current_eqn = dynamic_cast<DQMOMEqn&>(t_current_eqn);
          d_moisture_mass_label = current_eqn.getTransportEqnLabel();
          d_moisture_scaling_factor = current_eqn.getScalingConstant();
          tsk->requires(Task::OldDW, d_moisture_mass_label, Ghost::None, 0);
        } else {
          std::string errmsg = "ARCHES: ConstantSize: Invalid variable given in <variable> tag for ConstantSize model";
          errmsg += "\nCould not find given moisture mass variable \"";
          errmsg += *iter;
          errmsg += "\" in DQMOMEqnFactory.";
          throw InvalidValue(errmsg,__FILE__,__LINE__);
        }
      }
      */
    } else {
      // reached end of map, so can't find required variable in labels-to-roles map
      std::string errmsg = "ARCHES: ConstantSize: You specified that the variable \"" + *iter + 
                           "\" was required, but you did not specify a role for it!\n";
      throw InvalidValue( errmsg, __FILE__, __LINE__);
    }
  }
  
  // for each required scalar variable:

  for( vector<std::string>::iterator iter = d_scalarLabels.begin();
       iter != d_scalarLabels.end(); ++iter) {
    map<string, string>::iterator iMap = LabelToRoleMap.find(*iter);
    
    /*
    if( iMap != LabelToRoleMap.end() ) {
      if( iMap->second == "" ) {
        if( eqn_factory.find_scalar_eqn(*iter) ) {
          EqnBase& current_eqn = eqn_factory.retrieve_scalar_eqn(*iter);
          d_""_label = current_eqn.getTransportEqnLabel();
          tsk->requires(Task::OldDW, d_""_label, Ghost::None, 0);
        } else {
          std::string errmsg = "ARCHES: ConstantSize: Invalid variable given in <scalarVars> block for <variable> tag for ConstantSize model.";
          errmsg += "\nCould not find given \"\" variable \"";
          errmsg += *iter;
          errmsg += "\" in EqnFactory.";
          throw InvalidValue(errmsg,__FILE__,__LINE__);
        }
      }
    } else {
      // can't find this required variable in the labels-to-roles map!
      std::string errmsg = "ARCHES: ConstantSize: You specified that the variable \"" + *iter + 
                           "\" was required, but you did not specify a role for it!\n";
      throw InvalidValue( errmsg, __FILE__, __LINE__);
    }
    */

  } //end for

  sched->addTask(tsk, level->eachPatch(), d_sharedState->allArchesMaterials()); 
}

void
ConstantSize::computeParticleDensity( const ProcessorGroup* pc,
                                      const PatchSubset* patches,
                                      const MaterialSubset* matls,
                                      DataWarehouse* old_dw,
                                      DataWarehouse* new_dw )
{
  for( int p=0; p < patches->size(); p++ ) {  // Patch loop

    //Ghost::GhostType  gaf = Ghost::AroundFaces;
    //Ghost::GhostType  gac = Ghost::AroundCells;
    Ghost::GhostType  gn  = Ghost::None;

    const Patch* patch = patches->get(p);
    int archIndex = 0;
    int matlIndex = d_fieldLabels->d_sharedState->getArchesMaterial(archIndex)->getDWIndex(); 

    // compute density
    CCVariable<double> density;
    if( new_dw->exists( d_density_label, matlIndex, patch) ) {
      new_dw->getModifiable( density, d_density_label, matlIndex, patch );
    } else {
      new_dw->allocateAndPut( density, d_density_label, matlIndex, patch );
      density.initialize(0.0);
    }

    constCCVariable<double> raw_coal_mass;
    old_dw->get( raw_coal_mass, d_raw_coal_mass_label, matlIndex, patch, gn, 0 );

    constCCVariable<double> length;
    old_dw->get( length, d_length_label, matlIndex, patch, gn, 0 );

    //constCCVariable<double> char_mass;
    //old_dw->get( char_mass, d_char_mass_label, matlIndex, patch, gn, 0 );

    //constCCVariable<double> moisture_mass;
    //old_dw->get( moisture_mass, d_moisture_mass_label, matlIndex, patch, gn, 0 );

    for (CellIterator iter=patch->getCellIterator(); !iter.done(); iter++){
      IntVector c = *iter; 

      double m_p = raw_coal_mass[c] + ash_mass[d_quadNode]; // particle mass

      density[c] = ( m_p / pow(length[c],3) )*( 6.0 / pi );
    }

  }
}


