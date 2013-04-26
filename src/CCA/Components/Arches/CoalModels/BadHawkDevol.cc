#include <CCA/Components/Arches/CoalModels/BadHawkDevol.h>
#include <CCA/Components/Arches/TransportEqns/EqnFactory.h>
#include <CCA/Components/Arches/TransportEqns/EqnBase.h>
#include <CCA/Components/Arches/TransportEqns/DQMOMEqn.h>
#include <CCA/Components/Arches/ArchesLabel.h>

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
BadHawkDevolBuilder::BadHawkDevolBuilder( const std::string         & modelName,
                                                  const vector<std::string> & reqICLabelNames,
                                                  const vector<std::string> & reqScalarLabelNames,
                                                  ArchesLabel         * fieldLabels,
                                                  SimulationStateP          & sharedState,
                                                  int qn ) :
  ModelBuilder( modelName, fieldLabels, reqICLabelNames, reqScalarLabelNames, sharedState, qn )
{
}

BadHawkDevolBuilder::~BadHawkDevolBuilder(){}

ModelBase* BadHawkDevolBuilder::build() {
  return scinew BadHawkDevol( d_modelName, d_sharedState, d_fieldLabels, d_icLabels, d_scalarLabels, d_quadNode );
}
// End Builder
//---------------------------------------------------------------------------

BadHawkDevol::BadHawkDevol( std::string modelName, 
                                    SimulationStateP& sharedState,
                                    ArchesLabel* fieldLabels,
                                    vector<std::string> icLabelNames, 
                                    vector<std::string> scalarLabelNames,
                                    int qn ) 
: ModelBase(modelName, sharedState, fieldLabels, icLabelNames, scalarLabelNames, qn), 
  d_fieldLabels(fieldLabels)
{
  d_quad_node = qn;
  
  compute_part_temp = false;

  // Create a label for this model
  d_modelLabel = VarLabel::create( modelName, CCVariable<double>::getTypeDescription() );

  // Create the gas phase source term associated with this model
  std::string gasSourceName = modelName + "_gasSource";
  d_gasLabel = VarLabel::create( gasSourceName, CCVariable<double>::getTypeDescription() );
}

BadHawkDevol::~BadHawkDevol()
{}

//---------------------------------------------------------------------------
// Method: Problem Setup
//---------------------------------------------------------------------------
  void 
BadHawkDevol::problemSetup(const ProblemSpecP& params, int qn)
{
  ProblemSpecP db = params; 
  compute_part_temp = false;

  // Look for required internal coordinates
  ProblemSpecP db_icvars = params->findBlock("ICVars");
  for (ProblemSpecP variable = db_icvars->findBlock("variable"); variable != 0; variable = variable->findNextBlock("variable") ) {

    string label_name;
    string role_name;
    string temp_label_name;

    variable->getAttribute("label",label_name);
    variable->getAttribute("role", role_name);

    temp_label_name = label_name;
    
    string node;
    std::stringstream out;
    out << qn;
    node = out.str();
    temp_label_name += "_qn";
    temp_label_name += node;

    // user specifies "role" of each internal coordinate
    // if it isn't an internal coordinate or a scalar, it's required explicitly
    // ( see comments in Arches::registerModels() for details )
    if ( role_name == "raw_coal_mass" ) {
      LabelToRoleMap[temp_label_name] = role_name;
    } else if( role_name == "particle_temperature" ) {  
      LabelToRoleMap[temp_label_name] = role_name;
      compute_part_temp = true;
    } else {
      std::string errmsg;
      errmsg = "Invalid variable role for Badzioch Hawksley Devolatilization model: must be \"particle_temperature\" or \"raw_coal_mass\", you specified \"" + role_name + "\".";
      throw InvalidValue(errmsg,__FILE__,__LINE__);
    }

    // set model clipping (not used yet...)
    db->getWithDefault( "low_clip",  d_lowModelClip,  1.0e-6 );
    db->getWithDefault( "high_clip", d_highModelClip, 999999 );

  }

  // Look for required scalars
  //   ( Badzioch Hawksley model doesn't use any extra scalars (yet)
  //     but if it did, this "for" loop would have to be un-commented )
  /*
  ProblemSpecP db_scalarvars = params->findBlock("scalarVars");
  for( ProblemSpecP variable = db_scalarvars->findBlock("variable");
       variable != 0; variable = variable->findNextBlock("variable") ) {

    string label_name;
    string role_name;
    string temp_label_name;

    variable->getAttribute("label", label_name);
    variable->getAttribute("role",  role_name);

    temp_label_name = label_name;

    string node;
    std::stringstream out;
    out << qn;
    node = out.str();
    temp_label_name += "_qn";
    temp_label_name += node;

    // user specifies "role" of each scalar
    // if it isn't an internal coordinate or a scalar, it's required explicitly
    // ( see comments in Arches::registerModels() for details )
    if ( role_name == "raw_coal_mass") {
      LabelToRoleMap[temp_label_name] = role_name;
    } else if( role_name == "particle_temperature" ) {  
      LabelToRoleMap[temp_label_name] = role_name;
      compute_part_temp = true;
    } else {
      std::string errmsg;
      errmsg = "Invalid variable role for Badzioch Hawksley Devolatilization model: must be \"particle_temperature\" or \"raw_coal_mass\", you specified \"" + role_name + "\".";
      throw InvalidValue(errmsg,__FILE__,__LINE__);
    }

  }
  */


  // fix the d_icLabels to point to the correct quadrature node (since there is 1 model per quad node)
  for ( vector<std::string>::iterator iString = d_icLabels.begin(); 
        iString != d_icLabels.end(); ++iString) {
    std::string temp_ic_name        = (*iString);
    std::string temp_ic_name_full   = temp_ic_name;

    std::string node;
    std::stringstream out;
    out << qn;
    node = out.str();
    temp_ic_name_full += "_qn";
    temp_ic_name_full += node;

    std::replace( d_icLabels.begin(), d_icLabels.end(), temp_ic_name, temp_ic_name_full);
  }

  // fix the d_scalarLabels to point to the correct quadrature node (since there is 1 model per quad node)
  // (Not needed for BadHawkDevol model (yet)... If it is, uncomment the block below)
  /*
  for ( vector<std::string>::iterator iString = d_scalarLabels.begin(); 
        iString != d_scalarLabels.end(); ++iString) {
    temp_ic_name      = (*iString);
    temp_ic_name_full = temp_ic_name;

    out << qn;
    node = out.str();
    temp_ic_name_full += "_qn";
    temp_ic_name_full += node;

    std::replace( d_scalarLabels.begin(), d_scalarLabels.end(), temp_ic_name, temp_ic_name_full);
  }
  */
  
}

//---------------------------------------------------------------------------
// Method: Schedule dummy initialization
//---------------------------------------------------------------------------
void
BadHawkDevol::sched_dummyInit( const LevelP& level, SchedulerP& sched ) 
{
  string taskname = "BadHawkDevol::dummyInit"; 

  Ghost::GhostType  gn = Ghost::None;

  Task* tsk = scinew Task(taskname, this, &BadHawkDevol::dummyInit);

  tsk->computes(d_modelLabel);
  tsk->computes(d_gasLabel); 

  tsk->requires( Task::OldDW, d_modelLabel, gn, 0);
  tsk->requires( Task::OldDW, d_gasLabel,   gn, 0);

  sched->addTask(tsk, level->eachPatch(), d_fieldLabels->d_sharedState->allArchesMaterials());
}

//-------------------------------------------------------------------------
// Method: Actually do the dummy initialization
//-------------------------------------------------------------------------
/** @details
This is called from ExplicitSolver::noSolve(), which skips the first timestep
 so that the initial conditions are correct.

This method was originally in ModelBase, but it requires creating CCVariables
 for the model and gas source terms, and the CCVariable type (double, Vector, &c.)
 is model-dependent.  Putting the method here eliminates if statements in 
 ModelBase and keeps the ModelBase class as generic as possible.
 */
void
BadHawkDevol::dummyInit( const ProcessorGroup* pc,
                                  const PatchSubset* patches, 
                                  const MaterialSubset* matls, 
                                  DataWarehouse* old_dw, 
                                  DataWarehouse* new_dw )
{
  for( int p=0; p < patches->size(); ++p ) {

    Ghost::GhostType  gn = Ghost::None;

    const Patch* patch = patches->get(p);
    int archIndex = 0;
    int matlIndex = d_fieldLabels->d_sharedState->getArchesMaterial(archIndex)->getDWIndex(); 

    CCVariable<double> model;
    CCVariable<double> gasSource;
    
    constCCVariable<double> oldModel;
    constCCVariable<double> oldGasSource;

    new_dw->allocateAndPut( model,     d_modelLabel, matlIndex, patch );
    new_dw->allocateAndPut( gasSource, d_gasLabel,   matlIndex, patch ); 

    old_dw->get( oldModel,     d_modelLabel, matlIndex, patch, gn, 0 );
    old_dw->get( oldGasSource, d_gasLabel,   matlIndex, patch, gn, 0 );

    model.copyData(oldModel);
    gasSource.copyData(oldGasSource);

  }
}

//---------------------------------------------------------------------------
// Method: Schedule the initialization of some variables 
//---------------------------------------------------------------------------
void 
BadHawkDevol::sched_initVars( const LevelP& level, SchedulerP& sched )
{

  std::string taskname = "BadHawkDevol::initVars";
  Task* tsk = scinew Task(taskname, this, &BadHawkDevol::initVars);

  sched->addTask(tsk, level->eachPatch(), d_sharedState->allArchesMaterials()); 
}

//-------------------------------------------------------------------------
// Method: Initialize variables
//-------------------------------------------------------------------------
void
BadHawkDevol::initVars( const ProcessorGroup * pc, 
                                 const PatchSubset    * patches, 
                                 const MaterialSubset * matls, 
                                 DataWarehouse        * old_dw, 
                                 DataWarehouse        * new_dw )
{
  // No special local variables for this model...
  for( int p=0; p < patches->size(); p++ ) {  // Patch loop
  }
}

//---------------------------------------------------------------------------
// Method: Schedule the calculation of the Model 
//---------------------------------------------------------------------------
void 
BadHawkDevol::sched_computeModel( const LevelP& level, SchedulerP& sched, int timeSubStep )
{
  std::string taskname = "BadHawkDevol::computeModel";
  Task* tsk = scinew Task(taskname, this, &BadHawkDevol::computeModel);

  Ghost::GhostType gn = Ghost::None;

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

  //EqnFactory& eqn_factory = EqnFactory::self();
  DQMOMEqnFactory& dqmom_eqn_factory = DQMOMEqnFactory::self();

  // construct the weight label corresponding to this quad node
  std::string temp_weight_name = "w_qn";
  std::string node;
  std::stringstream out;
  out << d_quad_node;
  node = out.str();
  temp_weight_name += node;
  EqnBase& t_weight_eqn = dqmom_eqn_factory.retrieve_scalar_eqn( temp_weight_name );
  DQMOMEqn& weight_eqn = dynamic_cast<DQMOMEqn&>(t_weight_eqn);
  d_weight_label = weight_eqn.getTransportEqnLabel();
  d_w_small = weight_eqn.getSmallClip();
  d_w_scaling_factor = weight_eqn.getScalingConstant();
  tsk->requires(Task::OldDW, d_weight_label, gn, 0);

  // require gas temperature
  tsk->requires(Task::OldDW, d_fieldLabels->d_tempINLabel, Ghost::AroundCells, 1);

  // For each required variable, determine what role it plays
  // - "gas_temperature" - require the "tempIN" label
  // - "particle_temperature" - look in DQMOMEqnFactory
  // - "raw_coal_mass" - look in DQMOMEqnFactory


  // for each required internal coordinate:
  for (vector<std::string>::iterator iter = d_icLabels.begin(); 
      iter != d_icLabels.end(); iter++) { 

    map<string, string>::iterator iMap = LabelToRoleMap.find(*iter);

    if ( iMap != LabelToRoleMap.end() ) {
      if ( iMap->second == "particle_temperature") {
        if (dqmom_eqn_factory.find_scalar_eqn(*iter) ) {
          EqnBase& t_current_eqn = dqmom_eqn_factory.retrieve_scalar_eqn(*iter);
          DQMOMEqn& current_eqn = dynamic_cast<DQMOMEqn&>(t_current_eqn);
          d_particle_temperature_label = current_eqn.getTransportEqnLabel();
          d_pt_scaling_factor = current_eqn.getScalingConstant();
          tsk->requires(Task::OldDW, d_particle_temperature_label, Ghost::None, 0);
        } else {
          std::string errmsg = "ARCHES: BadHawkDevol: Invalid variable given in <variable> tag for BadHawkDevol model";
          errmsg += "\nCould not find given particle temperature variable \"";
          errmsg += *iter;
          errmsg += "\" in EqnFactory or in DQMOMEqnFactory.";
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
          std::string errmsg = "ARCHES: BadHawkDevol: Invalid variable given in <variable> tag for BadHawkDevol model";
          errmsg += "\nCould not find given raw coal mass variable \"";
          errmsg += *iter;
          errmsg += "\" in DQMOMEqnFactory.";
          throw InvalidValue(errmsg,__FILE__,__LINE__);
        }
      }

    } else {
      // can't find this required variable in the labels-to-roles map!
      std::string errmsg = "ARCHES: BadHawkDevol: You specified that the variable \"" + *iter + 
                           "\" was required, but you did not specify a role for it!\n";
      throw InvalidValue( errmsg, __FILE__, __LINE__);
    }
  }
  
  // for each required scalar variable:
  //  (but no scalar equation variables should be required for the BadHawkDevol model, at least not for now...)
  /*
  for( vector<std::string>::iterator iter = d_scalarLabels.begin();
       iter != d_scalarLabels.end(); ++iter) {
    map<string, string>::iterator iMap = LabelToRoleMap.find(*iter);
    
    if( iMap != LabelToRoleMap.end() ) {
      if( iMap->second == <insert role name here> ) {
        if( eqn_factory.find_scalar_eqn(*iter) ) {
          EqnBase& current_eqn = eqn_factory.retrieve_scalar_eqn(*iter);
          d_<insert role name here>_label = current_eqn.getTransportEqnLabel();
          tsk->requires(Task::OldDW, d_<insert role name here>_label, Ghost::None, 0);
        } else {
          std::string errmsg = "ARCHES: BadHawkDevol: Invalid variable given in <scalarVars> block for <variable> tag for BadHawkDevol model.";
          errmsg += "\nCould not find given <insert role name here> variable \"";
          errmsg += *iter;
          errmsg += "\" in EqnFactory.";
          throw InvalidValue(errmsg,__FILE__,__LINE__);
        }
      }
    } else {
      // can't find this required variable in the labels-to-roles map!
      std::string errmsg = "ARCHES: BadHawkDevol: You specified that the variable \"" + *iter + 
                           "\" was required, but you did not specify a role for it!\n";
      throw InvalidValue( errmsg, __FILE__, __LINE__);
    }

  } //end for
  */

  sched->addTask(tsk, level->eachPatch(), d_sharedState->allArchesMaterials()); 

}

//---------------------------------------------------------------------------
// Method: Actually compute the source term 
//---------------------------------------------------------------------------
void
BadHawkDevol::computeModel( const ProcessorGroup * pc, 
    const PatchSubset    * patches, 
    const MaterialSubset * matls, 
    DataWarehouse        * old_dw, 
    DataWarehouse        * new_dw )
{
  for( int p=0; p < patches->size(); p++ ) {  // Patch loop

    //Ghost::GhostType  gaf = Ghost::AroundFaces;
    Ghost::GhostType  gac = Ghost::AroundCells;
    Ghost::GhostType  gn  = Ghost::None;

    const Patch* patch = patches->get(p);
    int archIndex = 0;
    int matlIndex = d_fieldLabels->d_sharedState->getArchesMaterial(archIndex)->getDWIndex(); 

    CCVariable<double> devol_rate;
    if( new_dw->exists( d_modelLabel, matlIndex, patch ) ) {
      new_dw->getModifiable( devol_rate, d_modelLabel, matlIndex, patch ); 
    } else {
      new_dw->allocateAndPut( devol_rate, d_modelLabel, matlIndex, patch );
      devol_rate.initialize(0.0);
    }

    CCVariable<double> gas_devol_rate; 
    if (new_dw->exists( d_gasLabel, matlIndex, patch )){
      new_dw->getModifiable( gas_devol_rate, d_gasLabel, matlIndex, patch ); 
    } else {
      new_dw->allocateAndPut( gas_devol_rate, d_gasLabel, matlIndex, patch ); 
      gas_devol_rate.initialize(0.0);

    }

    constCCVariable<double> temperature;
    if (compute_part_temp) {
      old_dw->get( temperature, d_particle_temperature_label, matlIndex, patch, gn, 0 );
    } else {
      old_dw->get( temperature, d_fieldLabels->d_tempINLabel, matlIndex, patch, gac, 1 );
    }
    
    constCCVariable<double> wa_raw_coal_mass;
    old_dw->get( wa_raw_coal_mass, d_raw_coal_mass_label, matlIndex, patch, gn, 0 );

    constCCVariable<double> weight;
    old_dw->get( weight, d_weight_label, matlIndex, patch, gn, 0 );

    for (CellIterator iter=patch->getCellIterator(); !iter.done(); iter++){

      IntVector c = *iter; 

      if ((weight[c] < d_w_small) && !d_unweighted) {
        devol_rate[c] = 0.0;
        gas_devol_rate[c] = 0.0;
      } else {
        if(compute_part_temp) {
          double particle_temperature;
          if(d_unweighted){
            particle_temperature = temperature[c]*d_pt_scaling_factor;
          } else {
            particle_temperature = temperature[c]*d_pt_scaling_factor/weight[c];  
          }   
          k1 = A1*exp(E1/(R*particle_temperature)); // 1/s
          k2 = A2*exp(E2/(R*particle_temperature)); // 1/s     
        } else {   
          k1 = A1*exp(E1/(R*temperature[c])); // 1/s
          k2 = A2*exp(E2/(R*temperature[c])); // 1/s
        }

        double raw_coal_mass = wa_raw_coal_mass[c];
        if(d_unweighted){
          raw_coal_mass = wa_raw_coal_mass[c];
        } else {
          raw_coal_mass = wa_raw_coal_mass[c] / weight[c];
        }
        double testVal = (k1+k2)>1 ? -raw_coal_mass : -1.0*(k1+k2)*(raw_coal_mass));  
        if (testVal < -1.0e-16 )
          devol_rate[c] = testVal;  
        else 
          devol_rate[c] = 0.0;
    
        // what is Y1_ and Y2_??
        if(d_unweighted){
          testVal = (Y1_*k1 + Y2_*k2)*wa_raw_coal_mass[c]*weight[c]*d_rc_scaling_factor*d_w_scaling_factor;
        } else {
          testVal = (Y1_*k1 + Y2_*k2)*wa_raw_coal_mass[c]*d_rc_scaling_factor*d_w_scaling_factor; 
        }
 
        //testVal uses the weighted abscissa so that the gas source is from all (total) particles
        if (testVal > 1.0e-16 )
          gas_devol_rate[c] = testVal; 
        else 
          gas_devol_rate[c] = 0.0;

      }
    }
  }
}

