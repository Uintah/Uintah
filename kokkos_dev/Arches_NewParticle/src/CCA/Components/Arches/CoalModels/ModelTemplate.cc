#include <CCA/Components/Arches/CoalModels/TemplateName.h>
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
TemplateNameBuilder::TemplateNameBuilder( const std::string         & modelName,
                                                            const vector<std::string> & reqICLabelNames,
                                                            const vector<std::string> & reqScalarLabelNames,
                                                            ArchesLabel         * fieldLabels,
                                                            SimulationStateP          & sharedState,
                                                            int qn ) :
  ModelBuilder( modelName, reqICLabelNames, reqScalarLabelNames, fieldLabels, sharedState, qn )
{
}

TemplateNameBuilder::~TemplateNameBuilder(){}

ModelBase* TemplateNameBuilder::build() {
  return scinew TemplateName( d_modelName, d_sharedState, d_fieldLabels, d_icLabels, d_scalarLabels, d_quadNode );
}
// End Builder
//---------------------------------------------------------------------------

TemplateName::TemplateName( std::string modelName, 
                                              SimulationStateP& sharedState,
                                              ArchesLabel* fieldLabels,
                                              vector<std::string> icLabelNames, 
                                              vector<std::string> scalarLabelNames,
                                              int qn ) 
: ModelBase(modelName, sharedState, fieldLabels, icLabelNames, scalarLabelNames, qn)
{
  // gas/model labels are created in parent class
}

TemplateName::~TemplateName()
{}

//-----------------------------------------------------------------------------
//Problem Setup
//-----------------------------------------------------------------------------
void 
TemplateName::problemSetup(const ProblemSpecP& params)
{
  ProblemSpecP db = params; 

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
      if ( role_name == "required_variable" ) {
        LabelToRoleMap[temp_label_name] = role_name;
        d_useRequiredVariable = true;
      } else {
        std::string errmsg;
        errmsg = "ERROR: Arches: TemplateName: Invalid variable role for DQMOM equation: must be \"required_variable\", you specified \"" + role_name + "\".";
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

  // -----------------------------------------------------------------
  // Look for required scalars
  ProblemSpecP db_scalarvars = params->findBlock("scalarVars");
  if (db_scalarvars) {
    for( ProblemSpecP variable = db_scalarvars->findBlock("variable");
         variable != 0; variable = variable->findNextBlock("variable") ) {

      variable->getAttribute("label", label_name);
      variable->getAttribute("role",  role_name);

      // user specifies "role" of each scalar
      if ( role_name == "required_variable_2" ) {
        LabelToRoleMap[label_name] = role_name;
        d_useRequiredVariable2 = true;
      } else {
        std::string errmsg;
        errmsg = "ERROR: Arches: TemplateName: Invalid variable role for scalar equation: must be \"required_variable_2\", you specified \"" + role_name + "\".";
        throw InvalidValue(errmsg,__FILE__,__LINE__);
      }
    }
  }

  if(!d_useRequiredVariable) {
    string errmsg = "ERROR: Arches: TemplateName: No required variable internal coordinate was specified.  Quitting...";
    throw ProblemSetupException(errmsg,__FILE__,__LINE__);
  }

  if(!d_useRequiredVariable2) {
    string errmsg = "ERROR: Arches: TemplateName: No required variable 2 scalar variable was specified.  Quitting...";
    throw ProblemSetupException(errmsg,__FILE__,__LINE__);
  }


  ///////////////////////////////////////////


  DQMOMEqnFactory& dqmom_eqn_factory = DQMOMEqnFactory::self();
  EqnFactory& eqn_factory = EqnFactory::self();

  // assign labels for each required internal coordinate
  for( map<string,string>::iterator iter = LabelToRoleMap.begin();
       iter != LabelToRoleMap.end(); ++iter ) {

    EqnBase* current_eqn;
    if( dqmom_eqn_factory.find_scalar_eqn(iter->first) ) {
      current_eqn = &(dqmom_eqn_factory.retrieve_scalar_eqn(iter->first));
    } else if( eqn_factory.find_scalar_eqn(iter->first) ) {
      current_eqn = &(eqn_factory.retrieve_scalar_eqn(iter->first));
    } else {
      string errmsg = "ERROR: Arches: TempateName: Invalid variable \"" + iter->first + "\" given for \""+iter->second+"\" role, could not find in EqnFactory or DQMOMEqnFactory!";
      throw ProblemSetupException(errmsg,__FILE__,__LINE__);
    }

    if( iter->second == "required_variable" ){
      d_required_variable_label = current_eqn->getTransportEqnLabel();
      d_rv_scaling_constant = current_eqn->getScalingConstant();
    } else if( iter->second == "required_variable_2" ){
      d_required_variable_2_label = current_eqn->getTransportEqnLabel();
      d_rv2_scaling_constant = current_eqn->getScalingConstant();
    } else {
      // can't find this required variable in the labels-to-roles map!
      std::string errmsg = "ERROR: Arches: TemplateName: You specified that the variable \"" + iter->first + 
                           "\" was required, but you did not specify a valid role for it! (You specified \"" + iter->second + "\")\n";
      throw InvalidValue( errmsg, __FILE__, __LINE__);
    }
  
  }//end for ic/scalar labels
  
  // // set model clipping (not used)
  //db->getWithDefault( "low_clip", d_lowModelClip,   1.0e-6 );
  //db->getWithDefault( "high_clip", d_highModelClip, 999999 );  

}



//-----------------------------------------------------------------------------
//Schedule the calculation of the Model 
//-----------------------------------------------------------------------------
void 
TemplateName::sched_computeModel( const LevelP& level, SchedulerP& sched, int timeSubStep )
{
  std::string taskname = "TemplateName::computeModel";
  Task* tsk = scinew Task(taskname, this, &TemplateName::computeModel, timeSubStep);
  
  Ghost::GhostType gn = Ghost::None;

  d_timeSubStep = timeSubStep; 

  // require timestep label
  tsk->requires(Task::OldDW, d_fieldLabels->d_sharedState->get_delt_label() );

  if (d_timeSubStep == 0 && !d_labelSchedInit) {
    // Every model term needs to set this flag after the varLabel is computed. 
    // transportEqn.cleanUp should reinitialize this flag at the end of the time step. 
    d_labelSchedInit = true;
  }

  if( timeSubStep == 0 ) {
    tsk->computes(d_modelLabel);
    tsk->computes(d_gasLabel); 

    tsk->requires(Task::OldDW, d_weight_label, gn, 0);
    tsk->requires(Task::OldDW, d_required_variable_label, gn, 0);
    tsk->requires(Task::OldDW, d_required_variable_2_label, gn, 0);

  } else {
    tsk->modifies(d_modelLabel);
    tsk->modifies(d_gasLabel);  

    tsk->requires(Task::NewDW, d_weight_label, gn, 0);
    tsk->requires(Task::NewDW, d_required_variable_label, gn, 0);
    tsk->requires(Task::NewDW, d_required_variable_2_label, gn, 0);
  }

  sched->addTask(tsk, level->eachPatch(), d_sharedState->allArchesMaterials()); 

}



//-----------------------------------------------------------------------------
//Actually compute the source term 
//-----------------------------------------------------------------------------
void
TemplateName::computeModel( const ProcessorGroup * pc, 
                            const PatchSubset    * patches, 
                            const MaterialSubset * matls, 
                            DataWarehouse        * old_dw, 
                            DataWarehouse        * new_dw,
                            int timeSubStep )
{
  for( int p=0; p < patches->size(); p++ ) {  // Patch loop

    Ghost::GhostType  gn  = Ghost::None;

    const Patch* patch = patches->get(p);
    int archIndex = 0;
    int matlIndex = d_fieldLabels->d_sharedState->getArchesMaterial(archIndex)->getDWIndex(); 

    /*
    delt_vartype delta_t;
    old_dw->get( delta_t, d_fieldLabels->d_sharedState->get_delt_label() );
    double dt = delta_t;
    */

    constCCVariable<double> weight;
    constCCVariable<double> wa_required_variable;
    constCCVariable<double> required_variable_2;

    CCVariable<double> model_term;
    CCVariable<double> gas_model_term;

    if( timeSubStep == 0 ) {
      old_dw->get( weight,               d_weight_label,              matlIndex, patch, gn, 0 );
      old_dw->get( wa_required_variable, d_required_variable_label,   matlIndex, patch, gn, 0 );
      old_dw->get( required_variable_2,  d_required_variable_2_label, matlIndex, patch, gn, 0 );

      new_dw->allocateAndPut( model_term,     d_modelLabel,    matlIndex, patch );
      new_dw->allocateAndPut( gas_model_term, d_gasModelLabel, matlIndex, patch );

      model_term.initialize(0.0);
      gas_model_term.initialize(0.0);
      
    } else {
      new_dw->get( weight,               d_weight_label,              matlIndex, patch, gn, 0 );
      new_dw->get( wa_required_variable, d_required_variable_label,   matlIndex, patch, gn, 0 );
      new_dw->get( required_variable_2,  d_required_variable_2_label, matlIndex, patch, gn, 0 );

      new_dw->getModifiable( model_term,     d_modelLabel,    matlIndex, patch );
      new_dw->getModifiable( gas_model_term, d_gasModelLabel, matlIndex, patch );

    }

    for (CellIterator iter=patch->getCellIterator(); !iter.done(); iter++){

      IntVector c = *iter; 
        
    }//end cell loop
  
  }//end patch loop
}


