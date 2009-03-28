#include <CCA/Components/SpatialOps/CoalModels/KobayashiSarofimDevol.h>
#include <CCA/Components/SpatialOps/TransportEqns/EqnFactory.h>
#include <CCA/Components/SpatialOps/TransportEqns/EqnFactory.h>
#include <CCA/Components/SpatialOps/TransportEqns/EqnBase.h>
#include <CCA/Components/SpatialOps/TransportEqns/DQMOMEqn.h>

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
KobayashiSarofimDevolBuilder::KobayashiSarofimDevolBuilder( const std::string         & modelName,
                                                            const vector<std::string> & reqLabelNames,
                                                            const Fields              * fieldLabels,
                                                            SimulationStateP          & sharedState,
                                                            int qn ) :
  ModelBuilder( modelName, fieldLabels, reqLabelNames, sharedState, qn )
{}

KobayashiSarofimDevolBuilder::~KobayashiSarofimDevolBuilder(){}

ModelBase*
KobayashiSarofimDevolBuilder::build(){
  return scinew KobayashiSarofimDevol( d_modelName, d_sharedState, d_fieldLabels, d_icLabels, d_quadNode );
}
// End Builder
//---------------------------------------------------------------------------

KobayashiSarofimDevol::KobayashiSarofimDevol( std::string srcName, SimulationStateP& sharedState,
                            const Fields* fieldLabels,
                            vector<std::string> icLabelNames, int qn ) 
: ModelBase(srcName, sharedState, fieldLabels, icLabelNames, qn),d_fieldLabels(fieldLabels)
{
  A1 = 1.0;
  E1 = 1.0;
  A2 = 1.0;
  E2 = 1.0;
  R = 1.0;
}

KobayashiSarofimDevol::~KobayashiSarofimDevol()
{}
//---------------------------------------------------------------------------
// Method: Problem Setup
//---------------------------------------------------------------------------
void 
KobayashiSarofimDevol::problemSetup(const ProblemSpecP& params, int qn)
{
  ProblemSpecP db_dqmom = params->findBlock("DQMOM");
  ProblemSpecP db_Models = db_dqmom->findBlock("Models");
  
  for ( ProblemSpecP db_model = db_Models->findBlock("model"); db_model != 0; db_model = db_Models->findNextBlock("model") ) {
    string modeltype;
    db_model->getAttribute("type",modeltype);
    
    if (modeltype == "KobayashiSarofimDevol") {
      ProblemSpecP db_icvars = db_model->findBlock("ICVars");
      for (ProblemSpecP variable = db_icvars->findBlock("variable"); variable != 0; variable = db_icvars->findNextBlock("variable") ) {
        string label_name;
        string role_name;
        
        variable->getAttribute("label",label_name);
        variable->getAttribute("role",role_name);
        
        /*
        // This way restricts what "roles" the user can specify (less flexible)
        if (role_name == "temperature" || role_name == "coal_mass_fraction") {
          LabelToRoleMap[label_name] = role_name;
        } else {
          throw InvalidValue( "Invalid variable role for Kobayashi Sarofim Devolatilization model!  Must be \"temperature\" or \"coal_mass_fraction\".",__FILE__,__LINE__);
        }
        */

        //This way does not restrict what "roles" the user can specify (more flexible)
        LabelToRoleMap[label_name] = role_name;
      }
    }
  
  }

}
//---------------------------------------------------------------------------
// Method: Schedule the calculation of the Model 
//---------------------------------------------------------------------------
void 
KobayashiSarofimDevol::sched_computeModel( const LevelP& level, SchedulerP& sched, int timeSubStep )
{
  std::string taskname = "KobayashiSarofimDevol::computeModel";
  Task* tsk = scinew Task(taskname, this, &KobayashiSarofimDevol::computeModel);

  d_timeSubStep = timeSubStep; 

  if (d_timeSubStep == 0 && !d_labelSchedInit) {
    // Every model term needs to set this flag after the varLabel is computed. 
    // transportEqn.cleanUp should reinitialize this flag at the end of the time step. 
    d_labelSchedInit = true;

    tsk->computes(d_modelLabel);
  } else {
    tsk->modifies(d_modelLabel); 
  }

  EqnFactory& eqn_factory = EqnFactory::self();
  DQMOMEqnFactory& dqmom_eqn_factory = DQMOMEqnFactory::self();

  // For each required variable, determine if it plays the role of temperature or mass fraction;
  //  if it plays the role of mass fraction, then look for it in equation factories
  for (vector<std::string>::iterator iter = d_icLabels.begin(); 
       iter != d_icLabels.end(); iter++) { 
    
    // put labels into a label vector in the "correct" order for the model
    // orderedLabels[1] = temperature
    // orderedLabels[2] = coal mass fraction
    map<string, string>::iterator iMap = LabelToRoleMap.find(*iter);
    if (iMap->second == "temperature") {
      // automatically use Fields.cc's temperature label if role="temperature"
      orderedLabels[1] = d_fieldLabels->propLabels.temperature;
    } else if (iMap->second == "coal_mass_fraction") {

      // I added the find_scalar_eqn() function to the equation factories
      // so that we could determine what type each required variable is (normal scalar or dqmom scalar)

      // Only require() variables found in equation factories (temperature will be require()'d later)
      const VarLabel* current_label;
      if ( eqn_factory.find_scalar_eqn(*iter) ) {
        EqnBase& current_eqn = eqn_factory.retrieve_scalar_eqn(*iter);
        current_label = current_eqn.getTransportEqnLabel();
        tsk->requires(Task::OldDW, current_label, Ghost::AroundCells, 1);
      } else if (dqmom_eqn_factory.find_scalar_eqn(*iter) ) {
        EqnBase& current_eqn = dqmom_eqn_factory.retrieve_scalar_eqn(*iter);
        current_label = current_eqn.getTransportEqnLabel();
        tsk->requires(Task::OldDW, current_label, Ghost::AroundCells, 1);
      } else {
        throw InvalidValue( "Invalid variable given in <variable> tag for Kobayashi Sarofim devolatilization model: could not find given "
                            "coal mass fraction variable \"" + *iter + "\" in EqnFactory or in DQMOMEqnFactory!",__FILE__,__LINE__);
      }
      orderedLabels[2] = current_label;
    } //else... we don't need that variable!!!
  
  }

  tsk->requires(Task::OldDW, d_fieldLabels->propLabels.temperature, Ghost::AroundCells, 1);

  sched->addTask(tsk, level->eachPatch(), d_sharedState->allSpatialOpsMaterials()); 

}
//---------------------------------------------------------------------------
// Method: Actually compute the source term 
//---------------------------------------------------------------------------
void
KobayashiSarofimDevol::computeModel( const ProcessorGroup * pc, 
                                     const PatchSubset    * patches, 
                                     const MaterialSubset * matls, 
                                     DataWarehouse        * old_dw, 
                                     DataWarehouse        * new_dw )
{
  for( int p=0; p < patches->size(); p++ ) {  // Patch loop

    //Ghost::GhostType  gaf = Ghost::AroundFaces;
    Ghost::GhostType  gac = Ghost::AroundCells;
    //Ghost::GhostType  gn  = Ghost::None;

    const Patch* patch = patches->get(p);
    int matlIndex = 0;

    CCVariable<double> model; 
    if (new_dw->exists( d_modelLabel, matlIndex, patch )){
      new_dw->getModifiable( model, d_modelLabel, matlIndex, patch ); 
    } else {
      new_dw->allocateAndPut( model, d_modelLabel, matlIndex, patch );
      model.initialize(0.0);
    }

    constCCVariable<double> temperature;
    constCCVariable<double> alphac;
    new_dw->get( temperature, orderedLabels[1], matlIndex, patch, gac, 0 );
    new_dw->get( alphac, orderedLabels[2], matlIndex, patch, gac, 0 );
    
    for (CellIterator iter=patch->getCellIterator__New(); !iter.done(); iter++){
      IntVector c = *iter; 
      double k1 = A1*exp(-E1/(R*temperature[c]));
      double k2 = A2*exp(-E2/(R*temperature[c]));

      model[c] = -(k1+k2)*alphac[c];
    }
  }
}
