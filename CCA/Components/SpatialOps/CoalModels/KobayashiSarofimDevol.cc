#include <Core/ProblemSpec/ProblemSpec.h>
#include <CCA/Ports/Scheduler.h>
#include <Core/Grid/SimulationState.h>
#include <Core/Grid/Variables/VarTypes.h>
#include <Core/Grid/Variables/CCVariable.h>
#include <CCA/Components/SpatialOps/CoalModels/KobayashiSarofimDevol.h>

//===========================================================================

using namespace std;
using namespace Uintah; 

//---------------------------------------------------------------------------
// Builder:
KobayashiSarofimDevolBuilder::KobayashiSarofimDevolBuilder(std::string srcName, 
                                         vector<std::string> icLabelNames, 
                                         SimulationStateP& sharedState, int qn)
: ModelBuilder(srcName, icLabelNames, sharedState, qn)
{}

KobayashiSarofimDevolBuilder::~KobayashiSarofimDevolBuilder(){}

ModelBase*
KobayashiSarofimDevolBuilder::build(){
  return scinew KobayashiSarofimDevol( d_modelName, d_sharedState, d_icLabels, 
                              d_quadNode );
}
// End Builder
//---------------------------------------------------------------------------

KobayashiSarofimDevol::KobayashiSarofimDevol( std::string srcName, SimulationStateP& sharedState,
                            const Fields* fieldLabels,
                            vector<std::string> icLabelNames, int qn ) 
: ModelBase(srcName, sharedState, icLabelNames, qn),d_fieldLabels(fieldLabels)
{}

KobayashiSarofimDevol::~KobayashiSarofimDevol()
{}
//---------------------------------------------------------------------------
// Method: Problem Setup
//---------------------------------------------------------------------------
void 
KobayashiSarofimDevol::problemSetup(const ProblemSpecP& inputdb, int qn)
{

  ProblemSpecP db = inputdb; 

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

  for (vector<std::string>::iterator iter = d_icLabels.begin(); 
       iter != d_icLabels.end(); iter++) { 
    // HERE I WOULD REQUIRE ANY VARIABLES NEEDED TO COMPUTE THE MODEL
    tsk->requires(Task::OldDW, d_fieldLabels->d_temperature, Ghost::AroundCells, 1);
  }

  sched->addTask(tsk, level->eachPatch(), d_sharedState->allSpatialOpsMaterials()); 

}
//---------------------------------------------------------------------------
// Method: Actually compute the source term 
//---------------------------------------------------------------------------
void
KobayashiSarofimDevol::computeModel( const ProcessorGroup* pc, 
                   const PatchSubset* patches, 
                   const MaterialSubset* matls, 
                   DataWarehouse* old_dw, 
                   DataWarehouse* new_dw )
{
  //patch loop
  for (int p=0; p < patches->size(); p++){

    Ghost::GhostType  gaf = Ghost::AroundFaces;
    Ghost::GhostType  gac = Ghost::AroundCells;
    Ghost::GhostType  gn  = Ghost::None;

    const Patch* patch = patches->get(p);
    int matlIndex = 0;

    CCVariable<double> model; 
    if (new_dw->exists( d_modelLabel, matlIndex, patch )){
      new_dw->getModifiable( model, d_modelLabel, matlIndex, patch ); 
    } else {
      new_dw->allocateAndPut( model, d_modelLabel, matlIndex, patch );
      model.initialize(0.0);
    }

    const CCVariable<double> temperature;
    const CCVariable<double> alphac;
    new_dw->get(temperature, d_fieldLabels->d_temperature, matlIndex, patch);
    new_dw->get(alphac, "coal_mass_fraction", matlIndex, patch);
    
    for (vector<std::string>::iterator iter = d_icLabels.begin(); 
         iter != d_icLabels.end(); iter++) { 
      // how to differentiate between different internal coordinates?
    }

    for (CellIterator iter=patch->getCellIterator__New(); !iter.done(); iter++){
      IntVector c = *iter; 
      k1 = A1*exp(-E1/(R*temperature[c]));
      k2 = A2*exp(-E2/(R*temperature[c]));

      model[c] = -(k1+k2)*alphac;//change this to the function 
    }
  }
}
