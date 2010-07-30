// A class to hold some fields for getting us started

#include <Core/Grid/Variables/VarLabel.h>
#include <Core/Grid/Variables/PerPatch.h>
#include <Core/Grid/Variables/CCVariable.h>
#include <Core/Grid/Variables/SFCXVariable.h>
#include <Core/Grid/Variables/SFCYVariable.h>
#include <Core/Grid/Variables/SFCZVariable.h>
#include <Core/Grid/Variables/VarTypes.h>
#include <Core/Grid/SimulationState.h>
#include <CCA/Components/SpatialOps/Fields.h>
#include <Core/Grid/SimulationStateP.h>

//===========================================================================

using namespace Uintah;

Fields::Fields()
{

  // --- Properties ---    
  std::string varname = "lambda";   
  propLabels.lambda = VarLabel::create(varname, 
            CCVariable<double>::getTypeDescription());
  varname = "density"; 
  propLabels.density = VarLabel::create(varname, 
            CCVariable<double>::getTypeDescription());

  varname = "temperature";
  propLabels.temperature = VarLabel::create(varname,
            CCVariable<double>::getTypeDescription());

  // --- Velocities ---
  varname = "uVelocity"; 
  velocityLabels.uVelocity = VarLabel::create(varname, 
            SFCXVariable<double>::getTypeDescription());
  varname = "vVelocity"; 
  velocityLabels.vVelocity = VarLabel::create(varname, 
            SFCYVariable<double>::getTypeDescription());
  varname = "wVelocity"; 
  velocityLabels.wVelocity = VarLabel::create(varname, 
            SFCZVariable<double>::getTypeDescription());

  varname = "ccVelocity"; 
  velocityLabels.ccVelocity = VarLabel::create(varname, 
            CCVariable<Vector>::getTypeDescription());

}

Fields::~Fields()
{
  VarLabel::destroy(propLabels.lambda); 
  VarLabel::destroy(propLabels.density); 
}
//---------------------------------------------------------------------------
// Method: Set Shared State
//---------------------------------------------------------------------------
void Fields::setSharedState(SimulationStateP& sharedState)
{
    d_sharedState = sharedState;
}

//---------------------------------------------------------------------------
// Method: Schedule the copy of old data to a newly allocated variable
//---------------------------------------------------------------------------
void 
Fields::schedCopyOldToNew( const LevelP& level, SchedulerP& sched )
{
  string taskname = "Fields::CopyOldToNew";
  Task* tsk = scinew Task(taskname, this, &Fields::CopyOldToNew);

  //--New
  tsk->computes(propLabels.lambda); 
  tsk->computes(propLabels.density); 
  tsk->computes(propLabels.temperature);
  tsk->computes(velocityLabels.uVelocity); 
  tsk->computes(velocityLabels.vVelocity); 
  tsk->computes(velocityLabels.wVelocity);
  tsk->computes(velocityLabels.ccVelocity); 

  //--Old
  tsk->requires(Task::OldDW, propLabels.lambda, Ghost::None, 0);
  tsk->requires(Task::OldDW, propLabels.density, Ghost::None, 0);  
  tsk->requires(Task::OldDW, propLabels.temperature, Ghost::None, 0);
  tsk->requires(Task::OldDW, velocityLabels.uVelocity, Ghost::None, 0);  
  tsk->requires(Task::OldDW, velocityLabels.vVelocity, Ghost::None, 0);  
  tsk->requires(Task::OldDW, velocityLabels.wVelocity, Ghost::None, 0);
  tsk->requires(Task::OldDW, velocityLabels.ccVelocity, Ghost::None, 0);   

  sched->addTask(tsk, level->eachPatch(), d_sharedState->allSpatialOpsMaterials());
}
//---------------------------------------------------------------------------
// Method: Copy old data into a newly allocated new data spot
//---------------------------------------------------------------------------
void Fields::CopyOldToNew( const ProcessorGroup* pc, 
                           const PatchSubset* patches, 
                           const MaterialSubset* matls, 
                           DataWarehouse* old_dw, 
                           DataWarehouse* new_dw )
{
   //patch loop
  for (int p=0; p < patches->size(); p++){

    //Ghost::GhostType  gaf = Ghost::AroundFaces;
    //Ghost::GhostType  gac = Ghost::AroundCells;
    Ghost::GhostType  gn  = Ghost::None;

    const Patch* patch = patches->get(p);
    int matlIndex = 0;

    CCVariable<double> lambda; 
    CCVariable<double> density;
    CCVariable<double> temperature;
    SFCXVariable<double> uVelocity; 
    SFCYVariable<double> vVelocity; 
    SFCZVariable<double> wVelocity; 
    CCVariable<Vector> ccVelocity; 

    constCCVariable<double> old_lambda;
    constCCVariable<double> old_density;  
    constCCVariable<double> old_temperature;
    constSFCXVariable<double> old_uVelocity; 
    constSFCYVariable<double> old_vVelocity; 
    constSFCZVariable<double> old_wVelocity; 
    constCCVariable<Vector> old_ccVelocity; 

    new_dw->allocateAndPut( lambda, propLabels.lambda, matlIndex, patch ); 
    old_dw->get(old_lambda, propLabels.lambda, matlIndex, patch, gn, 0); 
    lambda.copy(old_lambda); 

    new_dw->allocateAndPut( density, propLabels.density, matlIndex, patch ); 
    old_dw->get(old_density, propLabels.density, matlIndex, patch, gn, 0); 
    density.copy(old_density); 

    new_dw->allocateAndPut( temperature, propLabels.temperature, matlIndex, patch );
    old_dw->get(old_temperature, propLabels.temperature, matlIndex, patch, gn, 0);
    temperature.copy(old_temperature);

    new_dw->allocateAndPut( uVelocity, velocityLabels.uVelocity, matlIndex, patch ); 
    old_dw->get(old_uVelocity, velocityLabels.uVelocity, matlIndex, patch, gn, 0); 
    uVelocity.copy(old_uVelocity); 
#ifdef YDIM
    new_dw->allocateAndPut( vVelocity, velocityLabels.vVelocity, matlIndex, patch ); 
    old_dw->get(old_vVelocity, velocityLabels.vVelocity, matlIndex, patch, gn, 0); 
    vVelocity.copy(old_vVelocity); 
#endif
#ifdef ZDIM
    new_dw->allocateAndPut( wVelocity, velocityLabels.wVelocity, matlIndex, patch ); 
    old_dw->get(old_wVelocity, velocityLabels.wVelocity, matlIndex, patch, gn, 0); 
    wVelocity.copy(old_wVelocity); 
#endif
    new_dw->allocateAndPut( ccVelocity, velocityLabels.ccVelocity, matlIndex, patch ); 
    old_dw->get(old_ccVelocity, velocityLabels.ccVelocity, matlIndex, patch, gn, 0); 
    ccVelocity.copy(old_ccVelocity); 
  } 
}
//---------------------------------------------------------------------------
// Method: Compute velocities for DQMOM transport
//---------------------------------------------------------------------------


