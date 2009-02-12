// RMCRTRadiationModel.cc-------------------------------------------------------
// Reverse Monte Carlo Ray Tracing Radiation Model interface
// 
// (other comments here) 
//
#include <Packages/Uintah/CCA/Components/Arches/MCRT/ArchesRMCRT/RMCRTnoInterpolation.h>
#include <Packages/Uintah/Core/Grid/Variables/PerPatch.h>
#include <Packages/Uintah/Core/Grid/Variables/CCVariable.h>
#include <Packages/Uintah/Core/Grid/Variables/VarTypes.h>
#include <Packages/Uintah/CCA/Ports/Scheduler.h>
#include <Packages/Uintah/Core/Grid/Task.h>
#include <Packages/Uintah/Core/Grid/Variables/CellIterator.h>
#include <Packages/Uintah/Core/Grid/SimulationState.h>

#include <Packages/Uintah/CCA/Components/Arches/ArchesLabel.h>
#include <Packages/Uintah/CCA/Components/Arches/MCRT/ArchesRMCRT/RMCRTRadiationModel.h>

using namespace Uintah; 
using namespace std;

//---------------------------------------------------------------------------
//  Constructor
//---------------------------------------------------------------------------
RMCRTRadiationModel::RMCRTRadiationModel( const ArchesLabel* label ) : 
d_lab(label) 
{
}

//---------------------------------------------------------------------------
//  Destructor
//---------------------------------------------------------------------------
RMCRTRadiationModel::~RMCRTRadiationModel()
{
}

//---------------------------------------------------------------------------
//  ProblemSetup
//---------------------------------------------------------------------------
void 
RMCRTRadiationModel::problemSetup( const ProblemSpecP& params )
{
  // This will look for a block in the input file called <RMCRTRadiationModel>
  ProblemSpecP db_rad = params->findBlock("RMCRTRadiationModel");
 
  // This looks for a value for <const_num_rays> in the <RMCRTRadiationModel> block 
  db_rad->require("const_num_rays", d_constNumRays);

  // add more inputs here...

}

//---------------------------------------------------------------------------
//  Schedule the solve of the RTE using RMCRT
//---------------------------------------------------------------------------

void 
RMCRTRadiationModel::sched_solve( const LevelP& level, SchedulerP& sched )
{
  const string taskname = "RMCRTRadiationModel::solve"; 
  Task* tsk = scinew Task(taskname, this, &RMCRTRadiationModel::solve); 

  //Variables needed from DW
  Ghost::GhostType  gac = Ghost::AroundCells;
  Ghost::GhostType  gaf = Ghost::AroundFaces;
  Ghost::GhostType  gn = Ghost::None;
  tsk->requires(Task::OldDW, d_lab->d_tempINLabel, gac, 1); // getting temperature w/1 ghost to use only (not to modify it)
  //tsk->requires(Task::OldDW, d_lab->d_absorpINLabel, gac, 1); // getting absorption coef w/1 ghost

  sched->addTask(tsk, level->eachPatch(), d_lab->d_sharedState->allArchesMaterials()); 

}

//---------------------------------------------------------------------------
//  Actually solve the RTE using RMCRT
//---------------------------------------------------------------------------
void 
RMCRTRadiationModel::solve(  const ProcessorGroup* pc, 
                             const PatchSubset* patches,
                             const MaterialSubset*, 
                             DataWarehouse* old_dw, 
                             DataWarehouse* new_dw )

{
  Ghost::GhostType  gac = Ghost::AroundCells;
  Ghost::GhostType  gaf = Ghost::AroundFaces;
  Ghost::GhostType  gn = Ghost::None;
  
  RMCRTnoInterpolation obRMCRT;
  // patch loop 
  for ( int p = 0; p < patches->size(); p++ ) {
    const Patch* patch = patches->get(p); 
    int archIndex = 0; // only one arches material for now
    int matlIndex = 0; 

    // get temperature and absorption coefficient 
    constCCVariable<double> temperature; 
    constCCVariable<double> absorpCoef; 

    old_dw->get( temperature, d_lab->d_tempINLabel, 0, patch, gac, 1 ); 
    //old_dw->get( absorpCoef, d_lab->d_absorpINLabel, 0, patch, gac, 1 );

    cout << "GOING TO CALL STAND ALONE SOLVER!\n"; 
    
    obRMCRT.RMCRTsolver();

  } // end patch loop 

}
