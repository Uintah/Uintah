/// RMCRTRadiationModel.cc-------------------------------------------------------
/// Reverse Monte Carlo Ray Tracing Radiation Model interface
/// 
/// @author Xiaojing Sun ( Paula ) and Jeremy
/// @date Feb 20, 2009.
///
/*

The MIT License

Copyright (c) 1997-2009 Center for the Simulation of Accidental Fires and 
Explosions (CSAFE), and  Scientific Computing and Imaging Institute (SCI), 
University of Utah.

License for the specific language governing rights and limitations under
Permission is hereby granted, free of charge, to any person obtaining a 
copy of this software and associated documentation files (the "Software"),
to deal in the Software without restriction, including without limitation 
the rights to use, copy, modify, merge, publish, distribute, sublicense, 
and/or sell copies of the Software, and to permit persons to whom the 
Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included 
in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS 
OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, 
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL 
THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER 
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING 
FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER 
DEALINGS IN THE SOFTWARE.

*/
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

  string prop_model; // Do i need this?

  // ask for number of rays
  db_rad->require("const_num_rays", d_constNumRays);

  // if using russian roulette
  if ( db_rad->findBlock("russian_roulette")){
    d_rr = true; 
    db_rad->getWithDefault("rr_threshold", d_StopLowerBound, 1.e-4);
  }
  
  // property model set up
  // radcoef requires opl 
  db_rad->require("opl",d_opl);
  db_rad->require("property_model", prop_model);
  
  if (prop_model == "radcoef"){  // have this in c++
    d_radcal     = false;
    d_wsgg       = false;
    d_ambda      = 1;
    d_planckmean = false;
    d_patchmean  = false;
    d_fssk       = false;
    d_fsck       = false;
  }

  if (prop_model == "patchmean"){ 
    cout << "WARNING! Serial and parallel results may deviate for this model" << endl;
    d_radcal     = true;
    d_wsgg       = false;
    d_ambda      = 6;
    d_planckmean = false;
    d_patchmean  = true;
    d_fssk       = false;
    d_fsck       = false;    
  }

  if (prop_model == "wsggm"){ // have this in c++
    throw InternalError("WSGG radiation model does not run in parallel and has been disabled", __FILE__, __LINE__);
    d_radcal       = false;
    d_wsgg         = true;
    d_ambda        = 4;
    d_planckmean   = false;
    d_patchmean    = false;
    d_fssk         = false;
    d_fsck         = false;    
  }


  // fssk and fsck needs the stretching factor a for different reference states
  // however 'a' is calculated every update
  // the reference state is also updated every radiation time
  // but we need an initial reference state
  if (prop_model == "fssk"){
    db_rad->require("reference_temperature", d_refTemp);
    // ref_comp is a vector of all gas mixtures composition
    // how to decide the size of ref_comp?
    db_rad->require("reference_composition", d_refComp);
    db_rad->require("scaling_function",d_uScale);    
    // db_rad->require("a_fssk", d_afssk);

    
    d_radcal       = false;
    d_wsgg         = false;
    //   d_ambda        = 4;
    d_planckmean   = false;
    d_patchmean    = false;
    d_fssk         = false;
    d_fsck         = false;    
  }

  if (prop_model == "fsck"){
    
    db_rad->require("reference_temperature", d_refTemp);
    // ref_comp is a vector of all gas mixtures composition
    // how to decide the size of ref_comp?
    db_rad->require("reference_composition", d_refComp);
    //  db_rad->require("a_fsck", d_afsck);

    d_radcal       = false;
    d_wsgg         = false;
    //   d_ambda        = 4;
    d_planckmean   = false;
    d_patchmean    = false;
    d_fssk         = false;
    d_fsck         = false;        

  }
  
  
  
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
  tsk->requires(Task::OldDW, d_lab->d_absorpINLabel, gac, 1); // getting absorption coef w/1 ghost

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
    constCCVariable<double> scatterCoeff;
    // emission coefficient on boundaries are CCVariable? or SFXVariable?
    // absorption coefficient on boundaries? define as what type?
    // rs, rd?
    
    // get emission coeff on boundaries
    // get X Y Z
    // get cellType wall or flow
    // X node, Y node, Z node numbers    
    //  CellInformation* cellinfo, cellinfo->x, cellinfo->y..
    //  ArchesVariables* vars, vars->temperature
    //  ArchesConstVariables* constvars) constvars->cellType
    
    old_dw->get( temperature, d_lab->d_tempINLabel, 0, patch, gac, 1 ); 
    old_dw->get( absorpCoef, d_lab->d_absorpINLabel, 0, patch, gac, 1 );


    
    // IntVector currCell(currI, currJ, currK);
    // temperature[currCell]??
    // where does the currCell starts? for b.c. cells and ghost cells
    
    cout << "GOING TO CALL STAND ALONE SOLVER!\n"; 
    
    obRMCRT.RMCRTsolver();

  } // end patch loop 

}


