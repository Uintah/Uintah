#include <sci_defs/malloc_defs.h>

#include <Packages/Uintah/CCA/Components/SimulationController/AMRSimulationController.h>
#include <Core/Geometry/IntVector.h>
#include <Core/Geometry/Vector.h>
#include <Core/Math/MiscMath.h>
#include <Core/Containers/Array3.h>
#include <Core/Thread/Time.h>
#include <Core/OS/ProcessInfo.h>
#include <Packages/Uintah/Core/Exceptions/ProblemSetupException.h>
#include <Packages/Uintah/Core/Grid/Grid.h>
#include <Packages/Uintah/Core/Grid/Level.h>
#include <Packages/Uintah/Core/Grid/SimulationTime.h>
#include <Packages/Uintah/Core/Grid/Patch.h>
#include <Packages/Uintah/Core/Grid/Variables/ReductionVariable.h>
#include <Packages/Uintah/Core/Grid/Variables/SoleVariable.h>
#include <Packages/Uintah/Core/Grid/Variables/PerPatch.h>
#include <Packages/Uintah/Core/Grid/Variables/VarLabel.h>
#include <Packages/Uintah/Core/Grid/SimulationState.h>
#include <Packages/Uintah/Core/Grid/Variables/VarLabelMatl.h>
#include <Packages/Uintah/CCA/Ports/SimulationInterface.h>
#include <Packages/Uintah/CCA/Ports/DataWarehouse.h>
#include <Packages/Uintah/CCA/Ports/Regridder.h>
#include <Packages/Uintah/CCA/Components/Regridder/PerPatchVars.h>
#include <Packages/Uintah/CCA/Ports/Output.h>
#include <Packages/Uintah/Core/ProblemSpec/ProblemSpec.h>
#include <Packages/Uintah/CCA/Ports/ProblemSpecInterface.h>
#include <Packages/Uintah/Core/ProblemSpec/ProblemSpecP.h>
#include <Packages/Uintah/CCA/Ports/Scheduler.h>
#include <Packages/Uintah/CCA/Components/PatchCombiner/PatchCombiner.h>
#include <Packages/Uintah/CCA/Components/PatchCombiner/UdaReducer.h>
#include <Packages/Uintah/CCA/Ports/LoadBalancer.h>
#include <Packages/Uintah/Core/Parallel/ProcessorGroup.h>
#include <Packages/Uintah/Core/Grid/Variables/VarTypes.h>
#include <TauProfilerForSCIRun.h>
#include <iostream>
#include <iomanip>

using std::cerr;
using std::cout;

using namespace SCIRun;
using namespace Uintah;

static DebugStream amrout("AMR", false);
static DebugStream dbg("AMRSimulationController", false);

AMRSimulationController::AMRSimulationController(const ProcessorGroup* myworld,
                                                 bool doAMR, ProblemSpecP pspec) :
  SimulationController(myworld, doAMR, pspec)
{
}

AMRSimulationController::~AMRSimulationController()
{
}

void AMRSimulationController::run()
{
   bool log_dw_mem=false;
#ifndef DISABLE_SCI_MALLOC
   ProblemSpecP debug = d_ups->findBlock("debug");
   if(debug){
     ProblemSpecP log_mem = debug->findBlock("logmemory");
     if(log_mem){
       log_dw_mem=true;
     }
   }
#endif

   // sets up sharedState, timeinfo, output
   preGridSetup();

   d_sharedState->d_simTime = d_timeinfo;

   // create grid
   GridP currentGrid = gridSetup();

   // set up scheduler, lb, sim, regridder, and finalize sharedState
   postGridSetup(currentGrid);

   calcStartTime();

   d_scheduler->initialize(1, 1);
   d_scheduler->advanceDataWarehouse(currentGrid);
    
   double t;

    
   if (d_restarting){
     restartSetup(currentGrid, t);
   }
   if (d_combinePatches) {
     // combine patches and reduce uda need the same things here
     Dir combineFromDir(d_fromDir);
     d_output->combinePatchSetup(combineFromDir);

     // somewhat of a hack, but the patch combiner specifies exact delt's
     // and should not use a delt factor.
     d_timeinfo->delt_factor = 1;
     d_timeinfo->delt_min = 0;
     if (d_reduceUda){
       d_timeinfo->maxTime = static_cast<UdaReducer*>(d_sim)->getMaxTime();
     }else{
       d_timeinfo->maxTime = static_cast<PatchCombiner*>(d_sim)->getMaxTime();
     }
     cout << " MaxTime: " << d_timeinfo->maxTime << endl;
     d_timeinfo->delt_max = d_timeinfo->maxTime;
   }

   // setup, compile, and run the taskgraph for the initialization timestep
   doInitialTimestep(currentGrid, t);

   setStartSimTime(t);
   initSimulationStatsVars();

   // this section is for "automatically" generating all the levels we can
   // so far, based on the existence of flagged cells, but limited to
   // the number of levels the regridder can handle .
   // Only do if not restarting

   if (d_doAMR && !d_restarting && d_regridder->isAdaptive()){
     while (currentGrid->numLevels() < d_regridder->maxLevels() &&
            d_regridder->flaggedCellsOnFinestLevel(currentGrid, d_scheduler)) {
       if (!doInitialTimestepRegridding(currentGrid)) {
         break;
       }
     }
   }


   ////////////////////////////////////////////////////////////////////////////
   // The main time loop; here the specified problem is actually getting solved
   
   bool first=true;
   int  iterations = 0;
   double delt = 0;

   // if we end the simulation for a timestep, decide whether to march max_iterations
   // or to end at a certain timestep
   int max_iterations = d_timeinfo->max_iterations;
   if (d_timeinfo->maxTimestep - d_sharedState->getCurrentTopLevelTimeStep() < max_iterations) {
     max_iterations = d_timeinfo->maxTimestep - d_sharedState->getCurrentTopLevelTimeStep();
   }
   while( t < d_timeinfo->maxTime && iterations < max_iterations) {

     if (d_doAMR && d_regridder->needsToReGrid() && !first) {
       doRegridding(currentGrid);
     }

     // Compute number of dataWarehouses - multiplies by the time refinement
     // ratio for each level you increase
     int totalFine=1;
     for(int i=1;i<currentGrid->numLevels();i++) {
       totalFine *= d_sharedState->timeRefinementRatio();
     }
     
     if (iterations != 0) {
       // will be set in SimCont::postGridSetup
       d_sharedState->d_prev_delt = delt;
     }
     iterations ++;
 
     // get delt and adjust it
     delt_vartype delt_var;
     DataWarehouse* newDW = d_scheduler->getLastDW();
     newDW->get(delt_var, d_sharedState->get_delt_label());

     delt = delt_var;
     
     // delt adjusted based on timeinfo parameters
     adjustDelT(delt, d_sharedState->d_prev_delt, d_sharedState->getCurrentTopLevelTimeStep(), t);
     newDW->override(delt_vartype(delt), d_sharedState->get_delt_label());

     // printSimulationStats( d_sharedState, delt, t );

     if(log_dw_mem){
       // Remember, this isn't logged if DISABLE_SCI_MALLOC is set
       // (So usually in optimized mode this will not be run.)
       d_scheduler->logMemoryUse();
       ostringstream fn;
       fn << "alloc." << setw(5) << setfill('0') << d_myworld->myrank() << ".out";
       string filename(fn.str());
#ifndef _WIN32
       DumpAllocator(DefaultAllocator(), filename.c_str());
#endif
     }

     // For material addition.  Once a material is added, need to
     // reset the flag, but can't do it til the subsequent timestep
     static int sub_step=0;

     if(d_sharedState->needAddMaterial() != 0){
       if(sub_step==1){
         d_sharedState->resetNeedAddMaterial();
         sub_step = -1;
       }
       sub_step++;
     }

     if(d_sharedState->needAddMaterial() != 0){
       d_sim->addMaterial(d_ups, currentGrid, d_sharedState);
       d_sharedState->finalizeMaterials();
       d_scheduler->initialize();
       for (int i = 0; i < currentGrid->numLevels(); i++) {
         d_sim->scheduleInitializeAddedMaterial(currentGrid->getLevel(i), d_scheduler);
         if (d_doAMR && i > 0){
           d_sim->scheduleRefineInterface(currentGrid->getLevel(i), d_scheduler, false, true);
         }
       }
       d_scheduler->compile();
       d_scheduler->get_dw(1)->setScrubbing(DataWarehouse::ScrubNone);
       d_scheduler->execute();
     }

     // After one step (either timestep or initialization) and correction
     // the delta we can finally, finalize our old timestep, eg. 
     // finalize and advance the Datawarehouse
     d_scheduler->advanceDataWarehouse(currentGrid);

     // Put the current time into the shared state so other components
     // can access it.  Also increment (by one) the current time step
     // number so components can tell what timestep they are on. 
     d_sharedState->setElapsedTime(t);
     d_sharedState->incrementCurrentTopLevelTimeStep();

     // Each component has their own init_delt specified.  On a switch
     // from one component to the next, we need to adjust the delt to
     // that specified in the input file.  To detect the switch of components,
     // we compare the old_init_delt before the needRecompile() to the 
     // new_init_delt after the needRecompile().  

     double old_init_delt = d_timeinfo->max_initial_delt;
     double new_init_delt = 0.;

     if(needRecompile(t, delt, currentGrid) || first ){
       new_init_delt = d_timeinfo->max_initial_delt;
       if (new_init_delt != old_init_delt) {
         // writes to the DW in the next section below
         delt = new_init_delt;
       }
       // this is the only way to get a new grid from UdaReducer
       if (d_reduceUda) currentGrid = static_cast<UdaReducer*>(d_sim)->getGrid();
       first=false;
       recompile(t, delt, currentGrid, totalFine);
     }
     else {
       if (d_output){
         // This is not correct if we have switched to a different
         // component, since the delt will be wrong 
         d_output->finalizeTimestep(t, delt, currentGrid, d_scheduler, 0);
       }
     }

     // adjust the delt for each level and store it in all applicable dws.
     double delt_fine = delt;
     int skip=totalFine;
     for(int i=0;i<currentGrid->numLevels();i++){
       const Level* level = currentGrid->getLevel(i).get_rep();
       if(d_doAMR && i != 0){
	 delt_fine /= d_sharedState->timeRefinementRatio();
	 skip /= d_sharedState->timeRefinementRatio();
       }
       for(int idw=0;idw<totalFine;idw+=skip){
	 DataWarehouse* dw = d_scheduler->get_dw(idw);
	 dw->override(delt_vartype(delt_fine), d_sharedState->get_delt_label(),
		      level);
       }
     }

     calcWallTime();

     // use timestep-1 since we need to do the printouts after the incrementing
     printSimulationStats(d_sharedState->getCurrentTopLevelTimeStep()-1,delt,t);
     // Execute the current timestep, restarting if necessary
     executeTimestep(t, delt, currentGrid, totalFine);

     if(d_output){
       d_output->executedTimestep(delt, currentGrid);
     }

     t += delt;
     TAU_DB_DUMP();
   }

   d_ups->releaseDocument();
}

//______________________________________________________________________
void AMRSimulationController::subCycleCompile(GridP& grid, int startDW, int dwStride, int step, int numLevel)
{
  //amrout << "Start AMRSimulationController::subCycleCompile, level=" << numLevel << '\n';
  // We are on (the fine) level numLevel
  LevelP fineLevel = grid->getLevel(numLevel);
  LevelP coarseLevel;
  int coarseStartDW;
  int coarseDWStride;
  int numSteps = d_sharedState->timeRefinementRatio(); 
  if (numLevel > 0) {
    coarseLevel = grid->getLevel(numLevel-1);
    coarseDWStride = dwStride * numSteps;
    coarseStartDW = (startDW/coarseDWStride)*coarseDWStride;
  }
  else {
    coarseDWStride = dwStride;
    coarseStartDW = startDW;
  }

  ASSERT(dwStride > 0 && numLevel < grid->numLevels())
  d_scheduler->clearMappings();
  d_scheduler->mapDataWarehouse(Task::OldDW, startDW);
  d_scheduler->mapDataWarehouse(Task::NewDW, startDW+dwStride);
  d_scheduler->mapDataWarehouse(Task::CoarseOldDW, coarseStartDW);
  d_scheduler->mapDataWarehouse(Task::CoarseNewDW, coarseStartDW+coarseDWStride);
  
  d_sim->scheduleTimeAdvance(fineLevel, d_scheduler);

  if(numLevel+1 < grid->numLevels()){
    int newStride = dwStride/numSteps;
    for(int substep=0;substep < numSteps;substep++){
      subCycleCompile(grid, startDW+substep*newStride, newStride, substep, numLevel+1);
    }
    // Coarsen and then refine_CFI at the end of the W-cycle
    d_scheduler->clearMappings();
    d_scheduler->mapDataWarehouse(Task::OldDW, 0);
    d_scheduler->mapDataWarehouse(Task::NewDW, startDW+dwStride);
    d_scheduler->mapDataWarehouse(Task::CoarseOldDW, startDW);
    d_scheduler->mapDataWarehouse(Task::CoarseNewDW, startDW+dwStride);
    d_sim->scheduleCoarsen(fineLevel, d_scheduler);
  }

  d_scheduler->clearMappings();
  d_scheduler->mapDataWarehouse(Task::OldDW, startDW);
  d_scheduler->mapDataWarehouse(Task::NewDW, startDW+dwStride);
  d_scheduler->mapDataWarehouse(Task::CoarseOldDW, coarseStartDW);
  d_scheduler->mapDataWarehouse(Task::CoarseNewDW, coarseStartDW+coarseDWStride);
  d_sim->scheduleFinalizeTimestep(fineLevel, d_scheduler);
  // do refineInterface after the freshest data we can get; after the finer
  // level's coarsen completes
  // do all the levels at this point in time as well, so all the coarsens go in order,
  // and then the refineInterfaces
  if (d_doAMR && (step < numSteps -1 || numLevel == 0)) {
    
    for (int i = fineLevel->getIndex(); i < fineLevel->getGrid()->numLevels(); i++) {
      if (i == 0)
        continue;
      if (i == fineLevel->getIndex() && numLevel != 0) {
        d_scheduler->mapDataWarehouse(Task::CoarseOldDW, coarseStartDW);
        d_scheduler->mapDataWarehouse(Task::CoarseNewDW, coarseStartDW+coarseDWStride);
        d_sim->scheduleRefineInterface(fineLevel, d_scheduler, true, true);
      }
      else {
        // look in the NewDW all the way down
        d_scheduler->mapDataWarehouse(Task::CoarseOldDW, 0);
        d_scheduler->mapDataWarehouse(Task::CoarseNewDW, startDW+dwStride);
        d_sim->scheduleRefineInterface(fineLevel->getGrid()->getLevel(i), d_scheduler, false, true);
      }
    }
  }
}

void AMRSimulationController::subCycleExecute(GridP& grid, int startDW, int dwStride, int levelNum, bool rootCycle)
{
  // there are 2n+1 taskgraphs, n for the basic timestep, n for intermediate 
  // timestep work, and 1 for the errorEstimate and stableTimestep, where n
  // is the number of levels.
  
  //amrout << "Start AMRSimulationController::subCycleExecute, level=" << numLevel << '\n';
  // We are on (the fine) level numLevel
  int numSteps;
  if (levelNum == 0)
    numSteps = 1;
  else {
    numSteps = d_sharedState->timeRefinementRatio();
  }
  
  int newDWStride = dwStride/numSteps;

  int curDW = startDW;
  for(int step=0;step < numSteps;step++){
    if (step > 0)
      curDW += newDWStride; // can't increment at the end, or the FINAL tg for L0 will use the wrong DWs

    d_scheduler->clearMappings();
    d_scheduler->mapDataWarehouse(Task::OldDW, curDW);
    d_scheduler->mapDataWarehouse(Task::NewDW, curDW+newDWStride);
    d_scheduler->mapDataWarehouse(Task::CoarseOldDW, startDW);
    d_scheduler->mapDataWarehouse(Task::CoarseNewDW, startDW+dwStride);

    // we really only need to pass in whether the current DW is mapped to 0
    // or not
    // TODO - fix inter-Taskgraph scrubbing
    d_scheduler->get_dw(curDW)->setScrubbing(DataWarehouse::ScrubNone);
    d_scheduler->get_dw(curDW+newDWStride)->setScrubbing(DataWarehouse::ScrubNone);
    d_scheduler->get_dw(startDW)->setScrubbing(DataWarehouse::ScrubNone);
    d_scheduler->get_dw(startDW+dwStride)->setScrubbing(DataWarehouse::ScrubNone);

    if (dbg.active())
      dbg << d_myworld->myrank() << "   Executing TG on level " << levelNum << " with old DW " 
          << curDW << " = " << d_scheduler->get_dw(curDW)->getID() << " and new " 
          << curDW+newDWStride << " = " << d_scheduler->get_dw(curDW+newDWStride)->getID() 
          << "CO-DW: " << startDW << " CNDW " << startDW+dwStride << endl;

    
    // we need to unfinalize because execute finalizes all new DWs, and we need to write into them still
    // (even if we finalized only the NewDW in execute, we will still need to write into that DW)
    d_scheduler->get_dw(curDW+newDWStride)->unfinalize();
    d_scheduler->execute(levelNum, curDW);

    if(levelNum+1 < grid->numLevels()){
      ASSERT(newDWStride > 0);
      subCycleExecute(grid, curDW, newDWStride, levelNum+1, false);
    }

    if (d_doAMR && grid->numLevels() > 1 && (step < numSteps-1 || levelNum == 0)) {
      // Since the execute of the intermediate is time-based,
      // execute the intermediate TG relevant to this level, if we are in the 
      // middle of the subcycle or at the end of level 0.
      // the end of the cycle will be taken care of by the parent level sybcycle
      d_scheduler->clearMappings();
      d_scheduler->mapDataWarehouse(Task::OldDW, curDW);
      d_scheduler->mapDataWarehouse(Task::NewDW, curDW+newDWStride);
      d_scheduler->mapDataWarehouse(Task::CoarseOldDW, startDW);
      d_scheduler->mapDataWarehouse(Task::CoarseNewDW, startDW+dwStride);
      if (dbg.active())
        dbg << d_myworld->myrank() << "   Executing INT TG on level " << levelNum << " with old DW " 
            << curDW << "=" << d_scheduler->get_dw(curDW)->getID() << " and new " 
            << curDW+newDWStride << "=" << d_scheduler->get_dw(curDW+newDWStride)->getID() 
            << " CO-DW: " << startDW << " CNDW " << startDW+dwStride << endl;
      d_scheduler->get_dw(curDW+newDWStride)->unfinalize();
      d_scheduler->execute(levelNum+grid->numLevels(), curDW+1);
    }
  }
  if (levelNum == 0) {
    // execute the final TG
    if (dbg.active())
      dbg << d_myworld->myrank() << "   Executing Final TG on level " << levelNum << " with old DW " 
          << curDW << " = " << d_scheduler->get_dw(curDW)->getID() << " and new " 
          << curDW+newDWStride << " = " << d_scheduler->get_dw(curDW+newDWStride)->getID() << endl;
    d_scheduler->get_dw(curDW+newDWStride)->unfinalize();
    d_scheduler->execute(d_scheduler->getNumTaskGraphs()-1);
  }
}

//______________________________________________________________________
bool
AMRSimulationController::needRecompile(double time, double delt,
				       const GridP& grid)
{
  // Currently, d_output, d_sim, d_lb, d_regridder can request a recompile.  --bryan
  bool recompile = false;
  
  // do it this way so everybody can have a chance to maintain their state
  recompile |= (d_output && d_output->needRecompile(time, delt, grid));
  recompile |= (d_sim && d_sim->needRecompile(time, delt, grid));
  recompile |= (d_lb && d_lb->needRecompile(time, delt, grid));
  if (d_doAMR){
    recompile |= (d_regridder && d_regridder->needRecompile(time, delt, grid));
  }
  return recompile;
}
//______________________________________________________________________
void AMRSimulationController::doInitialTimestep(GridP& grid, double& t)
{
  if(d_myworld->myrank() == 0){
    cout << "Compiling initialization taskgraph...\n";
  }

  double start = Time::currentSeconds();
  d_scheduler->mapDataWarehouse(Task::OldDW, 0);
  d_scheduler->mapDataWarehouse(Task::NewDW, 1);
  d_scheduler->mapDataWarehouse(Task::CoarseOldDW, 0);
  d_scheduler->mapDataWarehouse(Task::CoarseNewDW, 1);
  
  if(d_restarting){
    d_lb->possiblyDynamicallyReallocate(grid, true); 
    d_sim->restartInitialize();
  } else {
    // for dynamic lb's, set up initial patch config
    d_lb->possiblyDynamicallyReallocate(grid, true); 

    d_sharedState->setCurrentTopLevelTimeStep( 0 );
    t = d_timeinfo->initTime;
    // Initialize the CFD and/or MPM data
    for(int i=0;i<grid->numLevels();i++) {
      d_sim->scheduleInitialize(grid->getLevel(i), d_scheduler);
      d_sim->scheduleComputeStableTimestep(grid->getLevel(i),d_scheduler);
      
      if (d_doAMR) {
        // so we can initially regrid
        d_regridder->scheduleInitializeErrorEstimate(d_scheduler, grid->getLevel(i));
        d_sim->scheduleInitialErrorEstimate(grid->getLevel(i), d_scheduler);
	d_regridder->scheduleDilation(d_scheduler, grid->getLevel(i));
        if (i > 0) {
          d_sim->scheduleRefineInterface(grid->getLevel(i), d_scheduler, false, true);
        }
      }
    }
  }
  
  if(d_output)
    d_output->finalizeTimestep(t, 0, grid, d_scheduler, 1);

  d_scheduler->compile();
  
  if(d_myworld->myrank() == 0)
    cout << "done taskgraph compile (" << Time::currentSeconds() - start << " seconds)\n";
  // No scrubbing for initial step
  d_scheduler->get_dw(1)->setScrubbing(DataWarehouse::ScrubNone);
  d_scheduler->execute();

  if(d_output)
    d_output->executedTimestep(0, grid);
}
//______________________________________________________________________
bool AMRSimulationController::doInitialTimestepRegridding(GridP& currentGrid)
{
  double start = Time::currentSeconds();
  GridP oldGrid = currentGrid;      
 
  currentGrid = d_regridder->regrid(oldGrid.get_rep(), d_scheduler, d_ups);
  if (d_myworld->myrank() == 0) {
        cout << "  DOING ANOTHER INITIALIZATION REGRID!!!!\n";
        //cout << "---------- OLD GRID ----------" << endl << *(oldGrid.get_rep());
        amrout << "---------- NEW GRID ----------" << endl << *(currentGrid.get_rep());
  }
  double regridTime = Time::currentSeconds() - start;
  if (currentGrid == oldGrid)
    return false;
  
  if (d_myworld->myrank() == 0) {
    cout << "  ADDING ANOTHER LEVEL TO THE GRID:";
    for (int i = 0; i < currentGrid->numLevels(); i++) {
      cout << " Level " << i << " has " << currentGrid->getLevel(i)->numPatches() << " patches...";
    }
    cout << endl;

  }

  // reset the d_scheduler here - it doesn't hurt anything here
  // as we're going to have to recompile the TG anyway.
  d_scheduler->initialize(1, 1);
  d_scheduler->advanceDataWarehouse(currentGrid);
  
  // for dynamic lb's, set up patch config after changing grid
  d_lb->possiblyDynamicallyReallocate(currentGrid, true); 
  
  for(int i=0;i<currentGrid->numLevels();i++) {
    d_sim->scheduleInitialize(currentGrid->getLevel(i), d_scheduler);
    d_sim->scheduleComputeStableTimestep(currentGrid->getLevel(i),d_scheduler);
    if (d_doAMR) {
      d_regridder->scheduleInitializeErrorEstimate(d_scheduler, currentGrid->getLevel(i));
      d_sim->scheduleInitialErrorEstimate(currentGrid->getLevel(i), d_scheduler);
      d_regridder->scheduleDilation(d_scheduler, currentGrid->getLevel(i));
    }
  }
  d_scheduler->compile();
  
  // No scrubbing for initial step
  d_scheduler->get_dw(1)->setScrubbing(DataWarehouse::ScrubNone);
  d_scheduler->execute();

  double time=Time::currentSeconds() - start;
  if(d_myworld->myrank() == 0)
    cout << "done adding level (" << time << " seconds, regridding took " << regridTime << ")\n";

  return true;
}
//______________________________________________________________________
void AMRSimulationController::doRegridding(GridP& currentGrid)
{

  double start = Time::currentSeconds();
  GridP oldGrid = currentGrid;
  currentGrid = d_regridder->regrid(oldGrid.get_rep(), d_scheduler, d_ups);
  double regridTime = Time::currentSeconds() - start;
  d_sharedState->setRegridTimestep(false);
  
  if (currentGrid != oldGrid) {
    if (d_myworld->myrank() == 0) {
      cout << "  REGRIDDING:";
      d_sharedState->setRegridTimestep(true);
      //amrout << "---------- OLD GRID ----------" << endl << *(oldGrid.get_rep());
      for (int i = 0; i < currentGrid->numLevels(); i++) {
        cout << " Level " << i << " has " << currentGrid->getLevel(i)->numPatches() << " patches...";
      }
      cout << endl;
      if (amrout.active()) {
        amrout << "---------- NEW GRID ----------" << endl << *(currentGrid.get_rep());
      }
    }
         
    d_lb->possiblyDynamicallyReallocate(currentGrid, true); 
    double scheduleTime = Time::currentSeconds();
    d_scheduler->scheduleAndDoDataCopy(currentGrid, d_sim);
    scheduleTime = Time::currentSeconds() - scheduleTime;

    double time = Time::currentSeconds() - start;
    if(d_myworld->myrank() == 0){
      cout << "done regridding (" << time << " seconds, regridding took " << regridTime 
           << ", scheduling and copying took " << scheduleTime << ")\n";
    }
  }
}

//______________________________________________________________________
void AMRSimulationController::recompile(double t, double delt, GridP& currentGrid, int totalFine)
{
  if(d_myworld->myrank() == 0)
    cout << "Compiling taskgraph...\n";
  double start = Time::currentSeconds();
  
  d_scheduler->initialize(1, totalFine);
  d_scheduler->fillDataWarehouses(currentGrid);
  
  // Set up new DWs, DW mappings.
  d_scheduler->clearMappings();
  d_scheduler->mapDataWarehouse(Task::OldDW, 0);
  d_scheduler->mapDataWarehouse(Task::NewDW, totalFine);
  d_scheduler->mapDataWarehouse(Task::CoarseOldDW, 0);
  d_scheduler->mapDataWarehouse(Task::CoarseNewDW, totalFine);  
  
  if (d_sim->useLockstepTimeAdvance()) {
    d_sim->scheduleLockstepTimeAdvance(currentGrid, d_scheduler);
  }else {

    if (d_doMultiTaskgraphing) {
      for (int i = 0; i < currentGrid->numLevels(); i++) {
        // taskgraphs 0-numlevels-1
        if ( i > 0)
          // we have the first one already
          d_scheduler->addTaskGraph(Scheduler::NormalTaskGraph);
        dbg << d_myworld->myrank() << "   Creating level " << i << " tg " << endl;
        d_sim->scheduleTimeAdvance(currentGrid->getLevel(i), d_scheduler);
      }
      if (d_doAMR && currentGrid->numLevels() > 1) {
        for (int i = 0; i < currentGrid->numLevels(); i++) {
          dbg << d_myworld->myrank() << "   Doing Int TG level " << i << " tg " << endl;
          // taskgraphs numlevels-2*numlevels-1
          d_scheduler->addTaskGraph(Scheduler::IntermediateTaskGraph);
          // schedule a coarsen from the finest level to this level
          for (int j = currentGrid->numLevels()-2; j >= i; j--) {
            dbg << d_myworld->myrank() << "   schedule coarsen on level " << j << endl;
            d_sim->scheduleCoarsen(currentGrid->getLevel(j), d_scheduler);
            d_sim->scheduleFinalizeTimestep(currentGrid->getLevel(j), d_scheduler);
          }
          // schedule a refineInterface from this level to the finest level
          for (int j = i; j < currentGrid->numLevels(); j++) {
            if (j != 0) {
              dbg << d_myworld->myrank() << "   schedule RI on level " << j << " for tg " << i << " coarseold " << (j==i) << " coarsenew " << true << endl;
              d_sim->scheduleRefineInterface(currentGrid->getLevel(j), d_scheduler, j==i, true);
            }
          }
        }
      }
      // for the final error estimate and stable timestep tasks
      d_scheduler->addTaskGraph(Scheduler::IntermediateTaskGraph);
    }
    else {
      subCycleCompile(currentGrid, 0, totalFine, 0, 0);
      d_scheduler->clearMappings();
      d_scheduler->mapDataWarehouse(Task::OldDW, 0);
      d_scheduler->mapDataWarehouse(Task::NewDW, totalFine);
    }
  }
    
  for(int i = currentGrid->numLevels()-1; i >= 0; i--){
    dbg << d_myworld->myrank() << "   final TG " << i << endl;
    if (d_doAMR) {
      d_regridder->scheduleInitializeErrorEstimate(d_scheduler, currentGrid->getLevel(i));
      d_sim->scheduleErrorEstimate(currentGrid->getLevel(i), d_scheduler);
      d_regridder->scheduleDilation(d_scheduler, currentGrid->getLevel(i));
    }    
    d_sim->scheduleComputeStableTimestep(currentGrid->getLevel(i), d_scheduler);
  }

  if(d_output){
    d_output->finalizeTimestep(t, delt, currentGrid, d_scheduler, true, d_sharedState->needAddMaterial());
  }
  
  d_scheduler->compile();
 
  double dt=Time::currentSeconds() - start;
  if(d_myworld->myrank() == 0)
    cout << "DONE TASKGRAPH RE-COMPILE (" << dt << " seconds)\n";
  
  d_sharedState->setNeedAddMaterial(0);
}
//______________________________________________________________________
void AMRSimulationController::executeTimestep(double t, double& delt, GridP& currentGrid, int totalFine)
{
  // If the timestep needs to be
  // restarted, this loop will execute multiple times.
  bool success = true;
  double orig_delt = delt;
  do {
    bool restartable = d_sim->restartableTimesteps();
    d_scheduler->setRestartable(restartable);
    if (restartable)
      d_scheduler->get_dw(0)->setScrubbing(DataWarehouse::ScrubNonPermanent);
    else
      d_scheduler->get_dw(0)->setScrubbing(DataWarehouse::ScrubComplete);
    
    for(int i=0;i<=totalFine;i++) {
      if (d_doAMR || d_lb->getNthProc() > 1 || d_reduceUda)
        d_scheduler->get_dw(i)->setScrubbing(DataWarehouse::ScrubNone);
      else {
        d_scheduler->get_dw(1)->setScrubbing(DataWarehouse::ScrubNonPermanent);
      }
    }
    
    if (d_scheduler->getNumTaskGraphs() == 1)
      d_scheduler->execute();
    else {
      subCycleExecute(currentGrid, 0, totalFine, 0, true);
    }
    if(d_scheduler->get_dw(totalFine)->timestepRestarted()){
      ASSERT(restartable);
      // Figure out new delt
      double new_delt = d_sim->recomputeTimestep(delt);
      if(d_myworld->myrank() == 0){
        cout << "Restarting timestep at " << t << ", changing delt from "
             << delt << " to " << new_delt << '\n';
      }
      d_output->reEvaluateOutputTimestep(orig_delt, new_delt);
      delt = new_delt;
      d_scheduler->get_dw(0)->override(delt_vartype(new_delt),
                                       d_sharedState->get_delt_label());

      for (int i=1; i <= totalFine; i++){
        d_scheduler->replaceDataWarehouse(i, currentGrid);
      }

      double delt_fine = delt;
      int skip=totalFine;
      for(int i=0;i<currentGrid->numLevels();i++){
        const Level* level = currentGrid->getLevel(i).get_rep();
        if(i != 0){
          delt_fine /= d_sharedState->timeRefinementRatio();
          skip /= d_sharedState->timeRefinementRatio();
        }
        for(int idw=0;idw<totalFine;idw+=skip){
          DataWarehouse* dw = d_scheduler->get_dw(idw);
          dw->override(delt_vartype(delt_fine), d_sharedState->get_delt_label(),
                       level);
        }
      }
      success = false;
      
    } else {
      success = true;
      if(d_scheduler->get_dw(1)->timestepAborted()){
        throw InternalError("Execution aborted, cannot restart timestep\n", __FILE__, __LINE__);
      }
    }
  } while(!success);

}
