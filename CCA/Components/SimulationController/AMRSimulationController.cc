/*
 * The MIT License
 *
 * Copyright (c) 1997-2012 The University of Utah
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 */

#include <sci_defs/malloc_defs.h>
#include <sci_defs/gperftools_defs.h>

#include <CCA/Components/SimulationController/AMRSimulationController.h>

#include <Core/Containers/Array3.h>
#include <Core/Geometry/IntVector.h>
#include <Core/Geometry/Vector.h>
#include <Core/Math/MiscMath.h>
#include <Core/OS/ProcessInfo.h>
#include <Core/Thread/Time.h>
#include <Core/Thread/Thread.h>
#include <Core/Parallel/Parallel.h>

#include <Core/Exceptions/ProblemSetupException.h>
#include <Core/Grid/Box.h>
#include <Core/Grid/Grid.h>
#include <Core/Grid/Level.h>
#include <Core/Grid/Patch.h>
#include <Core/Grid/SimulationTime.h>
#include <Core/Grid/SimulationState.h>
#include <Core/Grid/Variables/PerPatch.h>
#include <Core/Grid/Variables/ReductionVariable.h>
#include <Core/Grid/Variables/SoleVariable.h>
#include <Core/Grid/Variables/VarLabel.h>
#include <Core/Grid/Variables/VarLabelMatl.h>
#include <Core/Grid/Variables/VarTypes.h>
#include <Core/Parallel/ProcessorGroup.h>
#include <Core/ProblemSpec/ProblemSpecP.h>
#include <Core/ProblemSpec/ProblemSpec.h>
#include <Core/Tracker/TrackerClient.h>

#include <CCA/Components/PatchCombiner/PatchCombiner.h>
#include <CCA/Components/PatchCombiner/UdaReducer.h>
#include <CCA/Components/Regridder/PerPatchVars.h>
#include <CCA/Ports/DataWarehouse.h>
#include <CCA/Ports/LoadBalancer.h>
#include <CCA/Ports/Output.h>
#include <CCA/Ports/ProblemSpecInterface.h>
#include <CCA/Ports/Regridder.h>
#include <CCA/Ports/Scheduler.h>
#include <CCA/Ports/SimulationInterface.h>

#include <TauProfilerForSCIRun.h>

#include <iostream>
#include <iomanip>

using namespace std;
using namespace SCIRun;
using namespace Uintah;

DebugStream amrout("AMR", false);
static DebugStream dbg("AMRSimulationController", false);
static DebugStream dbg_barrier("MPIBarriers",false);
static DebugStream dbg_dwmem("LogDWMemory",false);
static DebugStream gprofile("CPUProfiler",false);
static DebugStream gheapprofile("HeapProfiler",false);
static DebugStream gheapchecker("HeapChecker",false);

AMRSimulationController::AMRSimulationController(const ProcessorGroup* myworld,
                                                 bool doAMR, ProblemSpecP pspec) :
  SimulationController(myworld, doAMR, pspec)
{
}

AMRSimulationController::~AMRSimulationController()
{
}

double barrier_times[5]={0};
void
AMRSimulationController::run()
{
  MALLOC_TRACE_TAG_SCOPE("AMRSimulationController::run()");
 

  if (gprofile.active()){
#ifdef USE_GPERFTOOLS
    char gprofname[512];
    sprintf(gprofname, "cpuprof-rank%d", d_myworld->myrank());
    ProfilerStart(gprofname);
#endif
  }
  
  if (gheapprofile.active()){
#ifdef USE_GPERFTOOLS
    char gheapprofname[512];
    sprintf(gheapprofname, "heapprof-rank%d", d_myworld->myrank());
    HeapProfilerStart(gheapprofname);
#endif
  }

#ifdef USE_GPERFTOOLS
  HeapLeakChecker * heap_checker=NULL;
#endif
  if (gheapchecker.active()){
    if (!gheapprofile.active()){
#ifdef USE_GPERFTOOLS
      char gheapchkname[512];
      sprintf(gheapchkname, "heapchk-rank%d", d_myworld->myrank());
      heap_checker= new HeapLeakChecker(gheapchkname);
#endif
    } else {
      cout<< "HEAPCHECKER: Cannot start with heapprofiler" <<endl;
    }
  }

  bool log_dw_mem=false;

  if(dbg_dwmem.active()) {
    log_dw_mem=true;
  }

   // sets up sharedState, timeinfo, output, scheduler, lb
   preGridSetup();

   // create grid
   GridP currentGrid = gridSetup();

   d_scheduler->initialize(1, 1);
   d_scheduler->advanceDataWarehouse(currentGrid, true);
    
   double time;

   // set up sim, regridder, and finalize sharedState
   // also reload from the DataArchive on restart
   postGridSetup( currentGrid, time );

   calcStartTime();

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
   doInitialTimestep( currentGrid, time );

   setStartSimTime( time );
   initSimulationStatsVars();
#ifndef DISABLE_SCI_MALLOC
   AllocatorSetDefaultTagLineNumber(d_sharedState->getCurrentTopLevelTimeStep());
#endif
   ////////////////////////////////////////////////////////////////////////////
   // The main time loop; here the specified problem is actually getting solved
   
   bool   first = true;
   int    iterations = d_sharedState->getCurrentTopLevelTimeStep();
   double delt = 0;

   double start;
  
   d_lb->resetCostForecaster();
   while( ( time < d_timeinfo->maxTime ) &&
          ( iterations < d_timeinfo->maxTimestep ) && 
          ( d_timeinfo->max_wall_time == 0 || getWallTime() < d_timeinfo->max_wall_time )  ) {
  
     if (gheapprofile.active()){
#ifdef USE_GPERFTOOLS
       char heapename[512];
       sprintf(heapename, "Timestep %d", iterations);
       HeapProfilerDump(heapename);
#endif
     }
     TrackerClient::trackEvent( Tracker::TIMESTEP_STARTED, time );

     MALLOC_TRACE_TAG_SCOPE("AMRSimulationController::run()::control loop");
     if(dbg_barrier.active()) {
       for(int i=0;i<5;i++) {
         barrier_times[i]=0;
       }
     }
#ifdef USE_TAU_PROFILING
     char tmpname[512];
     sprintf (tmpname, "Iteration %d", iterations);
     TAU_PROFILE_TIMER_DYNAMIC(iteration_timer, tmpname, "", TAU_USER);
     TAU_PROFILE_START(iteration_timer); 
#endif
     
     if (d_regridder && d_regridder->needsToReGrid(currentGrid) && (!first || (d_restarting))) {
       doRegridding(currentGrid, false);
     }
    
     // Compute number of dataWarehouses - multiplies by the time refinement
     // ratio for each level you increase
     int totalFine=1;
     if (!d_sharedState->isLockstepAMR()) {
       for(int i=1;i<currentGrid->numLevels();i++) {
         totalFine *= currentGrid->getLevel(i)->getRefinementRatioMaxDim();
       }
     }
     
     d_sharedState->d_prev_delt = delt;
     iterations++;
 
     // get delt and adjust it
     delt_vartype delt_var;
     DataWarehouse* newDW = d_scheduler->getLastDW();
     newDW->get(delt_var, d_sharedState->get_delt_label());

     delt = delt_var;

     // a component may update the output interval or the checkpoint interval
     // during a simulation.  For example in deflagration -> detonation simulations
     if (d_output && d_sharedState->updateOutputInterval()) {
       //if no value computed such as during the init timestep, use the value from ups file
       if (!newDW->exists(d_sharedState->get_outputInterval_label())) {
         newDW->put(min_vartype(d_output->getOutputInterval()),d_sharedState->get_outputInterval_label());
       } else {
         min_vartype outputInv_var;
         newDW->get(outputInv_var, d_sharedState->get_outputInterval_label());
         d_output->updateOutputInterval(outputInv_var);
       }
     }

     if (d_output && d_sharedState->updateCheckpointInterval()) {
       if (!newDW->exists(d_sharedState->get_checkpointInterval_label())) {
         newDW->put(min_vartype(d_output->getCheckpointInterval()),d_sharedState->get_checkpointInterval_label());
       } else {
         min_vartype checkInv_var;
         newDW->get(checkInv_var, d_sharedState->get_checkpointInterval_label());
         d_output->updateCheckpointInterval(checkInv_var);
       }
     }
     
     // delt adjusted based on timeinfo parameters
     adjustDelT( delt, d_sharedState->d_prev_delt, first, time );
     newDW->override(delt_vartype(delt), d_sharedState->get_delt_label());

     // printSimulationStats( d_sharedState, delt, t );

     if(log_dw_mem){
       // Remember, this isn't logged if DISABLE_SCI_MALLOC is set
       // (So usually in optimized mode this will not be run.)
       d_scheduler->logMemoryUse();
       ostringstream fn;
       fn << "alloc." << setw(5) << setfill('0') << d_myworld->myrank() << ".out";
       string filename(fn.str());
#if !defined( _WIN32 ) && !defined( DISABLE_SCI_MALLOC )
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
     if(dbg_barrier.active()) {
       start=Time::currentSeconds();
       MPI_Barrier(d_myworld->getComm());
       barrier_times[2]+=Time::currentSeconds()-start;
     }

     // Yes, I know this is kind of hacky, but this is the only way to get a new grid from UdaReducer
     //   Needs to be done before advanceDataWarehouse
     if (d_reduceUda) currentGrid = static_cast<UdaReducer*>(d_sim)->getGrid();

     // After one step (either timestep or initialization) and correction
     // the delta we can finally, finalize our old timestep, eg. 
     // finalize and advance the Datawarehouse
     d_scheduler->advanceDataWarehouse(currentGrid);

     // Put the current time into the shared state so other components
     // can access it.  Also increment (by one) the current time step
     // number so components can tell what timestep they are on. 
     d_sharedState->setElapsedTime( time );
     d_sharedState->incrementCurrentTopLevelTimeStep();
#ifndef DISABLE_SCI_MALLOC
     AllocatorSetDefaultTagLineNumber(d_sharedState->getCurrentTopLevelTimeStep());
#endif
     // Each component has their own init_delt specified.  On a switch
     // from one component to the next, we need to adjust the delt to
     // that specified in the input file.  To detect the switch of components,
     // we compare the old_init_delt before the needRecompile() to the 
     // new_init_delt after the needRecompile().  

     double old_init_delt = d_timeinfo->max_initial_delt;
     double new_init_delt = 0.;

     bool nr;
     if( (nr=needRecompile( time, delt, currentGrid )) || first ){
       if(nr)
       {
         //if needRecompile returns true it has reload balanced and thus we need 
         //to assign the boundary conditions.
          currentGrid->assignBCS(d_grid_ps,d_lb);
          currentGrid->performConsistencyCheck();
       }
       new_init_delt = d_timeinfo->max_initial_delt;
       if (new_init_delt != old_init_delt) {
         // writes to the DW in the next section below
         delt = new_init_delt;
       }
       first = false;
       recompile( time, delt, currentGrid, totalFine );
     }
     else {
       if (d_output){
         // This is not correct if we have switched to a different
         // component, since the delt will be wrong 
         d_output->finalizeTimestep( time, delt, currentGrid, d_scheduler, 0 );
       }
     }

     if(dbg_barrier.active())
     {
       start=Time::currentSeconds();
       MPI_Barrier(d_myworld->getComm());
       barrier_times[3]+=Time::currentSeconds()-start;
     }

     // adjust the delt for each level and store it in all applicable dws.
     double delt_fine = delt;
     int skip=totalFine;
     for(int i=0;i<currentGrid->numLevels();i++){
       const Level* level = currentGrid->getLevel(i).get_rep();
       if(d_doAMR && i != 0 && !d_sharedState->isLockstepAMR()){
         int rr = level->getRefinementRatioMaxDim();
	 delt_fine /= rr;
	 skip /= rr;
       }
       for(int idw=0;idw<totalFine;idw+=skip){
	 DataWarehouse* dw = d_scheduler->get_dw(idw);
	 dw->override(delt_vartype(delt_fine), d_sharedState->get_delt_label(),
		      level);
       }
     }
     
     // override for the global level as well (which only matters on dw 0)
     d_scheduler->get_dw(0)->override(delt_vartype(delt),
                                      d_sharedState->get_delt_label());

     calcWallTime();

     printSimulationStats( d_sharedState->getCurrentTopLevelTimeStep()-1, delt, time );
     // Execute the current timestep, restarting if necessary
     d_sharedState->d_current_delt = delt;
     executeTimestep( time, delt, currentGrid, totalFine );
     
     // Print MPI statistics
     d_scheduler->printMPIStats();

     // Update the profiler weights
     d_lb->finalizeContributions(currentGrid);
     
     if(dbg_barrier.active()) {
       start=Time::currentSeconds();
       MPI_Barrier(d_myworld->getComm());
       barrier_times[4]+=Time::currentSeconds()-start;
       double avg[5];
       MPI_Reduce(&barrier_times,&avg,5,MPI_DOUBLE,MPI_SUM,0,d_myworld->getComm());
       if(d_myworld->myrank()==0) {
          cout << "Barrier Times: "; 
          for(int i=0;i<5;i++)
          {
            avg[i]/=d_myworld->size();
            cout << avg[i] << " ";
          }
          cout << endl;
       }
     }

     if(d_output){
       d_output->executedTimestep(delt, currentGrid);
     }
#ifdef USE_TAU_PROFILING
     TAU_PROFILE_STOP(iteration_timer);
//      TAU_PROFILE_TIMER(sleepy, "Sleep", "", TAU_USER);
//      TAU_PROFILE_START(sleepy);
//      sleep(1);
//      TAU_PROFILE_STOP(sleepy);
#endif
     time += delt;
     TAU_DB_DUMP();
   } // end while ( time )

   // print for the final timestep, as the one above is in the middle of a while loop - get new delt, and set walltime first
   delt_vartype delt_var;
   d_scheduler->getLastDW()->get(delt_var, d_sharedState->get_delt_label());
   delt = delt_var;
   adjustDelT( delt, d_sharedState->d_prev_delt, d_sharedState->getCurrentTopLevelTimeStep(), time );
   calcWallTime();
   printSimulationStats( d_sharedState->getCurrentTopLevelTimeStep(), delt, time );

   // d_ups->releaseDocument();
  if (gprofile.active()){
#ifdef USE_GPERFTOOLS
    ProfilerStop();
#endif
  }
  if (gheapprofile.active()){
#ifdef USE_GPERFTOOLS
    HeapProfilerStop();
#endif
  }
   if (gheapchecker.active() && !gheapprofile.active()){
#ifdef USE_GPERFTOOLS
     if (heap_checker && !heap_checker->NoLeaks()) 
       cout << "HEAPCHECKER: MEMORY LEACK DETECTED!" << endl;
     delete heap_checker;
#endif
   }
} // end run()

//______________________________________________________________________
void
AMRSimulationController::subCycleCompile(GridP& grid, int startDW, int dwStride, int step, int numLevel)
{
  MALLOC_TRACE_TAG_SCOPE("AMRSimulationController::subCycleCompile()");
  //amrout << "Start AMRSimulationController::subCycleCompile, level=" << numLevel << '\n';
  // We are on (the fine) level numLevel
  LevelP fineLevel = grid->getLevel(numLevel);
  LevelP coarseLevel;
  int coarseStartDW;
  int coarseDWStride;
  int numCoarseSteps; // how many steps between this level and the coarser
  int numFineSteps;   // how many steps between this level and the finer
  if (numLevel > 0) {
    numCoarseSteps = d_sharedState->isLockstepAMR() ? 1 : fineLevel->getRefinementRatioMaxDim();
    coarseLevel = grid->getLevel(numLevel-1);
    coarseDWStride = dwStride * numCoarseSteps;
    coarseStartDW = (startDW/coarseDWStride)*coarseDWStride;
  }
  else {
    coarseDWStride = dwStride;
    coarseStartDW = startDW;
    numCoarseSteps = 0;
  }
  

  ASSERT(dwStride > 0 && numLevel < grid->numLevels())
  d_scheduler->clearMappings();
  d_scheduler->mapDataWarehouse(Task::OldDW, startDW);
  d_scheduler->mapDataWarehouse(Task::NewDW, startDW+dwStride);
  d_scheduler->mapDataWarehouse(Task::CoarseOldDW, coarseStartDW);
  d_scheduler->mapDataWarehouse(Task::CoarseNewDW, coarseStartDW+coarseDWStride);
  
  d_sim->scheduleTimeAdvance(fineLevel, d_scheduler);

  if (d_doAMR) {
    if(numLevel+1 < grid->numLevels()){
      numFineSteps = d_sharedState->isLockstepAMR() ? 1 : fineLevel->getFinerLevel()->getRefinementRatioMaxDim();
      int newStride = dwStride/numFineSteps;
      for(int substep=0;substep < numFineSteps;substep++){
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
  if (d_doAMR && (step < numCoarseSteps -1 || numLevel == 0)) {
    
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

void
AMRSimulationController::subCycleExecute(GridP& grid, int startDW, int dwStride, int levelNum, bool rootCycle)
{
  MALLOC_TRACE_TAG_SCOPE("AMRSimulationController::subCycleExecutue()");
  // there are 2n+1 taskgraphs, n for the basic timestep, n for intermediate 
  // timestep work, and 1 for the errorEstimate and stableTimestep, where n
  // is the number of levels.
  
  //amrout << "Start AMRSimulationController::subCycleExecute, level=" << numLevel << '\n';
  // We are on (the fine) level numLevel
  int numSteps;
  if (levelNum == 0 || d_sharedState->isLockstepAMR())
    numSteps = 1;
  else {
    numSteps = grid->getLevel(levelNum)->getRefinementRatioMaxDim();
  }
  
  int newDWStride = dwStride/numSteps;

  DataWarehouse::ScrubMode oldScrubbing = (/*d_lb->isDynamic() ||*/ d_sim->restartableTimesteps()) ? 
    DataWarehouse::ScrubNonPermanent : DataWarehouse::ScrubComplete;

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
    //if (Uintah::Parallel::getMaxThreads() < 1) { 
    d_scheduler->get_dw(curDW)->setScrubbing(oldScrubbing); // OldDW
    d_scheduler->get_dw(curDW+newDWStride)->setScrubbing(DataWarehouse::ScrubNonPermanent); // NewDW
    d_scheduler->get_dw(startDW)->setScrubbing(oldScrubbing); // CoarseOldDW
    d_scheduler->get_dw(startDW+dwStride)->setScrubbing(DataWarehouse::ScrubNonPermanent); // CoarseNewDW
    //}
    
    // we need to unfinalize because execute finalizes all new DWs, and we need to write into them still
    // (even if we finalized only the NewDW in execute, we will still need to write into that DW)
    d_scheduler->get_dw(curDW+newDWStride)->unfinalize();

    // iteration only matters if it's zero or greater than 0
    int iteration = curDW + (d_lastRecompileTimestep == d_sharedState->getCurrentTopLevelTimeStep() ? 0 : 1);
    if (dbg.active())
      dbg << d_myworld->myrank() << "   Executing TG on level " << levelNum << " with old DW " 
          << curDW << "=" << d_scheduler->get_dw(curDW)->getID() << " and new " 
          << curDW+newDWStride << "=" << d_scheduler->get_dw(curDW+newDWStride)->getID() 
          << "CO-DW: " << startDW << " CNDW " << startDW+dwStride << " on iteration " << iteration << endl;

    d_scheduler->execute(levelNum, iteration);
    
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

    //if (Uintah::Parallel::getMaxThreads() < 1) { 
      d_scheduler->get_dw(curDW)->setScrubbing(oldScrubbing); // OldDW
      d_scheduler->get_dw(curDW+newDWStride)->setScrubbing(DataWarehouse::ScrubNonPermanent); // NewDW
      d_scheduler->get_dw(startDW)->setScrubbing(oldScrubbing); // CoarseOldDW
      d_scheduler->get_dw(startDW+dwStride)->setScrubbing(DataWarehouse::ScrubNonPermanent); // CoarseNewDW
    //}
      if (dbg.active())
        dbg << d_myworld->myrank() << "   Executing INT TG on level " << levelNum << " with old DW " 
            << curDW << "=" << d_scheduler->get_dw(curDW)->getID() << " and new " 
            << curDW+newDWStride << "=" << d_scheduler->get_dw(curDW+newDWStride)->getID() 
            << " CO-DW: " << startDW << " CNDW " << startDW+dwStride << " on iteration " << iteration << endl;
      d_scheduler->get_dw(curDW+newDWStride)->unfinalize();
      d_scheduler->execute(levelNum+grid->numLevels(), iteration);
    }
    
    if (curDW % dwStride != 0) {
      //the currentDW(old datawarehouse) should no longer be needed - in the case of NonPermanent OldDW scrubbing
      d_scheduler->get_dw(curDW)->clear();
    }
    
  }
  if (levelNum == 0) {
    // execute the final TG
    if (dbg.active())
      dbg << d_myworld->myrank() << "   Executing Final TG on level " << levelNum << " with old DW " 
          << curDW << " = " << d_scheduler->get_dw(curDW)->getID() << " and new " 
          << curDW+newDWStride << " = " << d_scheduler->get_dw(curDW+newDWStride)->getID() << endl;
    d_scheduler->get_dw(curDW+newDWStride)->unfinalize();
    d_scheduler->execute(d_scheduler->getNumTaskGraphs()-1, 1);
  }
}

//______________________________________________________________________
bool
AMRSimulationController::needRecompile(double time, double delt,
				       const GridP& grid)
{
  MALLOC_TRACE_TAG_SCOPE("AMRSimulationController::needRecompile()");
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
void
AMRSimulationController::doInitialTimestep(GridP& grid, double& t)
{
  MALLOC_TRACE_TAG_SCOPE("AMRSimulationController::doInitialTimestep()");
  double start = Time::currentSeconds();
  d_scheduler->mapDataWarehouse(Task::OldDW, 0);
  d_scheduler->mapDataWarehouse(Task::NewDW, 1);
  d_scheduler->mapDataWarehouse(Task::CoarseOldDW, 0);
  d_scheduler->mapDataWarehouse(Task::CoarseNewDW, 1);
  
  if(d_restarting){
    d_lb->possiblyDynamicallyReallocate(grid, LoadBalancer::restart); 
    grid->performConsistencyCheck();
    d_sim->restartInitialize();
    if (d_regridder && d_regridder->isAdaptive()) {
      // On restart:
      //   we must set up the tasks (but not compile) so we can have the
      //   initial OldDW Requirements in order to regrid straightaway
      for(int i=grid->numLevels()-1; i >= 0; i--) {
        d_sim->scheduleTimeAdvance(grid->getLevel(i), d_scheduler);
      }
    }  
  } else {
    d_sharedState->setCurrentTopLevelTimeStep( 0 );
    // for dynamic lb's, set up initial patch config
    d_lb->possiblyDynamicallyReallocate(grid, LoadBalancer::init); 
    grid->assignBCS(d_grid_ps,d_lb);
    grid->performConsistencyCheck();
    t = d_timeinfo->initTime;

    bool needNewLevel = false;
    do {
      if (needNewLevel) {
        d_scheduler->initialize(1, 1);
        d_scheduler->advanceDataWarehouse(grid, true);
      }

      if(d_myworld->myrank() == 0){
        cout << "Compiling initialization taskgraph...\n";
      }

      // Initialize the CFD and/or MPM data
      for(int i=grid->numLevels()-1; i >= 0; i--) {
        d_sim->scheduleInitialize(grid->getLevel(i), d_scheduler);

        if (d_regridder) {
          // so we can initially regrid
          d_regridder->scheduleInitializeErrorEstimate(grid->getLevel(i));
          d_sim->scheduleInitialErrorEstimate(grid->getLevel(i), d_scheduler);
          if (i < d_regridder->maxLevels()-1) // we don't use error estimates if we don't make another level, so don't dilate
            d_regridder->scheduleDilation(grid->getLevel(i));
        }
      }
      scheduleComputeStableTimestep(grid,d_scheduler);

      if(d_output)
        d_output->finalizeTimestep(t, 0, grid, d_scheduler, 1);
      
      d_scheduler->compile();
      double end = Time::currentSeconds() - start;
      if(d_myworld->myrank() == 0)
        cout << "done taskgraph compile (" << end << " seconds)\n";
      // No scrubbing for initial step
      d_scheduler->get_dw(1)->setScrubbing(DataWarehouse::ScrubNone);
      d_scheduler->execute();

      needNewLevel = d_regridder && d_regridder->isAdaptive() && grid->numLevels()<d_regridder->maxLevels() && doRegridding(grid, true);
    } while (needNewLevel);

    if(d_output)
      d_output->executedTimestep(0, grid);

  }
}

//______________________________________________________________________
bool AMRSimulationController::doRegridding(GridP& currentGrid, bool initialTimestep)
{
  MALLOC_TRACE_TAG_SCOPE("AMRSimulationController::doRegridding()");
  TAU_PROFILE("AMRSimulationController::doRegridding()", " ", TAU_USER);
  double start = Time::currentSeconds();
  GridP oldGrid = currentGrid;
  currentGrid = d_regridder->regrid(oldGrid.get_rep());
  if(dbg_barrier.active()) {
    double start;
    start=Time::currentSeconds();
    MPI_Barrier(d_myworld->getComm());
    barrier_times[0]+=Time::currentSeconds()-start;
  }
  double regridTime = Time::currentSeconds() - start;
  d_sharedState->regriddingTime += regridTime;
  d_sharedState->setRegridTimestep(false);

  int lbstate = initialTimestep ? LoadBalancer::init : LoadBalancer::regrid;

  if (currentGrid != oldGrid) {
    d_lb->possiblyDynamicallyReallocate(currentGrid, lbstate); 
    if(dbg_barrier.active()) {
      double start;
      start=Time::currentSeconds();
      MPI_Barrier(d_myworld->getComm());
      barrier_times[1]+=Time::currentSeconds()-start;
    }
    currentGrid->assignBCS(d_grid_ps,d_lb);
    currentGrid->performConsistencyCheck();

    if (d_myworld->myrank() == 0) {
      cout << "  REGRIDDING:";
      d_sharedState->setRegridTimestep(true);
      //amrout << "---------- OLD GRID ----------" << endl << *(oldGrid.get_rep());
      for (int i = 0; i < currentGrid->numLevels(); i++) {
        cout << " Level " << i << " has " << currentGrid->getLevel(i)->numPatches() << " patches...";
      }
      cout << endl;
      if (amrout.active()) {
        amrout << "---------- NEW GRID ----------" << endl;
        amrout << "Grid has " << currentGrid->numLevels() << " level(s)" << endl;
        for ( int levelIndex = 0; levelIndex < currentGrid->numLevels(); levelIndex++ ) {
          LevelP level = currentGrid->getLevel( levelIndex );
          amrout << "  Level " << level->getID()
            << ", indx: "<< level->getIndex()
            << " has " << level->numPatches() << " patch(es)" << endl;
          for ( Level::patchIterator patchIter = level->patchesBegin(); patchIter < level->patchesEnd(); patchIter++ ) {
            const Patch* patch = *patchIter;
            amrout << "(Patch " << patch->getID() << " proc " << d_lb->getPatchwiseProcessorAssignment(patch)
              << ": box=" << patch->getExtraBox()
              << ", lowIndex=" << patch->getExtraCellLowIndex() << ", highIndex="
              << patch->getExtraCellHighIndex() << ")" << endl;
          }
        }
      }
    }

    double scheduleTime = Time::currentSeconds();
    if (!initialTimestep)
      d_scheduler->scheduleAndDoDataCopy(currentGrid, d_sim);
    scheduleTime = Time::currentSeconds() - scheduleTime;

    double time = Time::currentSeconds() - start;
    if(d_myworld->myrank() == 0){
      cout << "done regridding (" << time << " seconds, regridding took " << regridTime;
      if (!initialTimestep)
        cout << ", scheduling and copying took " << scheduleTime << ")";
      cout << endl;
    }
    return true;
  }
  return false;
}

//______________________________________________________________________
void
AMRSimulationController::recompile(double t, double delt, GridP& currentGrid, int totalFine)
{
  MALLOC_TRACE_TAG_SCOPE("AMRSimulationController::Recompile()");
  if(d_myworld->myrank() == 0)
    cout << "Compiling taskgraph...\n";
  d_lastRecompileTimestep = d_sharedState->getCurrentTopLevelTimeStep();
  double start = Time::currentSeconds();
  
  d_scheduler->initialize(1, totalFine);
  d_scheduler->fillDataWarehouses(currentGrid);
  
  // Set up new DWs, DW mappings.
  d_scheduler->clearMappings();
  d_scheduler->mapDataWarehouse(Task::OldDW, 0);
  d_scheduler->mapDataWarehouse(Task::NewDW, totalFine);
  d_scheduler->mapDataWarehouse(Task::CoarseOldDW, 0);
  d_scheduler->mapDataWarehouse(Task::CoarseNewDW, totalFine);  
  
  if (d_doMultiTaskgraphing) {
    for (int i = 0; i < currentGrid->numLevels(); i++) {
      // taskgraphs 0-numlevels-1
      if ( i > 0)
        // we have the first one already
        d_scheduler->addTaskGraph(Scheduler::NormalTaskGraph);
      dbg << d_myworld->myrank() << "   Creating level " << i << " tg " << endl;
      d_sim->scheduleTimeAdvance(currentGrid->getLevel(i), d_scheduler);
    }
    for (int i = 0; i < currentGrid->numLevels(); i++) {
      if (d_doAMR && currentGrid->numLevels() > 1) {
        dbg << d_myworld->myrank() << "   Doing Int TG level " << i << " tg " << endl;
        // taskgraphs numlevels-2*numlevels-1
        d_scheduler->addTaskGraph(Scheduler::IntermediateTaskGraph);
      }
      // schedule a coarsen from the finest level to this level
      for (int j = currentGrid->numLevels()-2; j >= i; j--) {
        dbg << d_myworld->myrank() << "   schedule coarsen on level " << j << endl;
        d_sim->scheduleCoarsen(currentGrid->getLevel(j), d_scheduler);
      }
      d_sim->scheduleFinalizeTimestep(currentGrid->getLevel(i), d_scheduler);
      // schedule a refineInterface from this level to the finest level
      for (int j = i; j < currentGrid->numLevels(); j++) {
        if (j != 0) {
          dbg << d_myworld->myrank() << "   schedule RI on level " << j << " for tg " << i << " coarseold " << (j==i) << " coarsenew " << true << endl;
          d_sim->scheduleRefineInterface(currentGrid->getLevel(j), d_scheduler, j==i, true);
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
    
  for(int i = currentGrid->numLevels()-1; i >= 0; i--){
    dbg << d_myworld->myrank() << "   final TG " << i << endl;
    if (d_regridder) {
      d_regridder->scheduleInitializeErrorEstimate(currentGrid->getLevel(i));
      d_sim->scheduleErrorEstimate(currentGrid->getLevel(i), d_scheduler);
      if (i < d_regridder->maxLevels()-1) // we don't use error estimates if we don't make another level, so don't dilate
        d_regridder->scheduleDilation(currentGrid->getLevel(i));
    }    
  }
  scheduleComputeStableTimestep(currentGrid, d_scheduler);

  if(d_output){
    d_output->finalizeTimestep(t, delt, currentGrid, d_scheduler, true, d_sharedState->needAddMaterial());
  }
  
  d_scheduler->compile();
 
  double dt=Time::currentSeconds() - start;
  if(d_myworld->myrank() == 0)
    cout << "DONE TASKGRAPH RE-COMPILE (" << dt << " seconds)\n";
  d_sharedState->compilationTime += dt;
  d_sharedState->setNeedAddMaterial(0);
}
//______________________________________________________________________
void
AMRSimulationController::executeTimestep(double t, double& delt, GridP& currentGrid, int totalFine)
{
  MALLOC_TRACE_TAG_SCOPE("AMRSimulationController::executeTimestep()");
  TAU_PROFILE("AMRSimulationController::executeTimestep()"," ", TAU_USER);
  // If the timestep needs to be
  // restarted, this loop will execute multiple times.
  bool success = true;
  double orig_delt = delt;
  do {
    bool restartable = d_sim->restartableTimesteps();
    d_scheduler->setRestartable(restartable);
    //if (Uintah::Parallel::getMaxThreads() < 1) { 
    if (restartable)
      d_scheduler->get_dw(0)->setScrubbing(DataWarehouse::ScrubNonPermanent);
    else
      d_scheduler->get_dw(0)->setScrubbing(DataWarehouse::ScrubComplete);
    
    for(int i=0;i<=totalFine;i++) {
      //AMRICE has a problem with scrubbing variables to early, getNthProc requires the variables after they would have been scrubbed
      //dynamic load balancing requires some particle variables for collectParticles that would be scrubbed
      if (/*(d_doAMR && !d_sharedState->isLockstepAMR()) ||*/ d_lb->getNthProc() > 1 /*|| d_lb->isDynamic()*/ || d_reduceUda)
        d_scheduler->get_dw(i)->setScrubbing(DataWarehouse::ScrubNone);
      else {
        d_scheduler->get_dw(1)->setScrubbing(DataWarehouse::ScrubNonPermanent);
      }
    //}
    }
    
    if (d_scheduler->getNumTaskGraphs() == 1)
      d_scheduler->execute(0, d_lastRecompileTimestep == d_sharedState->getCurrentTopLevelTimeStep() ? 0 : 1);
    else {
      subCycleExecute(currentGrid, 0, totalFine, 0, true);
    }
    
    //__________________________________
    //  If timestep has been restarted
    if(d_scheduler->get_dw(totalFine)->timestepRestarted()){
      ASSERT(restartable);
      
      // Figure out new delt
      double new_delt = d_sim->recomputeTimestep(delt);

      proc0cout << "Restarting timestep at " << t << ", changing delt from "
                << delt << " to " << new_delt << '\n';
     
      // bulletproofing
      if(new_delt < d_timeinfo->delt_min || new_delt <= 0 ){
        ostringstream warn;
        warn << "The new delT (" << new_delt << ") is either less than delT_min (" << d_timeinfo->delt_min
             << ") or equal to 0";
        throw InternalError(warn.str(), __FILE__, __LINE__);
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
        
        if(i != 0 && !d_sharedState->isLockstepAMR()){
          int trr = level->getRefinementRatioMaxDim();
          delt_fine /= trr;
          skip /= trr;
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
} // end executeTimestep()

void
AMRSimulationController::scheduleComputeStableTimestep( const GridP& grid,
                                                        SchedulerP& sched )
{
  MALLOC_TRACE_TAG_SCOPE("AMRSimulationController::scheduleComputeStableTimestep()");

  for (int i = 0; i < grid->numLevels(); i++) {
    d_sim->scheduleComputeStableTimestep(grid->getLevel(i), sched);
  }

  Task* task = scinew Task("reduceSysVar", this, &AMRSimulationController::reduceSysVar);

  //coarsenDelT task requires that delT is computed on every level, even if no tasks are 
  // run on that level.  I think this is a bug.  --Todd
  for (int i = 0; i < grid->numLevels(); i++) {
    task->requires(Task::NewDW, d_sharedState->get_delt_label(), grid->getLevel(i).get_rep());
  }

  if (d_sharedState->updateOutputInterval())
    task->requires(Task::NewDW, d_sharedState->get_outputInterval_label());

  if (d_sharedState->updateCheckpointInterval())
    task->requires(Task::NewDW, d_sharedState->get_checkpointInterval_label());
  
  //coarsen delt computes the global delt variable
  task->computes(d_sharedState->get_delt_label());
  task->setType(Task::OncePerProc);
  task->usesMPI(true);
  sched->addTask(task, d_lb->getPerProcessorPatchSet(grid), d_sharedState->allMaterials());
}

void
AMRSimulationController::reduceSysVar( const ProcessorGroup*,
                                      const PatchSubset* patches,
                                      const MaterialSubset* /*matls*/,
                                      DataWarehouse* /*old_dw*/,
                                      DataWarehouse* new_dw )
{
  MALLOC_TRACE_TAG_SCOPE("AMRSimulationController::reduceSysVar()");
  // the goal of this task is to line up the delt across all levels.  If the coarse one
  // already exists (the one without an associated level), then we must not be doing AMR
  if (patches->size() == 0 || new_dw->exists(d_sharedState->get_delt_label(), -1, 0))
    return;
  
  int multiplier = 1;
  const GridP grid = patches->get(0)->getLevel()->getGrid();
  
  for (int i = 0; i < grid->numLevels(); i++) {
    const LevelP level = grid->getLevel(i);
    
    if (i > 0 && !d_sharedState->isLockstepAMR()){
      multiplier *= level->getRefinementRatioMaxDim();
    }
        
    if (new_dw->exists(d_sharedState->get_delt_label(), -1, *level->patchesBegin())) {
      delt_vartype deltvar;
      double delt;
      new_dw->get(deltvar, d_sharedState->get_delt_label(), level.get_rep());
      
      delt = deltvar;
      new_dw->put(delt_vartype(delt*multiplier), d_sharedState->get_delt_label());
    }
  }

  if (d_myworld->size() > 1) {
    new_dw->reduceMPI(d_sharedState->get_delt_label() , 0 , 0 , -1 ) ;
    if (d_sharedState->updateOutputInterval())
      new_dw->reduceMPI(d_sharedState->get_outputInterval_label() , 0 , 0 , -1 ) ;
    if (d_sharedState->updateCheckpointInterval())
      new_dw->reduceMPI(d_sharedState->get_checkpointInterval_label() , 0 , 0 , -1 ) ;
  }
}
