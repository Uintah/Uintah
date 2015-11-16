/*
 * The MIT License
 *
 * Copyright (c) 1997-2015 The University of Utah
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

#include <CCA/Components/SimulationController/MultiScaleSimulationController.h>

#include <CCA/Ports/DataWarehouse.h>
#include <CCA/Ports/LoadBalancer.h>
#include <CCA/Ports/Output.h>
#include <CCA/Ports/ProblemSpecInterface.h>
#include <CCA/Ports/Regridder.h>
#include <CCA/Components/ReduceUda/UdaReducer.h>
#include <CCA/Components/Regridder/PerPatchVars.h>
#include <CCA/Ports/Scheduler.h>
#include <CCA/Ports/SimulationInterface.h>

#include <Core/Containers/Array3.h>
#include <Core/Exceptions/ProblemSetupException.h>
#include <Core/Geometry/IntVector.h>
#include <Core/Geometry/Vector.h>
#include <Core/Grid/Box.h>
#include <Core/Grid/Grid.h>
#include <Core/Grid/Level.h>
#include <Core/Grid/Patch.h>
#include <Core/Grid/SimulationState.h>
#include <Core/Grid/SimulationTime.h>
#include <Core/Grid/Variables/PerPatch.h>
#include <Core/Grid/Variables/ReductionVariable.h>
#include <Core/Grid/Variables/SoleVariable.h>
#include <Core/Grid/Variables/VarLabel.h>
#include <Core/Grid/Variables/VarLabelMatl.h>
#include <Core/Grid/Variables/VarTypes.h>
#include <Core/Math/MiscMath.h>
#include <Core/OS/ProcessInfo.h>
#include <Core/Parallel/Parallel.h>
#include <Core/Parallel/ProcessorGroup.h>
#include <Core/ProblemSpec/ProblemSpec.h>
#include <Core/ProblemSpec/ProblemSpecP.h>
#include <Core/Thread/Time.h>

#include <sci_defs/malloc_defs.h>
#include <sci_defs/gperftools_defs.h>

#include <iostream>
#include <iomanip>

using namespace SCIRun;
using namespace Uintah;

static DebugStream multiscaleout("MultiScale",                     false);
static DebugStream dbg(          "MultiScaleSimulationController", false);
static DebugStream dbg_barrier(  "MPIBarriers",                    false);
static DebugStream dbg_dwmem(    "LogDWMemory",                    false);
static DebugStream gprofile(     "CPUProfiler",                    false);
static DebugStream gheapprofile( "HeapProfiler",                   false);
static DebugStream gheapchecker( "HeapChecker",                    false);

double multi_scale_barrier_times[5]={0};

MultiScaleSimulationController::MultiScaleSimulationController( const ProcessorGroup * myworld,
                                                                      bool             doAMR,
                                                                      bool             doMultiScale,
                                                                      ProblemSpecP     pspec)
  : SimulationController(myworld, doAMR, doMultiScale, pspec)
  , d_multiscaleRunType(serial)
  , d_totalComponents(-1)
  , d_totalSteps(-1)
{

}

MultiScaleSimulationController::~MultiScaleSimulationController()
{

}

void
MultiScaleSimulationController::preGridSetup()
{
  ProblemSpecP multiScaleSpec = d_ups->findBlock("MultiScale");
  multiScaleSpec->require("TotalComponents",d_totalComponents);
  std::string runType;
  multiScaleSpec->require("RunType",runType);
  multiScaleSpec->require("NumberOfSteps",d_totalSteps);

  proc0cout << " Multiscale: " << std::endl << "\t"
            << d_totalComponents << " total components expected." << std::endl
            << runType << " run type. "
            << " (" << d_totalSteps << " steps.)" << std::endl;

  if (runType == "serial" || runType == "SERIAL") {
    d_multiscaleRunType = serial;
  }
  if (runType == "oscillatory" || runType == "OSCILLATORY") {
    d_multiscaleRunType = oscillatory;
  }

  SimulationController::preGridSetup();
}

//______________________________________________________________________
//
void
MultiScaleSimulationController::run()
{

#ifdef USE_GPERFTOOLS

  // CPU profiler
  if (gprofile.active()) {
    char gprofname[512];
    sprintf(gprofname, "cpuprof-rank%d", d_myworld->myrank());
    ProfilerStart(gprofname);
  }

  // Heap profiler
  if (gheapprofile.active()) {
    char gheapprofname[512];
    sprintf(gheapprofname, "heapprof-rank%d", d_myworld->myrank());
    HeapProfilerStart(gheapprofname);
  }

  // Heap checker
  HeapLeakChecker* heap_checker=NULL;
  if (gheapchecker.active()) {
    if (!gheapprofile.active()) {
      char gheapchkname[512];
      sprintf(gheapchkname, "heapchk-rank%d", d_myworld->myrank());
      heap_checker= new HeapLeakChecker(gheapchkname);
    } else {
      std::cout << "HEAPCHECKER: Cannot start with heapprofiler" << std::endl;
    }
  }

#endif

  // Sets up SimulationState (d_sharedState), output, scheduler, and timeinfo - also runs problemSetup for scheduler and output
  preGridSetup();

  // Create grid:

  GridP currentGrid = gridSetup();
  d_scheduler->initialize( 1, 1 );
  d_scheduler->advanceDataWarehouse( currentGrid, true );
  d_scheduler->setInitTimestep( true );
  
  bool first = true;
  if (d_restarting) {
    d_scheduler->setRestartInitTimestep(first);
  }

  double time;

  // set up sim, regridder, and finalize sharedState
  // also reload from the DataArchive on restart
  postGridSetup( currentGrid, time );
  calcStartTime();

  //__________________________________
  //  reduceUda
  if (d_reduceUda) {
    Dir fromDir(d_fromDir);
    d_output->reduceUdaSetup( fromDir );

    d_timeinfo->delt_factor       = 1;
    d_timeinfo->delt_min          = 0;
    d_timeinfo->delt_max          = 1e99;
    d_timeinfo->initTime          = static_cast<UdaReducer*>(d_sim)->getInitialTime();
    d_timeinfo->maxTime           = static_cast<UdaReducer*>(d_sim)->getMaxTime();
    d_timeinfo->max_delt_increase = 1e99;
    d_timeinfo->max_initial_delt  = 1e99;
  }

  // For multiscale, we may need to set up and run a number of components for the initialization step depending
  // on how we're running.

  // First we build the LevelSet for our initial components based on run type:
  std::vector<int> initializationComponentIndices;
  switch (d_multiscaleRunType) {
    // Here we simply build the index of the components which will be initialized when the overall
    // program is initialized, rather than being initialized later after a previous component has run.
    case serial:
    case oscillatory: {
      // Sequential component runs, so only the first gets passed to doInitialTimestep
      initializationComponentIndices.push_back(0);
      break;
    }
    default: {
      throw ProblemSetupException("ERROR:  Multiscale Run Type not recognized", __FILE__, __LINE__);
    }
  }
  LevelSet runningLevelSet;
  size_t numComponentsInitialized = initializationComponentIndices.size();
  for (size_t componentIndex = 0; componentIndex < numComponentsInitialized; ++componentIndex)
  {
    runningLevelSet.addAll(currentGrid->getLevelSubset(componentIndex)->getVector());
  }

  // Run the first timestep initialization only for the prescribed level sets
  doLevelSetBasedInitialTimestep(runningLevelSet, time);

//  doInitialTimestep( currentGrid, time );

  setStartSimTime( time );
  initSimulationStatsVars();

#ifndef DISABLE_SCI_MALLOC
  AllocatorSetDefaultTagLineNumber(d_sharedState->getCurrentTopLevelTimeStep());
#endif

  ////////////////////////////////////////////////////////////////////////////
  // The main time loop; here the specified problem is actually getting solved
   
  int    iterations = d_sharedState->getCurrentTopLevelTimeStep();

  double delt = 0;
   
  double start;
  
  d_lb->resetCostForecaster();

  d_scheduler->setInitTimestep(false);



  while( ( time < d_timeinfo->maxTime ) &&
	 ( iterations < d_timeinfo->maxTimestep ) && 
	 ( d_timeinfo->max_wall_time == 0 || getWallTime() < d_timeinfo->max_wall_time )  ) {

#ifdef USE_GPERFTOOLS
    if (gheapprofile.active()){
      char heapename[512];
      sprintf(heapename, "Timestep %d", iterations);
      HeapProfilerDump(heapename);
    }
#endif
     
    if (dbg_barrier.active()) {
      for (int i = 0; i < 5; i++) {
        multi_scale_barrier_times[i] = 0;
      }
    }
     
    //__________________________________
    //    Regridding
    if( d_regridder && d_regridder->doRegridOnce() && d_regridder->isAdaptive() ){
      proc0cout << "______________________________________________________________________\n";
      proc0cout << " Regridding once.\n";
      doRegridding(currentGrid, false);
      d_regridder->setAdaptivity(false);
      proc0cout << "______________________________________________________________________\n";
    }

    if (d_regridder && d_regridder->needsToReGrid(currentGrid) && (!first || (!d_restarting))) {
      doRegridding(currentGrid, false);
    }

    // Compute number of dataWarehouses - multiplies by the time refinement ratio for each level you increase
    // TODO FIXME JBH APH This logic needs to be fixed for AMR multicomponent.  11-15-2015
    int totalFine = 1;
    if (!d_doMultiScale && !d_sharedState->isLockstepAMR()) {
      for (int i = 1; i < currentGrid->numLevels(); i++) {
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

    // delt adjusted based on timeinfo parameters
    adjustDelT( delt, d_sharedState->d_prev_delt, first, time );
    newDW->override(delt_vartype(delt), d_sharedState->get_delt_label());

    if (dbg_dwmem.active()) {
      // Remember, this isn't logged if DISABLE_SCI_MALLOC is set
      // (So usually in optimized mode this will not be run.)
      d_scheduler->logMemoryUse();
      std::ostringstream fn;
      fn << "alloc." << std::setw(5) << std::setfill('0') << d_myworld->myrank() << ".out";
      std::string filename(fn.str());

#if !defined( DISABLE_SCI_MALLOC )
      DumpAllocator(DefaultAllocator(), filename.c_str());
#endif
    }
     
    if (dbg_barrier.active()) {
      start = Time::currentSeconds();
      MPI_Barrier(d_myworld->getComm());
      multi_scale_barrier_times[2] += Time::currentSeconds() - start;
    }

    // Yes, I know this is kind of hacky, but this is the only way to
    // get a new grid from UdaReducer. Needs to be done before advanceDataWarehouse.
    if (d_reduceUda){
      currentGrid = static_cast<UdaReducer*>(d_sim)->getGrid();
    }

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
    if ((nr = needRecompile(time, delt, currentGrid)) || first) {

      if (nr) {  // Recompile taskgraph, re-assign BCs, reset recompile flag.
        currentGrid->assignBCS(d_grid_ps, d_lb);
        currentGrid->performConsistencyCheck();
        d_sharedState->setRecompileTaskGraph(false);
      }

      new_init_delt = d_timeinfo->max_initial_delt;

      if (new_init_delt != old_init_delt) {
        // writes to the DW in the next section below
        delt = new_init_delt;
      }
      recompileLevelSet(time, delt, runningLevelSet, totalFine);
//      recompile(time, delt, currentGrid, totalFine);
    }
    else {
      if (d_output) {
        // This is not correct if we have switched to a different
        // component, since the delt will be wrong
        d_output->finalizeTimestep(time, delt, currentGrid, d_scheduler, 0);
        d_output->sched_allOutputTasks(delt, currentGrid, d_scheduler, 0);
      }
    }

    if (dbg_barrier.active()) {
      start = Time::currentSeconds();
      MPI_Barrier(d_myworld->getComm());
      multi_scale_barrier_times[3] += Time::currentSeconds() - start;
    }

    // adjust the delt for each level and store it in all applicable dws.

    // delT rectification is a potential mess for multiscale (concurrent simulations should rectify del_t across
    // multiple level sets.

    // TODO FIXME For now, this needs to be fixed to simply work with non-concurrent simulations, which means the
    // loop below shouldn't operate on bare grid levels, but on the current level set.
    double delt_fine = delt;
    int skip = totalFine;
    for (int i = 0; i < currentGrid->numLevels(); i++) {
      const Level* level = currentGrid->getLevel(i).get_rep();

      if (d_doAMR && i != 0 && !d_sharedState->isLockstepAMR()) {
        int rr = level->getRefinementRatioMaxDim();
        delt_fine /= rr;
        skip /= rr;
      }
       
      for (int idw = 0; idw < totalFine; idw += skip) {
        DataWarehouse* dw = d_scheduler->get_dw(idw);
        dw->override(delt_vartype(delt_fine), d_sharedState->get_delt_label(), level);
      }
    }
     
    // override for the global level as well (which only matters on dw 0)
    DataWarehouse* oldDW = d_scheduler->get_dw(0);
    oldDW->override( delt_vartype(delt), d_sharedState->get_delt_label() );

    // a component may update the output interval or the checkpoint interval
    // during a simulation.  For example in deflagration -> detonation simulations
    if (d_output && d_sharedState->updateOutputInterval() && !first) {
      min_vartype outputInv_var;
      oldDW->get(outputInv_var, d_sharedState->get_outputInterval_label());

      if (!outputInv_var.isBenignValue()) {
        d_output->updateOutputInterval(outputInv_var);
      }
    }

    if (d_output && d_sharedState->updateCheckpointInterval() && !first) {
      min_vartype checkInv_var;
      oldDW->get(checkInv_var, d_sharedState->get_checkpointInterval_label());

      if (!checkInv_var.isBenignValue()) {
        d_output->updateCheckpointInterval(checkInv_var);
      }
    }
 
    if (first) {
      first = false;
    }
     
    calcWallTime();
    printSimulationStats( d_sharedState->getCurrentTopLevelTimeStep()-1, delt, time );

    // Execute the current timestep, restarting if necessary
    d_sharedState->d_current_delt = delt;

    executeTimestep( time, delt, currentGrid, totalFine );
     
    // Print MPI statistics
    d_scheduler->printMPIStats();

    if (!first) {
      d_scheduler->setRestartInitTimestep(false);
    }

    // Update the profiler weights
    d_lb->finalizeContributions(currentGrid);
     
    if (dbg_barrier.active()) {
      start = Time::currentSeconds();
      MPI_Barrier(d_myworld->getComm());
      multi_scale_barrier_times[4] += Time::currentSeconds() - start;
      double avg[5];
      MPI_Reduce(&multi_scale_barrier_times, &avg, 5, MPI_DOUBLE, MPI_SUM, 0, d_myworld->getComm());

      if (d_myworld->myrank() == 0) {
        std::cout << "Barrier Times: ";
        for (int i = 0; i < 5; i++) {
          avg[i] /= d_myworld->size();
          std::cout << avg[i] << " ";
        }
        std::cout << "\n";
      }
    }

    if(d_output){
      d_output->findNext_OutputCheckPoint_Timestep(  delt, currentGrid );
      d_output->writeto_xml_files( delt, currentGrid );
    }

    time += delt;
  } // end while ( time is not up, etc )

  // print for the final timestep, as the one above is in the middle of a while loop - get new delt, and set walltime first
  delt_vartype delt_var;
  d_scheduler->getLastDW()->get(delt_var, d_sharedState->get_delt_label());
  delt = delt_var;
  adjustDelT( delt, d_sharedState->d_prev_delt, d_sharedState->getCurrentTopLevelTimeStep(), time );
  calcWallTime();
  printSimulationStats( d_sharedState->getCurrentTopLevelTimeStep(), delt, time );
  
  // d_ups->releaseDocument();

#ifdef USE_GPERFTOOLS
  if (gprofile.active()){
    ProfilerStop();
  }
  if (gheapprofile.active()){
    HeapProfilerStop();
  }
  if( gheapchecker.active() && !gheapprofile.active() ) {
    if( heap_checker && !heap_checker->NoLeaks() ) {
      cout << "HEAPCHECKER: MEMORY LEACK DETECTED!\n";
    }
    delete heap_checker;
  }
#endif

} // end run()

void
MultiScaleSimulationController::subCycleCompileLevelSet(  GridP   & grid
                                                        , int       startDW
                                                        , int       dwStride
                                                        , int       step
                                                        , int       levelIndex)
{
  LevelP base_level = grid->getLevel(levelIndex);

  const LevelSubset* currSubset = grid->getLevelSubset(grid->getSubsetIndex(levelIndex));
  int numLevelsInSubset = currSubset->size();
  int relativeIndexInSubset = -1;
  for (int indexInSubset = 0; indexInSubset < numLevelsInSubset; ++indexInSubset) {
    const Level* currLevel = currSubset->get(indexInSubset);
    if (levelIndex == currLevel->getIndex()) { // This is our level
      relativeIndexInSubset = indexInSubset;
    }
  }

  LevelP coarseLevel;
  int coarseStartDW;
  int coarseDWStride;
  int numCoarseSteps;  // how many steps between this level and the coarser
  int numFineSteps;    // how many steps between this level and the finer

  if (relativeIndexInSubset > 0) { // Not starting from the coarsest level of the subset, so we're recursing
    numCoarseSteps = d_sharedState->isLockstepAMR() ? 1 : base_level->getRefinementRatioMaxDim();
    coarseLevel = grid->getLevel(currSubset->get(relativeIndexInSubset-1)->getIndex());
    coarseDWStride = dwStride * numCoarseSteps;
    coarseStartDW = (startDW/coarseDWStride) * coarseDWStride;
  }
  else {  // We're currently on the coarsest level
    coarseDWStride = dwStride;
    coarseStartDW = startDW;
    numCoarseSteps = 0;
  }
  
  ASSERT(dwStride > 0 && relativeIndexInSubset < numLevelsInSubset)
//  ASSERT(dwStride > 0 && level_idx < grid->numLevels())
  d_scheduler->clearMappings();
  d_scheduler->mapDataWarehouse(Task::OldDW, startDW);
  d_scheduler->mapDataWarehouse(Task::NewDW, startDW+dwStride);
  d_scheduler->mapDataWarehouse(Task::CoarseOldDW, coarseStartDW);
  d_scheduler->mapDataWarehouse(Task::CoarseNewDW, coarseStartDW+coarseDWStride);
  
  d_sim->scheduleTimeAdvance(base_level, d_scheduler);

  if (d_doAMR) {
    if (relativeIndexInSubset + 1 < numLevelsInSubset) {
      const Level* finerLevel = currSubset->get(relativeIndexInSubset + 1);
      numFineSteps = d_sharedState->isLockstepAMR() ? 1 : finerLevel->getRefinementRatioMaxDim();
      int newStride = dwStride/numFineSteps;

      for (int substep=0; substep < numFineSteps; substep++) {
        subCycleCompile(grid, startDW+substep*newStride, newStride, substep, finerLevel->getIndex());
      }
    }
    // Coarsen and then refine CFI at the end of the W-cycle
    d_scheduler->clearMappings();
    d_scheduler->mapDataWarehouse(Task::OldDW, 0);
    d_scheduler->mapDataWarehouse(Task::NewDW, startDW+dwStride);
    d_scheduler->mapDataWarehouse(Task::CoarseOldDW, startDW);
    d_scheduler->mapDataWarehouse(Task::CoarseNewDW, startDW+dwStride);
    d_sim->scheduleCoarsen(base_level, d_scheduler);
  }


  d_scheduler->clearMappings();
  d_scheduler->mapDataWarehouse(Task::OldDW, startDW);
  d_scheduler->mapDataWarehouse(Task::NewDW, startDW+dwStride);
  d_scheduler->mapDataWarehouse(Task::CoarseOldDW, coarseStartDW);
  d_scheduler->mapDataWarehouse(Task::CoarseNewDW, coarseStartDW+coarseDWStride);

  d_sim->scheduleFinalizeTimestep(base_level, d_scheduler);

  // do refineInterface after the freshest data we can get; after the finer
  // level's coarsen completes do all the levels at this point in time as well,
  // so all the coarsens go in order, and then the refineInterfaces
  if (d_doAMR && (step < numCoarseSteps -1 || relativeIndexInSubset == 0)) {
    for (int indexInSubset = relativeIndexInSubset; indexInSubset < numLevelsInSubset; ++indexInSubset) {
      if (indexInSubset == 0) {
        continue;
      }

      if (indexInSubset == relativeIndexInSubset && relativeIndexInSubset != 0) {
        d_scheduler->mapDataWarehouse(Task::CoarseOldDW, coarseStartDW);
        d_scheduler->mapDataWarehouse(Task::CoarseNewDW, coarseStartDW + coarseDWStride);
        d_sim->scheduleRefineInterface(base_level, d_scheduler, true, true);
      }
      else {
        // look in the NewDW all the way down
        d_scheduler->mapDataWarehouse(Task::CoarseOldDW, 0);
        d_scheduler->mapDataWarehouse(Task::CoarseNewDW, startDW + dwStride);
        d_sim->scheduleRefineInterface(grid->getLevel(currSubset->get(indexInSubset)->getIndex()), d_scheduler,
                                                      false, true);
      }
    }
  }
}

//______________________________________________________________________
void
MultiScaleSimulationController::subCycleCompile( GridP & grid,
                                                 int     startDW,
                                                 int     dwStride,
                                                 int     step,
                                                 int     level_idx)
{
  LevelP base_level = grid->getLevel(level_idx);

  LevelP coarseLevel;
  int coarseStartDW;
  int coarseDWStride;
  int numCoarseSteps;  // how many steps between this level and the coarser
  int numFineSteps;    // how many steps between this level and the finer

  if (level_idx > 0) {
    numCoarseSteps = d_sharedState->isLockstepAMR() ? 1 : base_level->getRefinementRatioMaxDim();
    coarseLevel = grid->getLevel(level_idx - 1);
    coarseDWStride = dwStride * numCoarseSteps;
    coarseStartDW = (startDW / coarseDWStride) * coarseDWStride;
  }
  else {
    coarseDWStride = dwStride;
    coarseStartDW = startDW;
    numCoarseSteps = 0;
  }

  ASSERT(dwStride > 0 && level_idx < grid->numLevels())
  d_scheduler->clearMappings();
  d_scheduler->mapDataWarehouse(Task::OldDW, startDW);
  d_scheduler->mapDataWarehouse(Task::NewDW, startDW+dwStride);
  d_scheduler->mapDataWarehouse(Task::CoarseOldDW, coarseStartDW);
  d_scheduler->mapDataWarehouse(Task::CoarseNewDW, coarseStartDW+coarseDWStride);

  d_sim->scheduleTimeAdvance(base_level, d_scheduler);

  if (d_doAMR) {
    if(level_idx+1 < grid->numLevels()){
      numFineSteps = d_sharedState->isLockstepAMR() ? 1 : base_level->getFinerLevel()->getRefinementRatioMaxDim();
      int newStride = dwStride/numFineSteps;

      for(int substep=0;substep < numFineSteps;substep++){
        subCycleCompile(grid, startDW+substep*newStride, newStride, substep, level_idx+1);
      }

      // Coarsen and then refine_CFI at the end of the W-cycle
      d_scheduler->clearMappings();
      d_scheduler->mapDataWarehouse(Task::OldDW, 0);
      d_scheduler->mapDataWarehouse(Task::NewDW, startDW+dwStride);
      d_scheduler->mapDataWarehouse(Task::CoarseOldDW, startDW);
      d_scheduler->mapDataWarehouse(Task::CoarseNewDW, startDW+dwStride);
      d_sim->scheduleCoarsen(base_level, d_scheduler);
    }
  }

  d_scheduler->clearMappings();
  d_scheduler->mapDataWarehouse(Task::OldDW, startDW);
  d_scheduler->mapDataWarehouse(Task::NewDW, startDW+dwStride);
  d_scheduler->mapDataWarehouse(Task::CoarseOldDW, coarseStartDW);
  d_scheduler->mapDataWarehouse(Task::CoarseNewDW, coarseStartDW+coarseDWStride);

  d_sim->scheduleFinalizeTimestep(base_level, d_scheduler);

  // do refineInterface after the freshest data we can get; after the finer
  // level's coarsen completes do all the levels at this point in time as well,
  // so all the coarsens go in order, and then the refineInterfaces
  if (d_doAMR && (step < numCoarseSteps -1 || level_idx == 0)) {

    for (int i = base_level->getIndex(); i < base_level->getGrid()->numLevels(); i++) {

      if (i == 0) {
        continue;
      }

      if (i == base_level->getIndex() && level_idx != 0) {
        d_scheduler->mapDataWarehouse(Task::CoarseOldDW, coarseStartDW);
        d_scheduler->mapDataWarehouse(Task::CoarseNewDW, coarseStartDW + coarseDWStride);
        d_sim->scheduleRefineInterface(base_level, d_scheduler, true, true);
      }
      else {
        // look in the NewDW all the way down
        d_scheduler->mapDataWarehouse(Task::CoarseOldDW, 0);
        d_scheduler->mapDataWarehouse(Task::CoarseNewDW, startDW + dwStride);
        d_sim->scheduleRefineInterface(base_level->getGrid()->getLevel(i), d_scheduler, false, true);
      }
    }
  }
}
//______________________________________________________________________
//
void
MultiScaleSimulationController::subCycleExecute( GridP & grid,
                                                 int     startDW,
                                                 int     dwStride,
                                                 int     levelNum,
                                                 bool    rootCycle )
{
  // there are 2n+1 taskgraphs, n for the basic timestep, n for intermediate timestep work,
  // and 1 for the errorEstimate and stableTimestep, where n is the number of levels.
  if (multiscaleout.active()) {
    multiscaleout << "Start MultiScaleSimulationController::subCycleExecute, level=" << grid->numLevels() << '\n';
  }
  // We are on (the fine) level numLevel
  int numSteps;
  if (levelNum == 0 || d_sharedState->isLockstepAMR()) {
    numSteps = 1;
  }
  else {
    numSteps = grid->getLevel(levelNum)->getRefinementRatioMaxDim();
  }

  int newDWStride = dwStride / numSteps;

  DataWarehouse::ScrubMode oldScrubbing =
      (/*d_lb->isDynamic() ||*/d_sim->restartableTimesteps()) ? DataWarehouse::ScrubNonPermanent : DataWarehouse::ScrubComplete;

  int curDW = startDW;
  for (int step = 0; step < numSteps; step++) {

    if (step > 0) {
      curDW += newDWStride;  // can't increment at the end, or the FINAL tg for L0 will use the wrong DWs
    }

    d_scheduler->clearMappings();
    d_scheduler->mapDataWarehouse(Task::OldDW, curDW);
    d_scheduler->mapDataWarehouse(Task::NewDW, curDW + newDWStride);
    d_scheduler->mapDataWarehouse(Task::CoarseOldDW, startDW);
    d_scheduler->mapDataWarehouse(Task::CoarseNewDW, startDW + dwStride);

    // we really only need to pass in whether the current DW is mapped to 0 or not
    // TODO - fix inter-Taskgraph scrubbing
    d_scheduler->get_dw(curDW)->setScrubbing(oldScrubbing);  // OldDW
    d_scheduler->get_dw(curDW + newDWStride)->setScrubbing(DataWarehouse::ScrubNonPermanent);  // NewDW
    d_scheduler->get_dw(startDW)->setScrubbing(oldScrubbing);  // CoarseOldDW
    d_scheduler->get_dw(startDW + dwStride)->setScrubbing(DataWarehouse::ScrubNonPermanent);  // CoarseNewDW

    // we need to unfinalize because execute finalizes all new DWs, and we need to write into them still
    // (even if we finalized only the NewDW in execute, we will still need to write into that DW)
    d_scheduler->get_dw(curDW + newDWStride)->unfinalize();

    // iteration only matters if it's zero or greater than 0
    int iteration = curDW + (d_lastRecompileTimestep == d_sharedState->getCurrentTopLevelTimeStep() ? 0 : 1);

    if (dbg.active()) {
      dbg << "Rank-" << d_myworld->myrank() << "   Executing TG on level " << levelNum << " with old DW " << curDW << "="
          << d_scheduler->get_dw(curDW)->getID() << " and new " << curDW + newDWStride << "="
          << d_scheduler->get_dw(curDW + newDWStride)->getID() << "CO-DW: " << startDW << " CNDW " << startDW + dwStride
          << " on iteration " << iteration << std::endl;
    }

    d_scheduler->execute(levelNum, iteration);

    if (levelNum + 1 < grid->numLevels()) {
      ASSERT(newDWStride > 0);
      subCycleExecute(grid, curDW, newDWStride, levelNum + 1, false);
    }

    if (d_doAMR && grid->numLevels() > 1 && (step < numSteps - 1 || levelNum == 0)) {
      // Since the execute of the intermediate is time-based, execute the intermediate TG relevant
      // to this level, if we are in the middle of the subcycle or at the end of level 0.
      // the end of the cycle will be taken care of by the parent level sybcycle
      d_scheduler->clearMappings();
      d_scheduler->mapDataWarehouse(Task::OldDW, curDW);
      d_scheduler->mapDataWarehouse(Task::NewDW, curDW + newDWStride);
      d_scheduler->mapDataWarehouse(Task::CoarseOldDW, startDW);
      d_scheduler->mapDataWarehouse(Task::CoarseNewDW, startDW + dwStride);

      d_scheduler->get_dw(curDW)->setScrubbing(oldScrubbing);  // OldDW
      d_scheduler->get_dw(curDW + newDWStride)->setScrubbing(DataWarehouse::ScrubNonPermanent);  // NewDW
      d_scheduler->get_dw(startDW)->setScrubbing(oldScrubbing);  // CoarseOldDW
      d_scheduler->get_dw(startDW + dwStride)->setScrubbing(DataWarehouse::ScrubNonPermanent);  // CoarseNewDW

      if (dbg.active()) {
        dbg << "Rank-" << d_myworld->myrank() << "   Executing INT TG on level " << levelNum << " with old DW " << curDW << "="
            << d_scheduler->get_dw(curDW)->getID() << " and new " << curDW + newDWStride << "="
            << d_scheduler->get_dw(curDW + newDWStride)->getID() << " CO-DW: " << startDW << " CNDW " << startDW + dwStride
            << " on iteration " << iteration << std::endl;
      }

      d_scheduler->get_dw(curDW + newDWStride)->unfinalize();
      d_scheduler->execute(levelNum + grid->numLevels(), iteration);
    }

    if (curDW % dwStride != 0) {
      //the currentDW(old datawarehouse) should no longer be needed - in the case of NonPermanent OldDW scrubbing
      d_scheduler->get_dw(curDW)->clear();
    }

  }
  if (levelNum == 0) {
    // execute the final TG
    if (dbg.active())
      dbg << "Rank-" << d_myworld->myrank() << "   Executing Final TG on level " << levelNum << " with old DW " << curDW << " = "
          << d_scheduler->get_dw(curDW)->getID() << " and new " << curDW + newDWStride << " = "
          << d_scheduler->get_dw(curDW + newDWStride)->getID() << std::endl;
    d_scheduler->get_dw(curDW + newDWStride)->unfinalize();
    d_scheduler->execute(d_scheduler->getNumTaskGraphs() - 1, 1);
  }
}  // end subCycleExecute()

//______________________________________________________________________
bool
MultiScaleSimulationController::needRecompile(       double   time,
                                                     double   delt,
                                               const GridP  & grid )
{
  // Currently, d_output, d_sim, d_lb, d_regridder can request a recompile.  --bryan
  bool recompile = false;
  
  // do it this way so everybody can have a chance to maintain their state
  recompile |= ( d_output && d_output->needRecompile(time, delt, grid));
  recompile |= ( d_sim    && d_sim->needRecompile(   time, delt, grid));
  recompile |= ( d_lb     && d_lb->needRecompile(    time, delt, grid));
  recompile |= ( d_sharedState->getRecompileTaskGraph() );
  
  return recompile;
}

void
MultiScaleSimulationController::doLevelSetBasedInitialTimestep(  const LevelSet & initializationSets
                                                               ,       double   & time
                                                              )
{
  // FIXME Look here for current levelSet work FIXME FIXME FIXME TODO TODO TODO
  double start = Time::currentSeconds();
  // All levels in the included levelSets are assumed to want/need a combined dataWarehouse
  d_scheduler->mapDataWarehouse(Task::OldDW, 0);
  d_scheduler->mapDataWarehouse(Task::NewDW, 1);
  d_scheduler->mapDataWarehouse(Task::CoarseOldDW, 0);
  d_scheduler->mapDataWarehouse(Task::CoarseNewDW, 1);

  if (d_restarting) {
    // Not implemented yet
    throw ProblemSetupException("ERROR:  LevelSet based restarts are not yet implemented!", __FILE__, __LINE__);
  }
  else {
    d_sharedState->setCurrentTopLevelTimeStep( 0 );
    // for dynamic lb's, set up initial patch config
    // FIXME JBH APH 11-14-2015  Load balancers don't actually balance on grids.. they balance on levels.
    //   This means that EVERYTHING in a load balancer which eats a grid and then does stuff like loop over the
    //   levels of the grid ought to -actually- parse levelSets and loop over only the levels of the level set.
    // For right now we'll just assume the load balancer is null and not call the reallocate
    d_lb->possiblyDynamicallyReallocate(initializationSets, LoadBalancer::init);
    GridP grid = initializationSets.getSubset(0)->get(0)->getGrid();
    grid->assignBCS( initializationSets, d_grid_ps, d_lb);
    grid->performConsistencyCheck(initializationSets);
    time = d_timeinfo->initTime;

    bool needNewLevel = false;
    do {
      if (needNewLevel) {
        d_scheduler->initialize(1,1);
        d_scheduler->advanceDataWarehouse(grid, true);
      }

      proc0cout << "Compiling initialization taskgraph based on levelSets..." << std::endl;
      size_t numSubsets = initializationSets.size();
      for (size_t subsetIndex = 0; subsetIndex < numSubsets; ++subsetIndex) {
        // Verify that patches on a single level do not overlap
        const LevelSubset* currLevelSubset = initializationSets.getSubset(subsetIndex);
        size_t numLevels = currLevelSubset->size();
        proc0cout << "Seeing " << numLevels << " level(s) in subset " << subsetIndex << std::endl;
        for (size_t indexInSubset = numLevels; indexInSubset > 0; --indexInSubset) {
          proc0cout << " Current level index: " << indexInSubset-1 << " numLevels: " << numLevels << std::endl;
          LevelP levelHandle = grid->getLevel(currLevelSubset->get(indexInSubset-1)->getIndex());
          d_sim->scheduleInitialize(levelHandle, d_scheduler);
          if (d_regridder) {
            // so we can initially regrid
            d_regridder->scheduleInitializeErrorEstimate(levelHandle);
            d_sim->scheduleInitialErrorEstimate(levelHandle, d_scheduler);

            // Dilate only within a single level subset for now; will worry about dilating to accommodate
            // interface layers at a later time.  TODO FIXME JBH APH 11-14-2015
            if (indexInSubset < d_regridder->maxLevels()) {
              d_regridder->scheduleDilation(levelHandle);
            }
          }
        }
      }
      // This should probably be per level set? JBH APH FIXME TODO
      scheduleComputeStableTimestep(initializationSets, d_scheduler);

      //FIXME TODO JBH APH 11-14-2013
      // I believe d_output should actually be operating on the grid, as should
      // sched_allOutputTasks.  Even if we're not currently simulating on all grid levels, we probably
      // ought to output any data on those levels that's hanging around for switching to other components.
      // This does imply, of course, that components should be good about getting rid of excess data they no
      // longer need.
      if (d_output) {
        double delT = 0;
        bool   recompile = true;
        d_output->finalizeTimestep(time, delT, grid, d_scheduler, recompile);
        d_output->sched_allOutputTasks(delT, grid, d_scheduler, recompile);
      }

      d_scheduler->compile(&initializationSets);
      double end = Time::currentSeconds() - start;

      proc0cout << "Done initialization taskgraph based on levelSets (" << end << " seconds)" << std::endl;

      // No scrubbing for initial step
      // FIXME TODO JBH APH 11-14-2015 We need to be careful about scrubbing; we can't afford to scrub a switched
      // out component's data when we shouldn't.
      d_scheduler->get_dw(1)->setScrubbing(DataWarehouse::ScrubNone);
      d_scheduler->execute();

      // FIXME FIXME FIXME FIXME TODO TODO TODO TODO
      // FIXME TODO JBH APH 11-14-2015
      // THIS IS NOT RIGHT.  We should be checking to make sure the number of levels in the subset (currSubset->size()
      // is less than d_regridder->maxLevels;  the problem is it's unclear that this should be inside the subset loop.)
      // Should the whole "needNewLevel" construct be getting checked inside the levelSubset loop?!?
      // URGENT FIXME TODO URGENT
      needNewLevel = d_regridder && d_regridder->isAdaptive() && grid->numLevels() < d_regridder->maxLevels()
          && doRegridding(grid, true);

    }
    while (needNewLevel);

    // FIXME TODO JBH APH 11-14-2015
    // Again, output -should- be okay to work on whole grid.  However, double check this.
    if (d_output) {
      d_output->findNext_OutputCheckPoint_Timestep(0, grid);
      d_output->writeto_xml_files(0, grid);
    }

  }

}
//______________________________________________________________________
void
MultiScaleSimulationController::doInitialTimestep(  GridP  & grid
                                                  , double & t
                                                 )
{
  double start = Time::currentSeconds();
  d_scheduler->mapDataWarehouse(Task::OldDW, 0);
  d_scheduler->mapDataWarehouse(Task::NewDW, 1);
  d_scheduler->mapDataWarehouse(Task::CoarseOldDW, 0);
  d_scheduler->mapDataWarehouse(Task::CoarseNewDW, 1);
  
  if(d_restarting) {
    d_lb->possiblyDynamicallyReallocate(grid, LoadBalancer::restart);
    // tsaad & bisaac: At this point, during a restart, a grid does NOT have knowledge of the boundary conditions.
    // (See other comments in SimulationController.cc for why that is the case). Here, and given a
    // legitimate load balancer, we can assign the BCs to the grid in an efficient manner.
    grid->assignBCS( d_grid_ps, d_lb );
  
    grid->performConsistencyCheck();
  
    d_sim->restartInitialize();

    for (int i = grid->numLevels() - 1; i >= 0; i--) {
      d_sim->scheduleRestartInitialize(grid->getLevel(i), d_scheduler);
    }
    d_scheduler->compile();
    d_scheduler->get_dw(1)->setScrubbing(DataWarehouse::ScrubNone);
    d_scheduler->execute();

    // Now we know we're done with any additions to the new DW - finalize it
    d_scheduler->get_dw(1)->finalize();
  
    if (d_regridder && d_regridder->isAdaptive()) {
      // On restart:
      //   we must set up the tasks (but not compile) so we can have the
      //   initial OldDW Requirements in order to regrid straightaway
      for(int i=grid->numLevels()-1; i >= 0; i--) {
        d_sim->scheduleTimeAdvance(grid->getLevel(i), d_scheduler);
      }
    }  
  }
  else {
    d_sharedState->setCurrentTopLevelTimeStep( 0 );
    // for dynamic lb's, set up initial patch config
    d_lb->possiblyDynamicallyReallocate(grid, LoadBalancer::init); 
    grid->assignBCS( d_grid_ps, d_lb );
    grid->performConsistencyCheck();
    t = d_timeinfo->initTime;

    bool needNewLevel = false;
    do {
      if (needNewLevel) {
        d_scheduler->initialize(1, 1);
        d_scheduler->advanceDataWarehouse(grid, true);
      }

      proc0cout << "Compiling initialization taskgraph...\n";
      for (int i = grid->numLevels() - 1; i >= 0; i--) {
        d_sim->scheduleInitialize(grid->getLevel(i), d_scheduler);

        if (d_regridder) {
          // so we can initially regrid
          d_regridder->scheduleInitializeErrorEstimate(grid->getLevel(i));
          d_sim->scheduleInitialErrorEstimate(grid->getLevel(i), d_scheduler);

          if (i < d_regridder->maxLevels() - 1) {  // we don't use error estimates if we don't make another level, so don't dilate
            d_regridder->scheduleDilation(grid->getLevel(i));
          }
        }
      }
      scheduleComputeStableTimestep(grid, d_scheduler);

      if (d_output) {
        double delT = 0;
        bool recompile = true;
        d_output->finalizeTimestep(t, delT, grid, d_scheduler, recompile);
        d_output->sched_allOutputTasks(delT, grid, d_scheduler, recompile);
      }

      d_scheduler->compile();
      double end = Time::currentSeconds() - start;

      proc0cout << "done taskgraph compile (" << end << " seconds)\n";
      // No scrubbing for initial step
      d_scheduler->get_dw(1)->setScrubbing(DataWarehouse::ScrubNone);
      d_scheduler->execute();

      needNewLevel = d_regridder && d_regridder->isAdaptive() && grid->numLevels() < d_regridder->maxLevels()
          && doRegridding(grid, true);
    }
    while (needNewLevel);

    if (d_output) {
      d_output->findNext_OutputCheckPoint_Timestep(0, grid);
      d_output->writeto_xml_files(0, grid);
    }
  }
} // end doInitialTimestep()

//______________________________________________________________________

bool
MultiScaleSimulationController::doRegridding( GridP & currentGrid,
                                              bool    initialTimestep )
{
  double start = Time::currentSeconds();

  GridP oldGrid = currentGrid;
  currentGrid = d_regridder->regrid(oldGrid.get_rep());
  
  if (dbg_barrier.active()) {
    double start;
    start = Time::currentSeconds();
    MPI_Barrier(d_myworld->getComm());
    multi_scale_barrier_times[0] += Time::currentSeconds() - start;
  }
  
  double regridTime = Time::currentSeconds() - start;
  d_sharedState->regriddingTime += regridTime;
  d_sharedState->setRegridTimestep(false);

  int lbstate = initialTimestep ? LoadBalancer::init : LoadBalancer::regrid;

  if (currentGrid != oldGrid) {
    d_sharedState->setRegridTimestep(true);
     
    d_lb->possiblyDynamicallyReallocate(currentGrid, lbstate); 
    if(dbg_barrier.active()) {
      double start;
      start=Time::currentSeconds();
      MPI_Barrier(d_myworld->getComm());
      multi_scale_barrier_times[1]+=Time::currentSeconds()-start;
    }
    
    currentGrid->assignBCS( d_grid_ps, d_lb );
    currentGrid->performConsistencyCheck();

    //__________________________________
    //  output regridding stats
    if (d_myworld->myrank() == 0) {
      std::cout << "  REGRIDDING:";
     
      //amrout << "---------- OLD GRID ----------" << endl << *(oldGrid.get_rep());
      for (int i = 0; i < currentGrid->numLevels(); i++) {
        std::cout << " Level " << i << " has " << currentGrid->getLevel(i)->numPatches() << " patches...";
      }
      std::cout << std::endl;
      
      if (multiscaleout.active()) {
        multiscaleout << "---------- NEW GRID ----------" << std::endl;
        multiscaleout << "Grid has " << currentGrid->numLevels() << " level(s)" << std::endl;
      
        for ( int levelIndex = 0; levelIndex < currentGrid->numLevels(); levelIndex++ ) {
          LevelP level = currentGrid->getLevel( levelIndex );
          
          multiscaleout << "  Level " << level->getID()
                 << ", indx: "<< level->getIndex()
                 << " has " << level->numPatches() << " patch(es)" << std::endl;
            
          for ( Level::patchIterator patchIter = level->patchesBegin(); patchIter < level->patchesEnd(); patchIter++ ) {
            const Patch* patch = *patchIter;
            multiscaleout << "(Patch " << patch->getID() << " proc " << d_lb->getPatchwiseProcessorAssignment(patch)
                   << ": box=" << patch->getExtraBox()
                   << ", lowIndex=" << patch->getExtraCellLowIndex() << ", highIndex="
                   << patch->getExtraCellHighIndex() << ")" << std::endl;
          }
        }
      }
    }  // rank 0

    double scheduleTime = Time::currentSeconds();
    
    if (!initialTimestep) {
      d_scheduler->scheduleAndDoDataCopy(currentGrid, d_sim);
    }
    
    scheduleTime = Time::currentSeconds() - scheduleTime;

    double time = Time::currentSeconds() - start;
    
    if (d_myworld->myrank() == 0) {
      std::cout << "done regridding (" << time << " seconds, regridding took " << regridTime;

      if (!initialTimestep) {
        std::cout << ", scheduling and copying took " << scheduleTime << ")";
      }
      std::cout << std::endl;
    }
    return true;
  }  // grid != oldGrid
  return false;
}

void
MultiScaleSimulationController::recompileLevelSet(        double    time
                                                  ,       double    del_t
                                                  , const LevelSet& currentLevelSet
                                                  ,       int       totalFine
                                                 )
{
  proc0cout << "Recompiling taskgraph with level sets ..." << std::endl;

  d_lastRecompileTimestep = d_sharedState->getCurrentTopLevelTimeStep();
  double start = Time::currentSeconds();

  d_scheduler->initialize(1, totalFine);
  // FIXME TODO JBH APH We may need to look at fillDataWarehouses to levelSet-ize it.  11-15-2015
  GridP currentGrid = currentLevelSet.getSubset(0)->get(0)->getGrid();
  d_scheduler->fillDataWarehouses(currentGrid);

  // Set up new DWs, DW mappings.
  d_scheduler->clearMappings();
  d_scheduler->mapDataWarehouse(Task::OldDW, 0);
  d_scheduler->mapDataWarehouse(Task::NewDW, totalFine);
  d_scheduler->mapDataWarehouse(Task::CoarseOldDW, 0);
  d_scheduler->mapDataWarehouse(Task::CoarseNewDW, totalFine);

  // FIXME TODO JBH APH What are we doing here?  SchedulteTimeAdvance if the component is not AMR?  If so the logic
  // should actually check to see if the current levelSet has an AMR level, and if so it should probably
  // call subcycle compile on the AMR level subset(s), and call scheduleTimeAdvance on any non-amr subsets in the levelSet

  // Not sure if the check against d_doMultiScale && d_sharedState->getCurrentTopLevelTimeStep() > 1 is still necessary.
  // Feels like this was a hack to get around trying to scheduleTimeAdvance on the initial recompile because we'd have the
  // multiscale subgrid processes scheduling when they hadn't been initialized, but this should now be handled by proper
  // passing in of the current levelSet in the first place for the process being run.  We'll leave it in for now, see what
  // breaks.  11-15-2015 JBH APH FIXME TODO FIXME TODO FIXME
  if (d_doMultiScale && d_sharedState->getCurrentTopLevelTimeStep() > 1) {
    int numSubsets = currentLevelSet.size();
    for (int subsetIndex = 0; subsetIndex < numSubsets; ++subsetIndex) {
      const LevelSubset* currentSubset = currentLevelSet.getSubset(subsetIndex);
      int levelsInSubset = currentSubset->size();
      if (!currentSubset->get(0)->isAMR()) {
        // FIXME TODO JBH APH We should explicitly make sure an entire subset has levels which are/are not AMR.
        // Presumably if the first level in a subset is not AMR, the subset itself is not AMR.
        for (int indexInSubset = 0; indexInSubset < levelsInSubset; ++indexInSubset) {
          const LevelP levelHandle=currentGrid->getLevel(currentSubset->get(indexInSubset)->getIndex());
          d_sim->scheduleTimeAdvance(levelHandle, d_scheduler);
        }
      }
      else {
        // subset IS AMR; we need to schedule via subCycleCompile
        subCycleCompileLevelSet(currentGrid, 0, totalFine, 0, currentSubset->get(0)->getIndex());
      }
    }
  }

  d_scheduler->clearMappings();
  d_scheduler->mapDataWarehouse(Task::OldDW, 0);
  d_scheduler->mapDataWarehouse(Task::NewDW, totalFine);

  size_t numSubsets = currentLevelSet.size();
  for (size_t subsetIndex = 0; subsetIndex < numSubsets; ++subsetIndex) {
    // Verify that patches on a single level do not overlap
    const LevelSubset* currLevelSubset = currentLevelSet.getSubset(subsetIndex);
    size_t numLevels = currLevelSubset->size();
    proc0cout << "Seeing " << numLevels << " levels in subset " << subsetIndex << std::endl;
    for (size_t indexInSubset = numLevels; indexInSubset > 0; --indexInSubset) {
      proc0cout << " Current level index: " << indexInSubset-1 << " numLevels: " << numLevels << std::endl;
      LevelP levelHandle = currentGrid->getLevel(currLevelSubset->get(indexInSubset-1)->getIndex());
      if (d_regridder) {
        // so we can initially regrid
        d_regridder->scheduleInitializeErrorEstimate(levelHandle);
        d_sim->scheduleInitialErrorEstimate(levelHandle, d_scheduler);

        // Dilate only within a single level subset for now; will worry about dilating to accomodate
        // interface layers at a later time.  TODO FIXME JBH APH 11-14-2015
        if (indexInSubset < d_regridder->maxLevels()) {
          d_regridder->scheduleDilation(levelHandle);
        }
      }
    }
  }

  scheduleComputeStableTimestep(currentLevelSet, d_scheduler);

  if(d_output){
    d_output->finalizeTimestep(time, del_t, currentGrid, d_scheduler, true);
    d_output->sched_allOutputTasks( del_t, currentGrid, d_scheduler, true );
  }

  d_scheduler->compile(&currentLevelSet);

  double dt=Time::currentSeconds() - start;

  proc0cout << "DONE TASKGRAPH RE-COMPILE (" << dt << " seconds)\n";
  d_sharedState->compilationTime += dt;
} // end routine

//______________________________________________________________________
void
MultiScaleSimulationController::recompile( double  t,
                                           double  delt,
                                           GridP & currentGrid,
                                           int     totalFine )
{
  proc0cout << "Compiling taskgraph...\n";

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

  if (d_doMultiScale && d_sharedState->getCurrentTopLevelTimeStep() > 1) {
    for (int level_idx = currentGrid->numLevels() - 1; level_idx >= 0; --level_idx) {
      const LevelP current_level = currentGrid->getLevel(level_idx);
      d_sim->scheduleTimeAdvance(current_level, d_scheduler);
    }
  } else {
    subCycleCompile(currentGrid, 0, totalFine, 0, 0);
  }

  d_scheduler->clearMappings();
  d_scheduler->mapDataWarehouse(Task::OldDW, 0);
  d_scheduler->mapDataWarehouse(Task::NewDW, totalFine);
    
  for (int i = currentGrid->numLevels() - 1; i >= 0; i--) {
    if (d_regridder) {
      d_regridder->scheduleInitializeErrorEstimate(currentGrid->getLevel(i));
      d_sim->scheduleErrorEstimate(currentGrid->getLevel(i), d_scheduler);

      if (i < d_regridder->maxLevels() - 1) {  // we don't use error estimates if we don't make another level, so don't dilate
        d_regridder->scheduleDilation(currentGrid->getLevel(i));
      }
    }
  }

  scheduleComputeStableTimestep(currentGrid, d_scheduler);

  if(d_output){
    d_output->finalizeTimestep(t, delt, currentGrid, d_scheduler, true);
    d_output->sched_allOutputTasks( delt, currentGrid, d_scheduler, true );
  }
  
  d_scheduler->compile();
 
  double dt=Time::currentSeconds() - start;

  proc0cout << "DONE TASKGRAPH RE-COMPILE (" << dt << " seconds)\n";
  d_sharedState->compilationTime += dt;
}
//______________________________________________________________________
void
MultiScaleSimulationController::executeTimestep( double   t,
                                                 double & delt,
                                                 GridP  & currentGrid,
                                                 int      totalFine )
{
  // If the timestep needs to be restarted, this loop will execute multiple times.
  bool success = true;
  double orig_delt = delt;
  do {
    bool restartable = d_sim->restartableTimesteps();
    d_scheduler->setRestartable(restartable);

    if (restartable) {
      d_scheduler->get_dw(0)->setScrubbing(DataWarehouse::ScrubNonPermanent);
    }
    else {
      d_scheduler->get_dw(0)->setScrubbing(DataWarehouse::ScrubComplete);
    }

    for (int i = 0; i <= totalFine; i++) {
      // getNthProc requires the variables after they would have been scrubbed
      if (d_lb->getNthProc() > 1) {
        d_scheduler->get_dw(i)->setScrubbing(DataWarehouse::ScrubNone);
      }
      else {
        d_scheduler->get_dw(1)->setScrubbing(DataWarehouse::ScrubNonPermanent);
      }
    }

    if (d_scheduler->getNumTaskGraphs() == 1) {
      d_scheduler->execute(0, d_lastRecompileTimestep == d_sharedState->getCurrentTopLevelTimeStep() ? 0 : 1);
    }
    else {
      subCycleExecute(currentGrid, 0, totalFine, 0, true);
    }

    //__________________________________
    //  If timestep has been restarted
    if (d_scheduler->get_dw(totalFine)->timestepRestarted()) {
      ASSERT(restartable);

      // Figure out new delt
      double new_delt = d_sim->recomputeTimestep(delt);

      proc0cout << "Restarting timestep at " << t << ", changing delt from " << delt << " to " << new_delt << '\n';

      // bulletproofing
      if (new_delt < d_timeinfo->delt_min || new_delt <= 0) {
        std::ostringstream warn;
        warn << "The new delT (" << new_delt << ") is either less than delT_min (" << d_timeinfo->delt_min << ") or equal to 0";
        throw InternalError(warn.str(), __FILE__, __LINE__);
      }

      d_output->reEvaluateOutputTimestep(orig_delt, new_delt);
      delt = new_delt;

      d_scheduler->get_dw(0)->override(delt_vartype(new_delt), d_sharedState->get_delt_label());

      for (int i = 1; i <= totalFine; i++) {
        d_scheduler->replaceDataWarehouse(i, currentGrid);
      }

      double delt_fine = delt;
      int skip = totalFine;
      for (int i = 0; i < currentGrid->numLevels(); i++) {
        const Level* level = currentGrid->getLevel(i).get_rep();

        if (i != 0 && !d_sharedState->isLockstepAMR()) {
          int trr = level->getRefinementRatioMaxDim();
          delt_fine /= trr;
          skip /= trr;
        }

        for (int idw = 0; idw < totalFine; idw += skip) {
          DataWarehouse* dw = d_scheduler->get_dw(idw);
          dw->override(delt_vartype(delt_fine), d_sharedState->get_delt_label(), level);
        }
      }
      success = false;

    }
    else {
      success = true;
      if (d_scheduler->get_dw(1)->timestepAborted()) {
        throw InternalError("Execution aborted, cannot restart timestep\n", __FILE__, __LINE__);
      }
    }
  }
  while (!success);
}  // end executeTimestep()

//______________________________________________________________________
//

void
MultiScaleSimulationController::scheduleComputeStableTimestep( const LevelSet     & operatingLevels
                                                              ,      SchedulerP   & sched
                                                             )
{
  size_t numSubsets = operatingLevels.size();
  GridP  grid = operatingLevels.getSubset(0)->get(0)->getGrid();

  // Compute stable timesteps across all dimulation components on all current level subsets
  for (size_t subsetIndex = 0; subsetIndex < numSubsets; ++subsetIndex) {
    const LevelSubset* currLevelSubset = operatingLevels.getSubset(subsetIndex);
    size_t numLevels = currLevelSubset->size();
    for (size_t indexInSubset = 0; indexInSubset < numLevels; ++indexInSubset) {
      LevelP levelHandle = grid->getLevel(currLevelSubset->get(indexInSubset)->getIndex());
      d_sim->scheduleComputeStableTimestep(levelHandle, sched);
    }
  }

  // Schedule timestep reduction to determine real timestep for all components currently running
  Task* task = scinew Task("reduceSysVarLevelSet", this, &MultiScaleSimulationController::reduceSysVarLevelSet);

  // Add requirements for delT for each level
  for (size_t subsetIndex = 0; subsetIndex < numSubsets; ++subsetIndex) {
    const LevelSubset* currLevelSubset = operatingLevels.getSubset(subsetIndex);
    size_t numLevels = currLevelSubset->size();
    for (size_t indexInSubset = 0; indexInSubset < numLevels; ++indexInSubset) {
      LevelP levelHandle = grid->getLevel(currLevelSubset->get(indexInSubset)->getIndex());
      task->requires(Task::NewDW, d_sharedState->get_delt_label(), levelHandle.get_rep());
    }
  }

  if (d_sharedState->updateOutputInterval()) {
    task->requires(Task::NewDW, d_sharedState->get_outputInterval_label());
  }

  if (d_sharedState->updateCheckpointInterval()) {
    task->requires(Task::NewDW, d_sharedState->get_checkpointInterval_label());
  }

  //Coarsen delT computes the global delT variable
  task->computes(d_sharedState->get_delt_label());
  task->setType(Task::OncePerProc);
  task->usesMPI(true);

  sched->addTask(task, d_lb->getPerProcessorPatchSet(operatingLevels), d_sharedState->allMaterials());

}

void
MultiScaleSimulationController::scheduleComputeStableTimestep( const GridP      & grid,
                                                                     SchedulerP & sched )
{

  for (int i = 0; i < grid->numLevels(); i++) {
    d_sim->scheduleComputeStableTimestep(grid->getLevel(i), sched);
  }

  Task* task = scinew Task("reduceSysVar", this, &MultiScaleSimulationController::reduceSysVar);

  //coarsenDelT task requires that delT is computed on every level, even if no tasks are 
  // run on that level.  I think this is a bug.  --Todd
  for (int i = 0; i < grid->numLevels(); i++) {
    task->requires(Task::NewDW, d_sharedState->get_delt_label(), grid->getLevel(i).get_rep());
  }

  if (d_sharedState->updateOutputInterval()){
    task->requires(Task::NewDW, d_sharedState->get_outputInterval_label());
  }
  
  if (d_sharedState->updateCheckpointInterval()){
    task->requires(Task::NewDW, d_sharedState->get_checkpointInterval_label());
  }
  
  //coarsen delt computes the global delt variable
  task->computes(d_sharedState->get_delt_label());
  task->setType(Task::OncePerProc);
  task->usesMPI(true);
  sched->addTask(task, d_lb->getPerProcessorPatchSet(grid), d_sharedState->allMaterials());
}

void
MultiScaleSimulationController::reduceSysVarLevelSet(  const ProcessorGroup * /*pg*/
                                                     , const PatchSubset    *   patches
                                                     , const MaterialSubset * /*matls*/
                                                     ,       DataWarehouse  * /*oldDW*/
                                                     ,       DataWarehouse  *   newDW
                                                    ) {

  if (patches->size() != 0 && !newDW->exists(d_sharedState->get_delt_label(), -1, 0)) {
    int multiplier = 1;
    int levelIndex;
    levelIndex = patches->get(0)->getLevel()->getIndex();
    const GridP grid = patches->get(0)->getLevel()->getGrid();
    int subsetIndex = grid->getSubsetIndex(levelIndex);
    const LevelSubset* currSubset = grid->getLevelSubset(subsetIndex);
    size_t levelsInSubset = currSubset->size();

    for (size_t indexInSubset = 0; indexInSubset < levelsInSubset; ++indexInSubset) {
      int currLevelIndex = currSubset->get(indexInSubset)->getIndex();
      const LevelP levelHandle = grid->getLevel(currLevelIndex);

      // TODO FIXME Right now I assume that AMR is only lockstep with other AMR processes in the same subset.
      // If we need to do lockstep AMR with processes in other level sets (AMR in AMR, MD in AMR, etc.. concurrent)
      // We should revisit this.  JBH 11-14-2015
      if (indexInSubset > 0 && !d_sharedState->isLockstepAMR()) {
        multiplier *= levelHandle->getRefinementRatioMaxDim();
      }

      if (newDW->exists(d_sharedState->get_delt_label(), -1, *levelHandle->patchesBegin())) {
        delt_vartype deltVar;
        double       delt;
        newDW->get(deltVar, d_sharedState->get_delt_label(), levelHandle.get_rep());

        delt = deltVar;
        newDW->put(delt_vartype(delt * multiplier), d_sharedState->get_delt_label());
      }
    }
  }

  if (d_myworld->size() > 1) {
    newDW->reduceMPI(d_sharedState->get_delt_label(), 0, 0, -1);
  }

  // reduce output interval and checkpoint interval
  // if no value computed on that MPI rank,  benign value will be set
  // when the reduction result is also benign value, this value will be ignored
  // that means no MPI rank want to change the interval

  if (d_sharedState->updateOutputInterval()) {
    if (patches->size() != 0 && !newDW->exists(d_sharedState->get_outputInterval_label(), -1, 0)) {
      min_vartype inv;
      inv.setBenignValue();
      newDW->put(inv, d_sharedState->get_outputInterval_label());
    }
    if (d_myworld->size() > 1) {
      newDW->reduceMPI(d_sharedState->get_outputInterval_label(), 0, 0, -1);
    }

  }

  if (d_sharedState->updateCheckpointInterval()) {

    if (patches->size() != 0 && !newDW->exists(d_sharedState->get_checkpointInterval_label(), -1, 0)) {
      min_vartype inv;
      inv.setBenignValue();
      newDW->put(inv, d_sharedState->get_checkpointInterval_label());
    }
    if (d_myworld->size() > 1) {
      newDW->reduceMPI(d_sharedState->get_checkpointInterval_label(), 0, 0, -1);
    }

  }

}
//______________________________________________________________________
//
void
MultiScaleSimulationController::reduceSysVar( const ProcessorGroup * /*pg*/,
                                              const PatchSubset    * patches,
                                              const MaterialSubset * /*matls*/,
                                                    DataWarehouse  * /*old_dw*/,
                                                    DataWarehouse  *  new_dw )
{
  // the goal of this task is to line up the delt across all levels.  If the coarse one
  // already exists (the one without an associated level), then we must not be doing AMR
  if (patches->size() != 0 && !new_dw->exists(d_sharedState->get_delt_label(), -1, 0)) {
    int multiplier = 1;
    const GridP grid = patches->get(0)->getLevel()->getGrid();

    for (int i = 0; i < grid->numLevels(); i++) {
      const LevelP level = grid->getLevel(i);

      if (i > 0 && !d_sharedState->isLockstepAMR()) {
        multiplier *= level->getRefinementRatioMaxDim();
      }

      if (new_dw->exists(d_sharedState->get_delt_label(), -1, *level->patchesBegin())) {
        delt_vartype deltvar;
        double delt;
        new_dw->get(deltvar, d_sharedState->get_delt_label(), level.get_rep());

        delt = deltvar;
        new_dw->put(delt_vartype(delt * multiplier), d_sharedState->get_delt_label());
      }
    }
  }
  
  if (d_myworld->size() > 1) {
    new_dw->reduceMPI(d_sharedState->get_delt_label(), 0, 0, -1);
  }

  // reduce output interval and checkpoint interval 
  // if no value computed on that MPI rank,  benign value will be set
  // when the reduction result is also benign value, this value will be ignored 
  // that means no MPI rank want to change the interval

  if (d_sharedState->updateOutputInterval()) {

    if (patches->size() != 0 && !new_dw->exists(d_sharedState->get_outputInterval_label(), -1, 0)) {
      min_vartype inv;
      inv.setBenignValue();
      new_dw->put(inv, d_sharedState->get_outputInterval_label());
    }
    if (d_myworld->size() > 1) {
      new_dw->reduceMPI(d_sharedState->get_outputInterval_label(), 0, 0, -1);
    }

  }

  if (d_sharedState->updateCheckpointInterval()) {

    if (patches->size() != 0 && !new_dw->exists(d_sharedState->get_checkpointInterval_label(), -1, 0)) {
      min_vartype inv;
      inv.setBenignValue();
      new_dw->put(inv, d_sharedState->get_checkpointInterval_label());
    }
    if (d_myworld->size() > 1) {
      new_dw->reduceMPI(d_sharedState->get_checkpointInterval_label(), 0, 0, -1);
    }

  }

} // end reduceSysVar()
