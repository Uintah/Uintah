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

#include <CCA/Components/SimulationController/LevelSetSimulationController.h>

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
#include <Core/Exceptions/IllegalValue.h>
#include <Core/Geometry/IntVector.h>
#include <Core/Geometry/Vector.h>
#include <Core/Grid/Box.h>
#include <Core/Grid/Grid.h>
#include <Core/Grid/Level.h>
#include <Core/Grid/Patch.h>
#include <Core/Grid/SimulationState.h>
#include <Core/Grid/SimulationTime.h>
#include <Core/Grid/Variables/PerPatch.h>
#include <Core/Grid/Variables/SoleVariable.h>
#include <Core/Grid/Variables/VarLabel.h>
#include <Core/Grid/Variables/VarLabelMatl.h>
#include <Core/Grid/Variables/VarTypes.h>
#include <Core/Grid/Variables/ReductionVariable.h>
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

static DebugStream LevelSetout("LevelSet",                     false);
static DebugStream dbg(          "LevelSetSimulationController", false);
static DebugStream dbg_barrier(  "MPIBarriers",                    false);
static DebugStream dbg_dwmem(    "LogDWMemory",                    false);
static DebugStream gprofile(     "CPUProfiler",                    false);
static DebugStream gheapprofile( "HeapProfiler",                   false);
static DebugStream gheapchecker( "HeapChecker",                    false);

double level_set_barrier_times[5]={0};

void
LevelSetSimulationController::subcomponentPostGridSetup(
                                                                UintahParallelComponent * subcomponent
                                                        ,       SimulationTime          * subcomponentTimer
                                                        ,       LevelSet                * subcomponentLevelSet
                                                        ,       bool                      isRestarting

                                                        , const ProblemSpecP              subcomponentSpec
                                                        ,       SimulationStateP          subcomponentState
                                                        ,       int                       subcomponentTimestep
                                                       )
{
  ProblemSpecP         restartProblemSpec;
  SimulationInterface* subInterface = dynamic_cast<SimulationInterface*>    (subcomponent);
  Scheduler*           subScheduler = dynamic_cast<Scheduler*>              (subcomponent->getPort("scheduler"));

  GridP                grid = subcomponentLevelSet->getSubset(0)->get(0)->getGrid();

  if (isRestarting) {
    // Throw internal error for now.
    throw InternalError("ERROR: Restart is not currently supported in multiscale simulations.", __FILE__, __LINE__);
    restartProblemSpec = d_archive->getTimestepDocForComponent(d_restartIndex);
  }
  subInterface->problemSetup(subcomponentSpec, restartProblemSpec, grid, subcomponentState);

  if (isRestarting) {
    //  The following code is to get this code in the correct place.  However, right now we don't
    // support restarting.
    // FIXME TODO JBH APH 11-23-2015 Verify the restart code below once we have implementation
    // for restarting multiple levelset, multiple component simulations.
    throw InternalError("ERROR: Restart is not currently supported in multiscale simulations.", __FILE__, __LINE__);

    // If necessary, set this component's timestep.
    subcomponentState->setCurrentTopLevelTimeStep(subcomponentTimestep);
    // Tell the subcomponent scheduler the generation of the re-started simulation.
    // (Add +1 because the scheduler will be starting on the next timestep.)
    subScheduler->setGeneration(subcomponentTimestep + 1);

    // If the user wishes to change del_t for the subcomponent on a restart...
    // FIXME TODO JBH APH 11-23-2015 |  The following code is probably not right.
    // Specifically, it's set to recalculate delt for every subset which is in the levelSet
    // However, if multiple subsets are in a level set they ought to all have a common del_t,
    // and that del_t ought to be related to the levelSubset with the maximum number of levels which
    // subdivide the timestep.  At the current time, that means the subset with the maximum number
    // of AMR levels presuming isLockstepAMR isn't set.
    if (subcomponentTimer->override_restart_delt != 0) {
      double newdelt = d_timeinfo->override_restart_delt;
      proc0cout << "Overriding restart delt with " << newdelt << "\n";
      subScheduler->get_dw(1)->override(delt_vartype(newdelt), subcomponentState->get_delt_label());
      int numSubsets = subcomponentLevelSet->size();
      for (int subsetIndex = 0; subsetIndex < numSubsets; ++subsetIndex) {
        double delt_fine = newdelt;
        const LevelSubset* currentSubset = subcomponentLevelSet->getSubset(subsetIndex);
        int numLevelsInSubset = currentSubset->size();
        for (int indexInSubset = 0; indexInSubset < numLevelsInSubset; ++indexInSubset) {
          LevelP levelHandle = grid->getLevel(currentSubset->get(indexInSubset)->getIndex());
          if (indexInSubset != 0 && !subcomponentState->isLockstepAMR() && levelHandle->isAMR() )
          {
            delt_fine /= levelHandle->getRefinementRatioMaxDim();
          }
          subcomponentScheduler->get_dw(1)->override(delt_vartype(delt_fine),
                                                     d_sharedState->get_delt_label(),
                                                     levelHandle.get_rep());

        }
      }

    }
  }  // isRestarting
  subcomponentState->finalizeMaterials();

  // FIXME TODO JBH APH 11-23-2015 This almost certainly isn't right, since initialize will expect
  // an output handle that needs to be, well, initialized, instead of simply having the info from
  // one of N subcomponents initialized within it.
  //d_output->initializeOutput(subcomponentSpec);

}

LevelSetSimulationController::LevelSetSimulationController( const ProcessorGroup * myworld,
                                                                      bool             doAMR,
                                                                      bool             doMultiScale,
                                                                      ProblemSpecP     pspec)
  : SimulationController(myworld, doAMR, doMultiScale, pspec)
  , d_levelSetRunType(serial)
  , d_totalComponents(-1)
  , d_totalSteps(-1)
  , d_numPermDW(-1)
  , d_numTempDW(-1)
{

}

LevelSetSimulationController::~LevelSetSimulationController()
{

}

int LevelSetSimulationController::calculatePermanentDataWarehouses()
{
  switch (d_levelSetRunType) {
    case serial:
      // 1 for the previous component, and one for the current component
      return (2);
    case oscillatory:
      // 1 for each component to store their information
      return (d_totalComponents);
    case hierarchical:
      // 1 for the controller, one for the head process, and one for each seperate component
      return (1+d_totalComponents);
    default:
      return (1);
  }
}

int LevelSetSimulationController::calculateTemporaryDataWarehouses( ProblemSpecP  & multispec)
{
  ProblemSpecP component_spec = multispec->findBlock("Component");
  int numTemp = 0;
  if (!component_spec) return (numTemp);
  while (component_spec) {
    int component_index;
    ProblemSpecP next_spec = component_spec->get("Index",component_index);
    if (!next_spec) {
      throw ProblemSetupException("ERROR:  Component section does not specify component index!",
                                  __FILE__, __LINE__);
    }
    int num_instances;
    next_spec->getWithDefault("Instances",num_instances,1);

    // We have one by default for the component, but if we have more than one instance we'll
    // need more.
    numTemp+=(num_instances-1);
    component_spec = component_spec->findNextBlock("Component");
  }
  return(numTemp);
}

void
LevelSetSimulationController::preGridSetup()
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
    d_levelSetRunType = serial;
  }
  if (runType == "oscillatory" || runType == "OSCILLATORY") {
    d_levelSetRunType = oscillatory;
  }

  d_numPermDW = calculatePermanentDataWarehouses();
  d_numTempDW = calculateTemporaryDataWarehouses(multiScaleSpec);

  SimulationController::preGridSetup();
}


void
LevelSetSimulationController::postGridSetup(
                                                    GridP          & grid
                                            ,       double         & time
                                           )
{
  // Grab the regridder from the BASE component.
  // This is the correct thing to do, because if we regrid we must regrid the entire grid, not just
  // the current level set.
  d_regridder = dynamic_cast<Regridder*> (getPort("regridder"));
  if (d_regridder) {
    d_regridder->problemSetup(d_ups, grid, d_sharedState);
  }

  // Initialize load balancer.
  // FIXME TODO JBH APH 11-22-2015:
  // Investigate how the LB interacts with the scheduler, since this scheduler only contains the DW
  // for the base component.
  d_lb = d_scheduler->getLoadBalancer();
  d_lb->problemSetup( d_ups, grid, d_sharedState);

  if (d_restarting) {
    throw InternalError("ERROR:  Restart not currently supported.", __FILE__, __LINE__);
    // The below probably needs to be reworked for archives containing multiple subcomponents.
    simdbg << "Restarting ... loading data\n";
    d_archive->restartInitialize( d_restartIndex, grid, d_scheduler->get_dw(1), d_lb, &time);

    // Set prevDelt to what it was in the last simulation.
    // If, in the last sim, we were clamping del_t based on the values of prevDelt, then del_t will
    // be off if it doesn't match.
    d_sharedState->d_prev_delt = d_archive->getOldDelt(d_restartIndex);
    d_sharedState->setCurrentTopLevelTimeStep(d_restartTimestep);

    // Tell the scheduler the generation fo the re-started simulation.
    // (Add +1 because the scheduler will be starting on the next timestep.)
    d_scheduler->setGeneration(d_restartTimestep + 1);

    // If the user wishes to change the del_t on a restart... (base timestep)
    if (d_timeinfo->override_restart_delt != 0) {
      double newdelt = d_timeinfo->override_restart_delt;
      proc0cout << "Overriding restart del_t with " << newdelt << "\n";
      d_scheduler->get_dw(1)->override(delt_vartype(newdelt), d_sharedState->get_delt_label());
      // Note that we do NOT need to traverse levels here to determine delt_fine a-la the AMR
      // controller because in this architecture the AMR controller is a fully hosted subcomponent
      // and the varying del_t will be properly set at the subcomponent level.
    }
  }

  ComponentManager::ComponentListType  compList = ComponentManager::ComponentListType::principleandsub;
  // Loop through each subcomponent and set them up
  int activeSubcomponents = d_manager->getNumActiveComponents(compList);
  for (int subComponentIndex = 0; subComponentIndex < activeSubcomponents; ++subComponentIndex) {
    UintahParallelComponent * subComponent       = d_manager->getComponent(subComponentIndex, compList);
    LevelSet*                 subLevelSet        = d_manager->getLevelSet(subComponentIndex, compList);
    ProblemSpecP              subProblemSpec     = d_manager->getProblemSpec(subComponentIndex, compList);
    SimulationStateP          subSimulationState = d_manager->getState(subComponentIndex, compList);
    SimulationTime*           subTimeInfo        = d_manager->getTimeInfo(subComponentIndex, compList);


    SimulationInterface     * subInterface       = dynamic_cast<SimulationInterface*> (subComponent);

    int subRestartTimestep = 0;
    if (d_restarting) {
      throw InternalError("ERROR:  Restarting is currently not supported for multiscale simulation",
                          __FILE__, __LINE__);
      // Find the proper subcomponent timestep index for the Nth subcomponent
      subRestartTimestep = parseSubcomponentOldTimestep(subComponentIndex);
    }

    // TImestep is set in gridSetup.. we will eventually need to parse the subtimestep differently from the
    // top level timestep.
    // FIXME TODO JBH 11-25-2015 Why am I passing in a timestep here?  I don't see where it gets set/passed back currently.
    subcomponentPostGridSetup(subComponent, subTimeInfo, subLevelSet, d_restarting,
                              subProblemSpec, subSimulationState, subRestartTimestep);
  }

  d_sharedState->finalizeMaterials();

  d_output->initializeOutput(d_ups);

  if (d_restarting) {
    throw InternalError("ERROR:  Restart not yet implemented", __FILE__, __LINE__);
    Dir dir(d_fromDir);
    d_output->restartSetup(dir, 0, d_restartTimestep, time, d_restartFromScratch, d_restartRemoveOldDir);
  }


}

GridP
LevelSetSimulationController::parseGridFromRestart() {
  GridP parsedGrid;

  // Create the DataArchive here, and store it, as we use it a few times...
  // We need to read the grid before ProblemSetup, and we can't load all
  // the data until after problemSetup, so we have to do a few
  // different DataArchive operations

  Dir restartFromDir( d_fromDir );
  Dir checkpointRestartDir = restartFromDir.getSubdir( "checkpoints" );
  d_archive = scinew DataArchive( checkpointRestartDir.getName(),
                                  d_myworld->myrank(), d_myworld->size() );

  std::vector<int>    indices;
  std::vector<double> times;

  try {
    d_archive->queryTimesteps( indices, times );
  }
  catch( InternalError & ie ) {
    std::cerr << "\n";
    std::cerr << "An internal error was caught while trying to restart:\n";
    std::cerr << "\n";
    std::cerr << ie.message() << "\n";
    std::cerr << "This most likely means that the simulation UDA that you have specified\n";
    std::cerr << "to use for the restart does not have any checkpoint data in it.  Look\n";
    std::cerr << "in <uda>/checkpoints/ for timestep directories (t#####/) to verify.\n";
    std::cerr << "\n";
    Thread::exitAll(1);
  }

  // Find the right time to query the grid
  if (d_restartTimestep == 0) {
    d_restartIndex = 0; // timestep == 0 means use the first timestep
    // reset d_restartTimestep to what it really is
    d_restartTimestep = indices[0];
  }
  else if (d_restartTimestep == -1 && indices.size() > 0) {
    d_restartIndex = (unsigned int)(indices.size() - 1);
    // reset d_restartTimestep to what it really is
    d_restartTimestep = indices[indices.size() - 1];
  }
  else {
    for (int index = 0; index < (int)indices.size(); index++)
      if (indices[index] == d_restartTimestep) {
        d_restartIndex = index;
        break;
      }
  }

  if (d_restartIndex == (int) indices.size()) {
    // timestep not found
    std::ostringstream message;
    message << "Timestep " << d_restartTimestep << " not found";
    throw InternalError(message.str(), __FILE__, __LINE__);
  }

  // tsaad & bisaac: At this point, and during a restart, there's not a legitimate load balancer. This
  // means that the grid obtained from the data archiver will have global domain BCs on every MPI Rank -
  // i.e. every rank will have knowledge of ALL OTHER patches and their boundary conditions.
  // This leads to a noticeable and unacceptable increase in memory usage especially when
  // hundreds of boundaries (and boundary conditions) are present. That being said, we
  // query the grid WITHOUT requiring boundary conditions. Once that is done, a legitimate load balancer
  // will be created later on, after which we use said load balancer and assign BCs to the grid.
  // NOTE the "false" argument below.
  parsedGrid = d_archive->queryGrid( d_restartIndex, d_ups, false );

  return parsedGrid;
} // parseGridFromRestart()

void
LevelSetSimulationController::basePreGridSetup()
{
  // Reproducing SimulationController::preGridSetup() almost verbatim here
  // to strip out the doMultiTaskgraphing bit.
  d_sharedState = scinew SimulationState(d_ups);
  d_output      = dynamic_cast<Output*> (getPort("output"));

  Scheduler* baseScheduler = dynamic_cast<Scheduler*>(getPort("scheduler"));
  baseScheduler->problemSetup(d_ups, d_sharedState);
  d_scheduler = baseScheduler;

  if (!d_output) {
    throw InternalError("dynamic_cast of 'd_output' failed!", __FILE__, __LINE__);
  }
  d_output->problemSetup(d_ups, d_sharedState.get_rep());

  d_timeinfo = scinew SimulationTime(d_ups);
  d_sharedState->d_simTime = d_timeinfo;

}

LevelSetSimulationController::newRun()
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


  ComponentManager::ComponentListType subList = ComponentManager::ComponentListType::subcomponent;
  ComponentManager::ComponentListType principleList = ComponentManager::ComponentListType::principle;

  int activeSubcomponents = d_manager->getNumActiveComponents(subList);
  int principleComponents  = d_manager->getNumActiveComponents(principleList);
  // Do setup of the base component before grid creation
  basePreGridSetup();

  //SimulationControllerCommon::gridSetup() calls the individual component's preGridSetup routine
  //  before making the grid.  Since we have multiple components to do this with independently,
  //  we need to seperate this out into parts.
  // FIXME TODO JBH APH 11-23-2015 No attempt has been made to rectify how multiple, possibly
  //  contradictory component->preGridSetup() calls interact because currently only MD and
  //  Wasatch have them, and MD's is trivial.  (Wasatch's is decidedly not trivial and involves
  //  grid modification and should definitely be looked at and handled.)

  GridP baseSimulationGrid;
  if (!d_restarting) { // If we're restarting, we'll just read the grid in.
    baseSimulationGrid = scinew Grid();
    for (int subComponentIndex = 0; subComponentIndex < activeSubcomponents; ++subComponentIndex) {
      ProblemSpecP              subSpec      = d_manager->getProblemSpec(subComponentIndex, subList);
      UintahParallelComponent*  subComponent = d_manager->getComponent(subComponentIndex, subList);
      SimulationStateP          subState     = d_manager->getState(subComponentIndex, subList);

      SimulationInterface*      subInterface = dynamic_cast<SimulationInterface*> (subComponent);

      // Do preGridProblemSetup on each subcomponent.
      subInterface->preGridProblemSetup(subSpec, baseSimulationGrid, subState);
    }
    // Set up base grid
    baseSimulationGrid->problemSetup(d_ups, d_myworld, d_doAMR, d_doMultiScale);
  }
  else
  {
    throw InternalError("ERROR:  Restarts not currently supported in multiscale controller.", __FILE__, __LINE__);
    baseSimulationGrid = parseGridFromRestart();
  }
  // Do the base component portions of the grid setup

  // Initialize the scheduler of the base component
  initializeScheduler(d_scheduler, baseSimulationGrid);

  // And of each subcomponent
  for (int subComponentIndex = 0; subComponentIndex < activeSubcomponents; ++subComponentIndex) {
    // This will set up schedulers and time trackers for each subcomponent independently.
    UintahParallelComponent*  subComponent = d_manager->getComponent(subComponentIndex, subList);
    SchedulerP subScheduler = subComponent->getPort("scheduler");
    int requestedNewDW = d_manager->getRequestedNewDWCount(subComponentIndex, subList);
    int requestedOldDW = d_manager->getRequestedOldDWCount(subComponentIndex, subList);
    initializeScheduler(subScheduler, baseSimulationGrid, requestedOldDW, requestedNewDW);
  }

  bool firstTimestep = true;
  // FIXME TODO JBH APH 11-22-2015 This probably isn't right for interleaved multiple independent components.
  if (d_restarting) {
    d_scheduler->setRestartInitTimestep(firstTimestep);
  }

  double baseSimulationTime;
  postGridSetup(baseSimulationGrid, baseSimulationTime);
  // Save actual current start time for time tracking in main and subcomponents
  d_startTime = Time::currentSeconds();

  // FIXME TODO JBH APH 11-26-2015 I'm not entirely sure what d_reduceUda even does!
  // However, fixing it to work across multiple components of a simulation is likely to be super
  // fussy!
  if (d_reduceUda) {
    throw InternalError("ERROR:  Uda reduction not yet implemented for multiscale controller", __FILE__, __LINE__);

    Dir fromDir(d_fromDir);
    d_output->reduceUdaSetup(fromDir);

    d_timeinfo->delt_factor         = 1;
    d_timeinfo->delt_min            = 0;
    d_timeinfo->delt_max            = 1e99;
    d_timeinfo->initTime            = static_cast<UdaReducer*>(d_sim)->getInitialTime();
    d_timeinfo->maxTime             = static_cast<UdaReducer*>(d_sim)->getMaxTime();
    d_timeinfo->max_delt_increase   = 1e99;
    d_timeinfo->max_initial_delt    = 1e99;
  }
  // For now assume that we can treat the component lists independently.. we may need to look at
  // getting lists of principle components each of which contains a list of subcomponents attached to that principle
  // component eventually, but for the demonstrator we will have one principle component with multiple subcomponents
  // so are assuming that this is implicit.
  runInitialTimestepOnList(principleList);
  runInitialTimestepOnList(subList);

  setStartSimTime(baseSimulationTime);
  // We will need to initialize subcomponent simulation stat variables somewhere too.
  initSimulationStatsVars();

  int numBaseIterations = d_sharedState->getCurrentTopLevelTimeStep();
  double del_t   = 0;
  d_lb->resetCostForecaster();
  d_scheduler->setInitTimestep(false);

  while (
          ( baseSimulationTime < d_timeinfo->maxTime ) &&
          ( numBaseIterations < d_timeinfo->maxTimestep) &&
          ( d_timeinfo->max_wall_time == 0 || getWallTime() < d_timeinfo->max_wall_time )
        )
  { // Begin main simulation loop
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

    if ( d_regridder && d_regridder->doRegridOnce() && d_regridder->isAdaptive() ) {
      proc0cout << "______________________________________________________________________\n";
      proc0cout << " Regridding once.\n";
      doRegridding(baseSimulationGrid, false);
      d_regridder->setAdaptivity(false);
      proc0cout << "______________________________________________________________________\n";
    }

    if (d_regridder && d_regridder->needsToReGrid(baseSimulationGrid) && ( !firstTimestep || !d_restarting)) {
      doRegridding(baseSimulationGrid, false);
    }

    doGenericMainTimestep();
    d_sharedState->d_prev_delt = del_t;

    // Update number of time steps and simulation time
    ++numBaseIterations;
    baseSimulationTime += del_t;

  } // End main simulation loop

  d_wallTime = finalizeRunLoop(d_scheduler, d_sharedState, baseSimulationTime);
//  delt_vartype delt_var;
//  d_scheduler->getLastDW()->get(delt_var, d_sharedState->get_delt_label());
//  del_t = delt_var;
//  adjustDelT(del_t, d_sharedState->d_prev_delt, d_sharedState->getCurrentTopLevelTimeStep(), baseSimulationTime);
//  calcWallTime();
  printSimulationStats( d_sharedState->getCurrentTopLevelTimeStep(), del_t, baseSimulationTime);
}

void
LevelSetSimulationController::doGenericMainTimeStep(
                                                      const bool        firstTimestep
                                                    ,       GridP       grid
                                                   )
{

  // By necessity regridding works on the global grid.  So if we have per-level-set flags which
  // signal regridding, we should clear them ALL after regridding has been done.
  if ( d_regridder && d_regridder->doRegridOnce() && d_regridder->isAdaptive() ) {
    proc0cout << "______________________________________________________________________\n";
     proc0cout << " Regridding once.\n";
     doRegridding(grid, false);
     d_regridder->setAdaptivity(false);
     proc0cout << "______________________________________________________________________\n";
   }

   if (d_regridder && d_regridder->needsToReGrid(grid) && (!firstTimestep || (!d_restarting))) {
     doRegridding(grid, false);
   }

}

double
LevelSetSimulationController::runMainTimestepOnList(
                                                     const ComponentManager::ComponentListType componentList
                                                   )
{
  int listSize = d_manager->getNumActiveComponents(componentList);
  for (int listIndex = 0; listIndex < listSize; ++listIndex) {
    UintahParallelComponent * currComponent = d_manager->getComponent(listIndex, componentList);
    LevelSet                * currLevelSet  = d_manager->getLevelSet(listIndex, componentList);
    SimulationStateP        * currState     = d_manager->getState(listIndex, componentList);

    SimulationInterface     * currInterface = dynamic_cast<SimulationInterface*> (currComponent);

    doGenericMainTimestep(currLevelSet, currComponent, currState);
  }
}

void
LevelSetSimulationController::doGenericMainTimestep(
                                                      const LevelSet                * levels
                                                    ,       UintahParallelComponent * component
                                                    ,       SimulationStateP          state
                                                   )
{
  Scheduler* currentScheduler = component->getPort("scheduler");

  int maxTotalFineDW = 1;
  int numLevelSubsets = levels->size();
  for (int subsetIndex = 0; subsetIndex < numLevelSubsets; ++subsetIndex) {
    // Check to see if subset is AMR and is not Lockstep.  If so calculate the number of fine
    // DW levels needed for the AMR.  Since we may theoretically have multiple AMR subsets in a
    // single set, store only the max number of fine DW's needed, as this should be more than
    // sufficient for other subsets which request less.
    LevelSubset* currentSubset = levels->getSubset(0);
    if (currentSubset->get(0)->isAMR() && !state->isLockstepAMR()) {
      int numLevelsInSubset = currentSubset->size();
      int subsetFineDW= 1;
      for (int levelInSubset = 0; levelInSubset < numLevelsInSubset; ++levelInSubset) {
        subsetFineDW *= currentSubset->get(levelInSubset)->getRefinementRatioMaxDim();
      }
      maxTotalFineDW = Max(maxTotalFineDW, subsetFineDW);
    }
  }

  state->d_prev_delt = del_t;

  delt_vartype del_t_var;
  DataWarehouse* newDW = currentScheduler->getLastDW();
  newDW->get(del_t_var, state->get_delt_label());

  del_t = del_t_var;
}

void
LevelSetSimulationController::runInitialTimestepOnList(
                                                       const ComponentManager::ComponentListType componentList
                                                      )
{
  int listSize = d_manager->getNumActiveComponents(componentList);
  for (int listIndex = 0; listIndex < listSize; ++listIndex) {
    UintahParallelComponent   * currComponent = d_manager->getComponent(listIndex, componentList);
    LevelSet                  * currLevelSet  = d_manager->getLevelSet(listIndex, componentList);
    SimulationInterface       * currInterface = dynamic_cast<SimulationInterface*> (currComponent);

    doGenericInitialTimestep(currLevelSet, currComponent, currInterface);
  }
}


double
LevelSetSimulationController::finalizeRunLoop(
                                                      SchedulerP        workingScheduler
                                              ,       SimulationStateP  workingState
                                              ,       double            workingTime
                                             )
{
  delt_vartype del_t_var;
  workingScheduler->getLastDW()->get(del_t_var, workingState->get_delt_label());
  double del_time = del_t_var;
  adjustDelT(del_time, workingState->d_prev_delt, workingState->getCurrentTopLevelTimeStep(), workingTime);
  double localWallTime = Time::currentSeconds();
  return localWallTime;
}

UintahParallelComponent*
LevelSetSimulationController::instantiateNewComponent(
                                                        const ProcessorGroup * myWorld
                                                      ,       ProblemSpecP   & current_ups
                                                      // This ^^ should be const but can't be because
                                                      // *Factory::create calls routines on the problem spec
                                                      // which aren't properly const qualified for the problemspec
                                                     )
{
  bool doAMR = Grid::specIsAMR(current_ups);
  UintahParallelComponent * currentComponent = ComponentFactory::create(current_ups, myWorld, doAMR, "");

  SimulationInterface     * currentInterface = dynamic_cast<SimulationInterface*> (currentComponent);
  currentComponent->attachPort("interface", currentInterface);

  SolverInterface         * currentSolver = SolverFactory::create(current_ups, myWorld);
  currentComponent->attachPort("solver", currentSolver);

  return currentComponent;
  // Switching criteria should be attached at the base component level so that it can check and manage.

}
//______________________________________________________________________
//
SimulationStateP
LevelSetSimulationController::subcomponentPreGridSetup(
                                                               UintahParallelComponent * component
                                                       , const ProblemSpecP            & component_ups
                                                      )
{
  // Process the equivalent of the pre-grid setup for this subcomponent's independent data
  // Create a simulation state for this component
  SimulationStateP newState = scinew SimulationState(component_ups);
  Scheduler* componentScheduler = dynamic_cast<Scheduler*>(component->getPort("scheduler"));
  componentScheduler->problemSetup(component_ups, newState);

  // Attach SimulationTime object directly to the component's simulation state.
  SimulationTime* newState->d_simTime = scinew SimulationTime(component_ups);

}

void
LevelSetSimulationController::subcomponentLevelSetSetup(
                                                          const LevelSet                & currLevelSet
                                                        , const ProblemSpecP            & component_ups
                                                        ,       UintahParallelComponent * currentComponent
                                                        ,       SimulationStateP        & currentState
                                                        ,       bool                      isRestarting
                                                       )
{
  // Analogous to AMRSimulationController::gridSetup
  // Note that we skip the restarting stuff for now, and the grid setup portion actually has already occurred.
  GridP simulationGrid = currLevelSet.getSubset(0)->get(0)->getGrid();

  if (isRestarting) {
    std::ostringstream msg;
    msg << "ERROR:  LevelSetSimulationController::subcomponentLevelSetSetup -> Restart not currently supported.\n";
    throw ProblemSetupException(msg.str(), __FILE__, __LINE__ );
  }
  SimulationInterface* currentInterface = dynamic_cast<SimulationInterface*> (currentComponent->getPort("sim"));
  // FIXME TODO JBH APH 11-22-2015 THIS WILL BREAK IF preGridProblemSetup mucks with the grid!
  // This should actually be converted to work on a level-set eventually, though right now only MD (trivially)
  // and Wasatch (nontrivially) and MultiScaleSwitcher (which this scheduler is designed to replace) actually
  // use preGridProblemSetup.
  currentInterface->preGridProblemSetup(component_ups, simulationGrid, currentState);

  SCIRun::IntVector setSize(-1,-1,-1);
  int numSubsetsInSet = currLevelSet.size();
  for (int subsetIndex = 0; subsetIndex < numSubsetsInSet; ++subsetIndex) {
    const LevelSubset* currentSubset = currLevelSet.getSubset(subsetIndex);
    int numLevelsInSubset = currentSubset->size();
    for (int indexInSubset = 0; indexInSubset < numLevelsInSubset; ++indexInSubset) {
      SCIRun::IntVector levelLow, levelHigh, levelSize;
      const LevelP levelHandle = simulationGrid->getLevel(currentSubset->get(indexInSubset)->getIndex());
      levelHandle->findCellIndexRange(levelLow, levelHigh);
      levelSize = levelHigh - levelLow - levelHandle->getExtraCells()*SCIRun::IntVector(2,2,2);
      setSize = Max(setSize, levelSize);
    }
  }
  // If more than one cell index in a direction, then that direction is an expressed dimension.
  currentState->setDimensionality(setSize.x() > 1, setSize.y() > 1, setSize.z() > 1);
}


void
LevelSetSimulationController::firstTimeStepIndependentComponent(        SchedulerP              & scheduler
                                                                ,       SimulationStateP        & simState
                                                                ,const  LevelSet                & currLevelSet
                                                                ,       double                  & startTime
                                                                ,       UintahParallelComponent * component
                                                                ,       SimulationTime          * timeTracker
                                                               )
{
  // We may need to do pre-grid setup here.  If so, we'll ned to roll our own pre-grid setup routine
  GridP grid = currLevelSet.getSubset(0)->get(0)->getGrid();
  bool initialize;
  scheduler->initialize(1,1);
  scheduler->advanceDataWarehouse(grid, (initialize = true));
  scheduler->setInitTimestep(true);

  // We may need to do post-grid setup here.  If so, we'll need to roll our own post-grid setup routine.
  double time;
  //postGridSetup(grid, time);
  startTime = Time::currentSeconds();

  // reduceUda was here.. does it need to be back here?

  double start = Time::currentSeconds();
  scheduler->mapDataWarehouse(Task::OldDW, 0);
  scheduler->mapDataWarehouse(Task::NewDW, 1);
  // Are these actually necessary, or is this some AMR only weirdness?
  scheduler->mapDataWarehouse(Task::CoarseOldDW, 0);
  scheduler->mapDataWarehouse(Task::CoarseNewDW, 1);

  if (d_restarting) {
    // Not implemented yet
    throw ProblemSetupException("ERROR:  LevelSet based restarts are not yet implemented!", __FILE__, __LINE__);
  }
  else {
    simState->setCurrentTopLevelTimeStep(0);
    LoadBalancer* balancer = component->getPort("LoadBalancer");
    balancer->possiblyDynamicallyReallocate(currLevelSet, LoadBalancer::init);
    // If we want per component boundary conditions, we need to switch from passing in d_grid_ps
    // to passing in individual grid problemSpecs/boundary conditions
    grid->assignBCS(currLevelSet, d_grid_ps, balancer);
    grid->performConsistencyCheck(currLevelSet);
    time = timeTracker->initTime;

    bool needNewLevel = false;
    bool componentIsAMR;
    do {
      if (needNewLevel && componentIsAMR) {
        scheduler->initialize(1,1);
        bool initializeSub;
        scheduler->advanceDataWarehouse(grid,(initializeSub = true));
      }

      proc0cout << "Compiling initialization taskgraph using levelSets..." < std::endl;
      int numSubsets = currLevelSet.size();
      for (int subsetIndex = 0; subsetIndex < numSubsets; ++subsetIndex) {
        // Verify that patches in an isolated subset to not overlap
        const LevelSubset* currLevelSubset = currLevelSet.getSubset(subsetIndex);
        int   maxLevelsInSubset = currLevelSubset->size();
        proc0cout << "Seeing " << maxLevelsInSubset << " level(s) in subset "
                  << subsetIndex << "." <<std::endl;
        for (int indexInSubset = maxLevelsInSubset; indexInSubset > 0; --indexInSubset) {
          proc0cout << " Current level index: " << indexInSubset-1 << " numLevels: "
                    << maxLevelsInSubset << std::endl;

        }
      }
    }
    while (needNewLevel);
  }
}

void
LevelSetSimulationController::run()
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
  GridP baseGrid = gridSetup();
  // Initialize the scheduler and DW for the component manager
  bool baseFirstTimestep = true;
  d_scheduler->initialize(1,1);
  d_scheduler->advanceDataWarehouse( baseGrid, baseFirstTimestep );
  d_scheduler->setInitTimestep( baseFirstTimestep );
  
  if (d_restarting) {
    d_scheduler->setRestartInitTimestep(baseFirstTimestep);
  }
  double baseTime;

  // set up sim, regridder, and finalize sharedState
  // also reload from the DataArchive on restart
  postGridSetup( baseGrid, baseTime );
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

  // The general model for multiscale is this:
  // There is a base component which manages which other components to run in what order.
  // We will -always- initialize the base component, we may/may not need to initialize other components
  // at the same time.

  // The following should initialize the base component.  This should be a lightweight
  // manager that controls which subcomponent we're actually running.
  std::vector<int> baseComponentSubsetIndices;
  LevelSet baseLevelSet;  // Level set to be initialized with the base component.
  int numBaseComponents = baseComponentSubsetIndices.size();
  for (int componentIndex = 0; componentIndex < numBaseComponents; ++componentIndex) {
    baseLevelSet.addAll(baseGrid->getLevelSubset(componentIndex)->getVector());
  }
  doInitialTimestep(baseLevelSet, baseTime);
  setStartSimTime(baseTime);
  initSimulationStatsVars();

#ifndef DISABLE_SCI_MALLOC
  AllocatorSetDefaultTagLineNumber(d_sharedState->getCurrentTopLevelTimeStep());
#endif

  // !!!!!!! Main time stepping loop !!!!!! //

  // Once we've initialized our base manager, it should be ready to tell us what our
  // first "real" component to run is.
  //
  int       numBaseIterations   = d_sharedState->getCurrentTopLevelTimeStep();
  double    del_t               = 0;
  double    start;

  d_lb->resetCostForecaster();
  d_scheduler->setInitTimestep(false);

  while(
         ( baseTime < d_timeinfo->maxTime ) &&
         ( numBaseIterations < d_timeinfo->maxTimestep ) &&
         ( d_timeinfo->max_wall_time == 0 || d_wallTime < d_timeinfo->max_wall_time)
       )  {  // begin base loop

#ifdef USE_GPERFTOOLS
    if (gheapprofile.active()){
      char heapename[512];
      sprintf(heapename, "Timestep %d", iterations);
      HeapProfilerDump(heapename);
    }
#endif

    if (dbg_barrier.active()) {
      for (int index = 0; index < 5; ++index) {
        level_set_barrier_times[index] = 0;
      }
    }

    if (d_regridder && d_regridder->doRegridOnce() && d_regridder->isAdaptive() ) {
      proc0cout << "______________________________________________________________________\n"
                << "  Regridding once.\n" << std::endl;
      doRegridding(baseGrid, false);
      d_regridder->setAdaptivity(false);
      proc0cout << "______________________________________________________________________\n"
                << std::endl;
    }
    if (d_regridder && d_regridder->needsToReGrid(baseGrid) && (!baseFirstTimestep || !d_restarting)) {
      doRegridding(baseGrid, false);
    }

    // Compute the number of dataWarehouses needed for an AMR level subset.
    if (d_doAMR && !d_sharedState->isLockstepAMR()) {
      int numSubsetsInSet = baseLevelSet.size();
      for (int subsetIndex = 0; subsetIndex < numSubsetsInSet; ++subsetIndex) {
        const LevelSubset* currSubset = baseLevelSet.getSubset(subsetIndex);
        int numLevelsInSubset = currSubset->size();
        int totalFine = 1;
        for (int indexInSubset = 0; indexInSubset < numLevelsInSubset; ++indexInSubset) {
          totalFine *= baseGrid->getLevel(currSubset->get(indexInSubset)->getIndex());
        }
      }
    }

    d_sharedState->d_prev_delt = del_t;
    ++numBaseIterations;  // Increment because we've just finished the initial timestep

    // get del_t and adjust it
    delt_vartype del_t_var;
    DataWarehouse* newDW = d_scheduler->getLastDW();
    newDW->get(del_t_var, d_sharedState->get_delt_label());
    del_t = del_t_var;

    // del_t adjusted based on timeinfo parameters
    adjustDelT(del_t, d_sharedState->d_prev_delt, baseFirstTimestep, baseTime);
    newDW->override(delt_vartype(del_t), d_sharedState->get_delt_label());

    if (dbg_dwmem.active()) {
      // Remember this isn't logged if DISABLE_SCI_MALLOC is set
      // (So usually in optimized mode this will not be run.)
      d_scheduler->logMemoryUse();
      std::ostringstream fn;
      fn << "alloc." << std::setw(5) << std::setfill('0') << d_myworld->myrank()<< ".out";
      std::string filename (fn.str()));
      #if !defined( DISABLE_SCI_MALLOC )
        DumpAllocator(DefaultAllocator(), filename.c_str());
      #endif
    }
    if (dbg_barrier.active()) {
      start = Time::currentSeconds();
      MPI_Barrier(d_myworld->getComm());
      level_set_barrier_times[2] += Time::currentSeconds() - start;
    }

    // Yes, I know this is kind of hacky, but this is the only way to
    // get a new grid from UdaReducer. Needs to be done before advanceDataWarehouse.
    if (d_reduceUda){
      baseGrid = static_cast<UdaReducer*>(d_sim)->getGrid();
    }

    // After one step (either timestep or initialization) and correction
    // the delta we can finally, finalize our old timestep, eg.
    // finalize and advance the Datawarehouse
    d_scheduler->advanceDataWarehouse(baseGrid);

    // Put the current time into the shared state so other components
    // can access it.  Also increment (by one) the current time step
    // number so components can tell what timestep they are on.
    d_sharedState->setElapsedTime( baseTime );
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
    if ((nr = needRecompile(baseTime, del_t, baseGrid)) || baseFirstTimestep) {
      if (nr) { // Recompile taskgraph, re-assign BCs, reset recompile flag.
        baseGrid->assignBCS(d_grid_ps, d_lb);
        baseGrid->performConsistencyCheck(baseLevelSet);
        d_sharedState->setRecompileTaskGraph(false);
      }

      new_init_delt = d_timeinfo->max_initial_delt;
      if (new_init_delt != old_init_delt) {
        // Writes to the DW in the next section below
        del_t = new_init_delt;
      }
      recompile(baseTime, del_t, baseLevelSet, totalFine);

    }
  } // end base loop

    if (d_regridder && d_regridder->doRegridOnce())

  std::vector<int> initializationComponentIndices;
  switch (d_levelSetRunType) {
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
    runningLevelSet.addAll(baseGrid->getLevelSubset(componentIndex)->getVector());
  }
  // Run the first timestep initialization only for the prescribed level sets
  doInitialTimestep(runningLevelSet, baseTime);

//  doInitialTimestep( baseGrid, time );
  setStartSimTime( baseTime );
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



  while( ( baseTime < d_timeinfo->maxTime ) &&
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
        level_set_barrier_times[i] = 0;
      }
    }
     
    //__________________________________
    //    Regridding
    if( d_regridder && d_regridder->doRegridOnce() && d_regridder->isAdaptive() ){
      proc0cout << "______________________________________________________________________\n";
      proc0cout << " Regridding once.\n";
      doRegridding(baseGrid, false);
      d_regridder->setAdaptivity(false);
      proc0cout << "______________________________________________________________________\n";
    }

    if (d_regridder && d_regridder->needsToReGrid(baseGrid) && (!baseFirstTimestep || (!d_restarting))) {
      doRegridding(baseGrid, false);
    }

    // Compute number of dataWarehouses - multiplies by the time refinement ratio for each level you increase
    // TODO FIXME JBH APH This logic needs to be fixed for AMR multicomponent.  11-15-2015
    int totalFine = 1;
    if (!d_doMultiScale && !d_sharedState->isLockstepAMR()) {
      for (int i = 1; i < baseGrid->numLevels(); i++) {
        totalFine *= baseGrid->getLevel(i)->getRefinementRatioMaxDim();
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
    adjustDelT( delt, d_sharedState->d_prev_delt, baseFirstTimestep, baseTime );
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
      level_set_barrier_times[2] += Time::currentSeconds() - start;
    }

    // Yes, I know this is kind of hacky, but this is the only way to
    // get a new grid from UdaReducer. Needs to be done before advanceDataWarehouse.
    if (d_reduceUda){
      baseGrid = static_cast<UdaReducer*>(d_sim)->getGrid();
    }

    // After one step (either timestep or initialization) and correction
    // the delta we can finally, finalize our old timestep, eg. 
    // finalize and advance the Datawarehouse
    d_scheduler->advanceDataWarehouse(baseGrid);

    // Put the current time into the shared state so other components
    // can access it.  Also increment (by one) the current time step
    // number so components can tell what timestep they are on. 
    d_sharedState->setElapsedTime( baseTime );
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
    if ((nr = needRecompile(baseTime, delt, baseGrid)) || baseFirstTimestep) {

      if (nr) {  // Recompile taskgraph, re-assign BCs, reset recompile flag.
        baseGrid->assignBCS(d_grid_ps, d_lb);
        baseGrid->performConsistencyCheck();
        d_sharedState->setRecompileTaskGraph(false);
      }

      new_init_delt = d_timeinfo->max_initial_delt;

      if (new_init_delt != old_init_delt) {
        // writes to the DW in the next section below
        delt = new_init_delt;
      }
      recompile(baseTime, delt, runningLevelSet, totalFine);
//      recompile(time, delt, baseGrid, totalFine);
    }
    else {
      if (d_output) {
        // This is not correct if we have switched to a different
        // component, since the delt will be wrong
        d_output->finalizeTimestep(baseTime, delt, baseGrid, d_scheduler, 0);
        d_output->sched_allOutputTasks(delt, baseGrid, d_scheduler, 0);
      }
    }

    if (dbg_barrier.active()) {
      start = Time::currentSeconds();
      MPI_Barrier(d_myworld->getComm());
      level_set_barrier_times[3] += Time::currentSeconds() - start;
    }

    // adjust the delt for each level and store it in all applicable dws.

    // delT rectification is a potential mess for multiscale (concurrent simulations should rectify del_t across
    // multiple level sets.

    // TODO FIXME For now, this needs to be fixed to simply work with non-concurrent simulations, which means the
    // loop below shouldn't operate on bare grid levels, but on the current level set.
    double delt_fine = delt;
    int skip = totalFine;
    for (int i = 0; i < baseGrid->numLevels(); i++) {
      const Level* level = baseGrid->getLevel(i).get_rep();

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
    if (d_output && d_sharedState->updateOutputInterval() && !baseFirstTimestep) {
      min_vartype outputInv_var;
      oldDW->get(outputInv_var, d_sharedState->get_outputInterval_label());

      if (!outputInv_var.isBenignValue()) {
        d_output->updateOutputInterval(outputInv_var);
      }
    }

    if (d_output && d_sharedState->updateCheckpointInterval() && !baseFirstTimestep) {
      min_vartype checkInv_var;
      oldDW->get(checkInv_var, d_sharedState->get_checkpointInterval_label());

      if (!checkInv_var.isBenignValue()) {
        d_output->updateCheckpointInterval(checkInv_var);
      }
    }
 
    if (baseFirstTimestep) {
      baseFirstTimestep = false;
    }
     
    calcWallTime();
    printSimulationStats( d_sharedState->getCurrentTopLevelTimeStep()-1, delt, baseTime );

    // Execute the current timestep, restarting if necessary
    d_sharedState->d_current_delt = delt;

    executeTimestep( baseTime, delt, baseGrid, totalFine );
     
    // Print MPI statistics
    d_scheduler->printMPIStats();

    if (!baseFirstTimestep) {
      d_scheduler->setRestartInitTimestep(false);
    }

    // Update the profiler weights
    d_lb->finalizeContributions(baseGrid);
     
    if (dbg_barrier.active()) {
      start = Time::currentSeconds();
      MPI_Barrier(d_myworld->getComm());
      level_set_barrier_times[4] += Time::currentSeconds() - start;
      double avg[5];
      MPI_Reduce(&level_set_barrier_times, &avg, 5, MPI_DOUBLE, MPI_SUM, 0, d_myworld->getComm());

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
      d_output->findNext_OutputCheckPoint_Timestep(  delt, baseGrid );
      d_output->writeto_xml_files( delt, currentGrid );
    }

    baseTime += delt;
  } // end while ( time is not up, etc )

  // print for the final timestep, as the one above is in the middle of a while loop - get new delt, and set walltime first
  delt_vartype delt_var;
  d_scheduler->getLastDW()->get(delt_var, d_sharedState->get_delt_label());
  delt = delt_var;
  adjustDelT( delt, d_sharedState->d_prev_delt, d_sharedState->getCurrentTopLevelTimeStep(), baseTime );
  calcWallTime();
  printSimulationStats( d_sharedState->getCurrentTopLevelTimeStep(), delt, baseTime );
  
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
LevelSetSimulationController::subCycleCompile(  GridP   & grid
                                              , int       startDW
                                              , int       dwStride
                                              , int       step
                                              , int       levelIndex
                                             )
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
//
void
LevelSetSimulationController::subCycleExecute( GridP & grid,
                                                 int     startDW,
                                                 int     dwStride,
                                                 int     levelNum,
                                                 bool    rootCycle )
{
  // there are 2n+1 taskgraphs, n for the basic timestep, n for intermediate timestep work,
  // and 1 for the errorEstimate and stableTimestep, where n is the number of levels.
  if (LevelSetout.active()) {
    LevelSetout << "Start LevelSetSimulationController::subCycleExecute, level=" << grid->numLevels() << '\n';
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
LevelSetSimulationController::needRecompile(       double   time,
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
LevelSetSimulationController::doInitialTimestep(  const LevelSet & initializationSets
                                                ,       double   & time
                                               )
{
  double start = Time::currentSeconds();
  // All levels in the included levelSets are assumed to want/need a combined dataWarehouse
  d_scheduler->mapDataWarehouse(Task::OldDW, 2);
  d_scheduler->mapDataWarehouse(Task::NewDW, 3);
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
    //   For right now we'll just assume the load balancer is null and not call the reallocate
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
          proc0cout << " Current level index: " << indexInSubset << " numLevels: " << numLevels << std::endl;
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
      d_scheduler->get_dw(3)->setScrubbing(DataWarehouse::ScrubNone);
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

bool
LevelSetSimulationController::doRegridding( GridP & currentGrid,
                                              bool    initialTimestep )
{
  double start = Time::currentSeconds();

  GridP oldGrid = currentGrid;
  currentGrid = d_regridder->regrid(oldGrid.get_rep());

  if (dbg_barrier.active()) {
    double start;
    start = Time::currentSeconds();
    MPI_Barrier(d_myworld->getComm());
    level_set_barrier_times[0] += Time::currentSeconds() - start;
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
      level_set_barrier_times[1]+=Time::currentSeconds()-start;
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

      if (LevelSetout.active()) {
        LevelSetout << "---------- NEW GRID ----------" << std::endl;
        LevelSetout << "Grid has " << currentGrid->numLevels() << " level(s)" << std::endl;

        for ( int levelIndex = 0; levelIndex < currentGrid->numLevels(); levelIndex++ ) {
          LevelP level = currentGrid->getLevel( levelIndex );

          LevelSetout << "  Level " << level->getID()
                 << ", indx: "<< level->getIndex()
                 << " has " << level->numPatches() << " patch(es)" << std::endl;

          for ( Level::patchIterator patchIter = level->patchesBegin(); patchIter < level->patchesEnd(); patchIter++ ) {
            const Patch* patch = *patchIter;
            LevelSetout << "(Patch " << patch->getID() << " proc " << d_lb->getPatchwiseProcessorAssignment(patch)
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
LevelSetSimulationController::recompile(        double    time
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

  // If a level is an AMR level, then call subCycleCompile, which in turn calls scheduleTimeAdvance.
  //  Otherwise only call scheduleTimeAdvance, e.g. any non-AMR case
  size_t numSubsets = currentLevelSet.size();
  for (size_t subsetIndex = 0; subsetIndex < numSubsets; ++subsetIndex) {
    const LevelSubset* currentSubset = currentLevelSet.getSubset(subsetIndex);
    int levelsInSubset = currentSubset->size();
      for (int indexInSubset = 0; indexInSubset < levelsInSubset; ++indexInSubset) {
        // If the first level in a subset is AMR, the subset itself should be AMR.
        // FIXME TODO JBH APH - This logic will not work for MPMICE - Think about a way to fix this.  11-16-15
        const LevelP levelHandle = currentGrid->getLevel(currentSubset->get(indexInSubset)->getIndex());
        if (levelHandle.get_rep()->isAMR()) {
          subCycleCompile(currentGrid, 0, totalFine, 0, currentSubset->get(0)->getIndex());
        } else {
          d_sim->scheduleTimeAdvance(levelHandle, d_scheduler);
          d_sim->scheduleFinalizeTimestep(levelHandle, d_scheduler);
        }
      }
  }

  d_scheduler->clearMappings();
  d_scheduler->mapDataWarehouse(Task::OldDW, 0);
  d_scheduler->mapDataWarehouse(Task::NewDW, totalFine);

  numSubsets = currentLevelSet.size();
  for (size_t subsetIndex = 0; subsetIndex < numSubsets; ++subsetIndex) {
    // Verify that patches on a single level do not overlap
    const LevelSubset* currLevelSubset = currentLevelSet.getSubset(subsetIndex);
    size_t numLevels = currLevelSubset->size();
    proc0cout << "\n--------------------------------------------------------------\n"
              << "Seeing " << numLevels << " level(s) in subset " << subsetIndex << std::endl
              << "--------------------------------------------------------------\n" << std::endl;
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
LevelSetSimulationController::executeTimestep( double   t,
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
LevelSetSimulationController::scheduleComputeStableTimestep( const LevelSet     & operatingLevels
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
  Task* task = scinew Task("reduceSysVarLevelSet", this, &LevelSetSimulationController::reduceSysVar);

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
LevelSetSimulationController::reduceSysVar(  const ProcessorGroup * /*pg*/
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

