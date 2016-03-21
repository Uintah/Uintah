/*
 * The MIT License
 *
 * Copyright (c) 1997-2016 The University of Utah
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
#include <sci_defs/papi_defs.h> // for PAPI performance counters

#include <CCA/Components/SimulationController/SimulationController.h>

#include <Core/Exceptions/InternalError.h>
#include <Core/Exceptions/PapiInitializationError.h>
#include <Core/DataArchive/DataArchive.h>
#include <Core/Grid/Grid.h>
#include <Core/Grid/SimulationState.h>
#include <Core/Grid/SimulationTime.h>
#include <Core/Grid/Variables/VarTypes.h>
#include <Core/OS/Dir.h>
#include <Core/OS/ProcessInfo.h>
#include <Core/Parallel/Parallel.h>
#include <Core/Parallel/ProcessorGroup.h>
#include <Core/ProblemSpec/ProblemSpec.h>
#include <Core/Thread/Thread.h>
#include <Core/Util/DebugStream.h>
#include <Core/Util/DOUT.hpp>

#include <CCA/Ports/LoadBalancer.h>
#include <CCA/Ports/Output.h>
#include <CCA/Ports/ProblemSpecInterface.h>
#include <CCA/Ports/Regridder.h>
#include <CCA/Ports/Scheduler.h>
#include <CCA/Ports/SimulationInterface.h>

#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <sys/param.h>
#include <vector>

#define SECONDS_PER_MINUTE        60.0
#define SECONDS_PER_HOUR        3600.0
#define SECONDS_PER_DAY        86400.0
#define SECONDS_PER_WEEK      604800.0
#define SECONDS_PER_YEAR    31536000.0

#define AVERAGE_WINDOW 10

extern Uintah::DebugStream amrout;

namespace {

Uintah::DebugStream dbg(     "SimulationStats",            true  );
Uintah::DebugStream dbgTime( "SimulationTimeStats",        false );
Uintah::DebugStream simdbg(  "SimulationController",       false );
Uintah::DebugStream stats(   "ComponentTimings",           false );
Uintah::DebugStream istats(  "IndividualComponentTimings", false );

}

namespace Uintah {


SimulationController::SimulationController( const ProcessorGroup * myworld
                                          ,       bool             doAMR
                                          ,       ProblemSpecP     pspec
                                          )
  : UintahParallelComponent( myworld )
  , d_ups                    {pspec}
  , d_grid_ps                {d_ups->findBlock( "Grid" )}
  , d_sharedState            {nullptr}
  , d_scheduler              {nullptr}
  , d_lb                     {nullptr}
  , d_output                 {nullptr}
  , d_timeinfo               {nullptr}
  , d_sim                    {nullptr}
  , d_regridder              {nullptr}
  , d_archive                {nullptr}
  , d_doAMR                  {doAMR}
  , d_doMultiTaskgraphing    {false}
  , d_restarting             {false}
  , d_fromDir                {""}
  , d_restartTimestep        {0}
  , d_restartIndex           {0}
  , d_lastRecompileTimestep  {0}
  , d_reduceUda              {false}
  , d_restartFromScratch     {false}
  , d_restartRemoveOldDir    {false}
  , d_n                      {0}
  , d_wallTime               {0.0}
  , d_startTime              {0.0}
  , d_startSimTime           {0.0}
  , d_prevWallTime           {0.0}
  , d_movingAverage          {0.0}
  {

#ifdef HAVE_VISIT
  d_doVisIt                = false;
#endif

#ifdef USE_PAPI_COUNTERS
  /*
   * Setup PAPI events to track.
   *
   * Here and in printSimulationStats() are the only places code needs to be added for
   * additional events to track. Everything is parameterized and hopefully robust enough
   * to handle unsupported events on different architectures. Only supported events will
   * report stats in printSimulationStats().
   *
   * NOTE:
   *          All desired events may not be supported for a particular architecture and bad things,
   *      happen, e.g. misaligned event value array indices when an event can be queried but
   *      not added to an event set, hence the PapiEvent struct, map and logic in printSimulationStats().
   *
   *      On some platforms, errors about resource limitations may be encountered, and is why we limit
   *      this instrumentation to four events now (seems stable). At some point we will look into the
   *      cost of multiplexing with PAPI, which will allow a user to count more events than total
   *      physical counters by time sharing the existing counters. This comes at some loss in precision.
   *
   * PAPI_FP_OPS - floating point operations executed
   * PAPI_DP_OPS - floating point operations executed; optimized to count scaled double precision vector operations
   * PAPI_L2_TCM - level 2 total cache misses
   * PAPI_L3_TCM - level 3 total cache misses
   */
  d_papiEvents.insert(std::pair<int, PapiEvent>(PAPI_FP_OPS, PapiEvent("PAPI_FP_OPS", "FLOPS")));
  d_papiEvents.insert(std::pair<int, PapiEvent>(PAPI_DP_OPS, PapiEvent("PAPI_DP_OPS", "VFLOPS")));
  d_papiEvents.insert(std::pair<int, PapiEvent>(PAPI_L2_TCM, PapiEvent("PAPI_L2_TCM", "L2CacheMisses")));
  d_papiEvents.insert(std::pair<int, PapiEvent>(PAPI_L3_TCM, PapiEvent("PAPI_L3_TCM", "L3CacheMisses")));

  // For meaningful error reporting - PAPI Version: 5.1.0 has 25 error return codes:
  d_papiErrorCodes.insert(pair<int, string>( 0,  "No error"));
  d_papiErrorCodes.insert(pair<int, string>(-1,  "Invalid argument"));
  d_papiErrorCodes.insert(pair<int, string>(-2,  "Insufficient memory"));
  d_papiErrorCodes.insert(pair<int, string>(-3,  "A System/C library call failed"));
  d_papiErrorCodes.insert(pair<int, string>(-4,  "Not supported by substrate"));
  d_papiErrorCodes.insert(pair<int, string>(-5,  "Access to the counters was lost or interrupted"));
  d_papiErrorCodes.insert(pair<int, string>(-6,  "Internal error, please send mail to the developers"));
  d_papiErrorCodes.insert(pair<int, string>(-7,  "Hardware event does not exist"));
  d_papiErrorCodes.insert(pair<int, string>(-8,  "Hardware event exists, but cannot be counted due to counter resource limitations"));
  d_papiErrorCodes.insert(pair<int, string>(-9,  "EventSet is currently not running"));
  d_papiErrorCodes.insert(pair<int, string>(-10, "EventSet is currently counting"));
  d_papiErrorCodes.insert(pair<int, string>(-11, "No such EventSet available"));
  d_papiErrorCodes.insert(pair<int, string>(-12, "Event in argument is not a valid preset"));
  d_papiErrorCodes.insert(pair<int, string>(-13, "Hardware does not support performance counters"));
  d_papiErrorCodes.insert(pair<int, string>(-14, "Unknown error code"));
  d_papiErrorCodes.insert(pair<int, string>(-15, "Permission level does not permit operation"));
  d_papiErrorCodes.insert(pair<int, string>(-16, "PAPI hasn't been initialized yet"));
  d_papiErrorCodes.insert(pair<int, string>(-17, "Component index isn't set"));
  d_papiErrorCodes.insert(pair<int, string>(-18, "Not supported"));
  d_papiErrorCodes.insert(pair<int, string>(-19, "Not implemented"));
  d_papiErrorCodes.insert(pair<int, string>(-20, "Buffer size exceeded"));
  d_papiErrorCodes.insert(pair<int, string>(-21, "EventSet domain is not supported for the operation"));
  d_papiErrorCodes.insert(pair<int, string>(-22, "Invalid or missing event attributes"));
  d_papiErrorCodes.insert(pair<int, string>(-23, "Too many events or attributes"));
  d_papiErrorCodes.insert(pair<int, string>(-24, "Bad combination of features"));

  d_eventValues = new long long[d_papiEvents.size()];
  d_eventSet = PAPI_NULL;
  int retp = -1;

  // some PAPI boiler plate
  retp = PAPI_library_init(PAPI_VER_CURRENT);
  if (retp != PAPI_VER_CURRENT) {
    proc0cout << "Error: Cannot initialize PAPI library!" << endl
              << "       Error code = " << retp << " (" << d_papiErrorCodes.find(retp)->second << ")" << endl;
    throw PapiInitializationError("PAPI library initialization error occurred. Check that your PAPI library can be initialized correctly.", __FILE__, __LINE__);
  }
  retp = PAPI_thread_init(pthread_self);
  if (retp != PAPI_OK) {
    if (d_myworld->myrank() == 0) {
      cout << "Error: Cannot initialize PAPI thread support!" << endl
           << "       Error code = " << retp << " (" << d_papiErrorCodes.find(retp)->second << ")" << endl;
    }
    if (Parallel::getNumThreads() > 1) {
      throw PapiInitializationError("PAPI Pthread initialization error occurred. Check that your PAPI build supports Pthreads.", __FILE__, __LINE__);
    }
  }

  // query all the events to find that are supported, flag those that are unsupported
  for (std::map<int, PapiEvent>::iterator iter=d_papiEvents.begin(); iter!=d_papiEvents.end(); iter++) {
    retp = PAPI_query_event(iter->first);
    if (retp != PAPI_OK) {
      proc0cout << "WARNNING: Cannot query PAPI event: " << iter->second.name << "!" << endl
                << "          Error code = " << retp << " (" << d_papiErrorCodes.find(retp)->second << ")" << endl
                << "          No stats will be printed for " << iter->second.simStatName << endl;
    } else {
      iter->second.isSupported = true;
    }
  }

  // create a new empty PAPI event set
  retp = PAPI_create_eventset(&d_eventSet);
  if (retp != PAPI_OK) {
    proc0cout << "Error: Cannot create PAPI event set!" << endl
              << "       Error code = " << retp << " (" << d_papiErrorCodes.find(retp)->second << ")" << endl;
    throw PapiInitializationError("PAPI event set creation error. Unable to create hardware counter event set.", __FILE__, __LINE__);
  }

  /* Iterate through PAPI events that are supported, flag those that cannot be added.
   *   There are situations where an event may be queried but not added to an event set,
   *   this is the purpose of this block of code.
   */
  int index = 0;
  for (std::map<int, PapiEvent>::iterator iter = d_papiEvents.begin(); iter != d_papiEvents.end(); iter++) {
    if (iter->second.isSupported) {
      retp = PAPI_add_event(d_eventSet, iter->first);
      if (retp != PAPI_OK) { // this means the event queried OK but could not be added
        if (d_myworld->myrank() == 0) {
          cout << "WARNNING: Cannot add PAPI event: " << iter->second.name << "!"  << endl
               << "          Error code = " << retp << " (" << d_papiErrorCodes.find(retp)->second << ")" << endl
               << "          No stats will be printed for " << iter->second.simStatName << endl;
        }
        iter->second.isSupported = false;
      } else {
        iter->second.eventValueIndex = index;
        index++;
      }
    }
  }

  retp = PAPI_start(d_eventSet);
  if (retp != PAPI_OK) {
    proc0cout << "WARNNING: Cannot start PAPI event set!"  << endl
              << "          Error code = " << retp << " (" << d_papiErrorCodes.find(retp)->second << ")" << endl;
    throw PapiInitializationError("PAPI event set start error. Unable to start hardware counter event set.", __FILE__, __LINE__);
  }
#endif
} // end SimulationController constructor

//______________________________________________________________________
//

SimulationController::~SimulationController()
{
  delete d_archive;
  delete d_timeinfo;
#ifdef USE_PAPI_COUNTERS
  delete d_eventValues;
#endif
}

//______________________________________________________________________
//

void
SimulationController::setReduceUdaFlags( const std::string & fromDir )
{
  d_doAMR       = false;
  d_reduceUda   = true;
  d_fromDir     = fromDir;
}

//______________________________________________________________________
//
void
SimulationController::setUseLocalFileSystems()
{
  d_usingLocalFileSystems = true;
}

//______________________________________________________________________
//

void
SimulationController::doRestart( const std::string & restartFromDir, int timestep,
                                 bool fromScratch, bool removeOldDir )
{
  d_restarting          = true;
  d_fromDir             = restartFromDir;
  d_restartTimestep     = timestep;
  d_restartFromScratch  = fromScratch;
  d_restartRemoveOldDir = removeOldDir;
}

//______________________________________________________________________
//
void
SimulationController::preGridSetup( void )
{
  d_sharedState = new SimulationState(d_ups);

  d_sharedState->d_usingLocalFileSystems = d_usingLocalFileSystems;

  d_output = dynamic_cast<Output*>(getPort("output"));

  Scheduler* sched = dynamic_cast<Scheduler*>(getPort("scheduler"));
  sched->problemSetup(d_ups, d_sharedState);
  d_scheduler = sched;

  if (!d_output) {
    std::cout << "dynamic_cast of 'd_output' failed!\n";
    throw InternalError("dynamic_cast of 'd_output' failed!", __FILE__, __LINE__);
  }
  d_output->problemSetup(d_ups, d_sharedState.get_rep());

  ProblemSpecP amr_ps = d_ups->findBlock("AMR");
  if (amr_ps) {
    amr_ps->get("doMultiTaskgraphing", d_doMultiTaskgraphing);
  }

  // Parse time struct
  d_timeinfo = new SimulationTime(d_ups);
  d_sharedState->d_simTime = d_timeinfo;

#ifdef HAVE_VISIT
  d_sharedState->setVisIt( d_doVisIt );
#endif
}

//______________________________________________________________________
//
GridP
SimulationController::gridSetup( void )
{
  GridP grid;

  if( d_restarting ) {
    // Create the DataArchive here, and store it, as we use it a few times...
    // We need to read the grid before ProblemSetup, and we can't load all
    // the data until after problemSetup, so we have to do a few
    // different DataArchive operations

    Dir restartFromDir( d_fromDir );
    Dir checkpointRestartDir = restartFromDir.getSubdir( "checkpoints" );
    d_archive = new DataArchive( checkpointRestartDir.getName(),
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
  }

  if( !d_restarting ) {
    grid = new Grid;
    d_sim = dynamic_cast<SimulationInterface*>(getPort("sim"));
    if( !d_sim ) {
      throw InternalError("No simulation component", __FILE__, __LINE__);
    }
    d_sim->preGridProblemSetup(d_ups, grid, d_sharedState);
    grid->problemSetup(d_ups, d_myworld, d_doAMR);
  }
  else {
    // tsaad & bisaac: At this point, and during a restart, there not legitimate load balancer. This means
    // that the grid obtained from the data archiver will global domain BCs on every MPI Rank -
    // i.e. every rank will have knowledge of ALL OTHER patches and their boundary conditions.
    // This leads to a noticeable and unacceptable increase in memory usage especially when
    // hundreds of boundaries (and boundary conditions) are present. That being said, we
    // query the grid WITHOUT requiring boundary conditions. Once that is done, a legitimate load balancer
    // will be created later on - after which we use said balancer and assign BCs to the grid.
    // NOTE the "false" argument below.
    grid = d_archive->queryGrid( d_restartIndex, d_ups, false );
  }
  if(grid->numLevels() == 0){
    throw InternalError("No problem (no levels in grid) specified.", __FILE__, __LINE__);
  }

  // Print out meta data
  if (d_myworld->myrank() == 0){
    grid->printStatistics();
    amrout << "Restart grid\n" << *grid.get_rep() << std::endl;
  }

  // set the dimensionality of the problem.
  IntVector low, high, size;
  grid->getLevel(0)->findCellIndexRange(low, high);
  size = high-low - grid->getLevel(0)->getExtraCells()*IntVector(2,2,2);
  d_sharedState->setDimensionality(size[0] > 1, size[1] > 1, size[2] > 1);

  return grid;
}

//______________________________________________________________________
//
void
SimulationController::postGridSetup( GridP& grid, double& t )
{
  // Set up regridder with initial information about grid.
  // do before sim - so that Switcher (being a sim) can reset the state of the regridder
  d_regridder = dynamic_cast<Regridder*>(getPort("regridder"));
  if (d_regridder) {
    d_regridder->problemSetup( d_ups, grid, d_sharedState );
  }

  // Initialize load balancer.  Do here since we have the dimensionality in the shared state,
  // and we want that at initialization time. In addition do it after regridding since we need to
  // know the minimum patch size that the regridder will create
  d_lb = d_scheduler->getLoadBalancer();
  d_lb->problemSetup( d_ups, grid, d_sharedState );

  // Initialize the CFD and/or MPM components
  d_sim = dynamic_cast<SimulationInterface*>(getPort("sim"));
  if( !d_sim ) {
    throw InternalError("No simulation component", __FILE__, __LINE__);
  }

  ProblemSpecP restart_prob_spec_for_component = 0;

  if( d_restarting ) {
    // Do these before calling archive->restartInitialize, since problemSetup creates VarLabels the DA needs.
    restart_prob_spec_for_component = d_archive->getTimestepDocForComponent( d_restartIndex );
  }

  // Pass the restart_prob_spec_for_component to the Component's
  // problemSetup.  For restarting, pull the <MaterialProperties> from
  // the restart_prob_spec.  If it is not available, then we will pull
  // the properties from the d_ups instead.  Needs to be done before
  // DataArchive::restartInitialize
  d_sim->problemSetup(d_ups, restart_prob_spec_for_component, grid, d_sharedState);

  if( d_restarting ) {
    simdbg << "Restarting... loading data\n";
    d_archive->restartInitialize( d_restartIndex, grid, d_scheduler->get_dw(1), d_lb, &t );

    // Set prevDelt to what it was in the last simulation.  If in the last
    // sim we were clamping delt based on the values of prevDelt, then
    // delt will be off if it doesn't match.
    d_sharedState->d_prev_delt = d_archive->getOldDelt( d_restartIndex );

    d_sharedState->setCurrentTopLevelTimeStep( d_restartTimestep );
    // Tell the scheduler the generation of the re-started simulation.
    // (Add +1 because the scheduler will be starting on the next
    // timestep.)
    d_scheduler->setGeneration( d_restartTimestep + 1 );

    // If the user wishes to change the delt on a restart....
    if (d_timeinfo->override_restart_delt != 0) {
      double newdelt = d_timeinfo->override_restart_delt;
      proc0cout << "Overriding restart delt with " << newdelt << "\n";
      d_scheduler->get_dw(1)->override( delt_vartype(newdelt), d_sharedState->get_delt_label() );

      double delt_fine = newdelt;
      for( int i = 0; i < grid->numLevels(); i++ ) {
        const Level* level = grid->getLevel(i).get_rep();
        if( i != 0 && !d_sharedState->isLockstepAMR() ) {
          delt_fine /= level->getRefinementRatioMaxDim();
        }
        d_scheduler->get_dw(1)->override( delt_vartype(delt_fine), d_sharedState->get_delt_label(), level );
      }
    }
    // This delete is an enigma... I think if it is called then memory is not leaked, but sometimes if it
    // it is called, then everything segfaults...
    //
    // delete d_archive;
  }

  // Finalize the shared state/materials
  d_sharedState->finalizeMaterials();

  // done after the sim->problemSetup to get defaults into the
  // input.xml, which it writes along with index.xml
  d_output->initializeOutput(d_ups);

  if( d_restarting ) {
    Dir dir(d_fromDir);
    d_output->restartSetup( dir, 0, d_restartTimestep, t, d_restartFromScratch, d_restartRemoveOldDir );
  }
} // end postGridSetup()

//______________________________________________________________________
//
void
SimulationController::adjustDelT( double& delt, double prev_delt, bool first, double t )
{
#if 0
  cout << "maxTime = " << d_timeinfo->maxTime << "\n";
  cout << "initTime = " << d_timeinfo->initTime << "\n";
  cout << "delt_min = " << d_timeinfo->delt_min << "\n";
  cout << "delt_max = " << d_timeinfo->delt_max << "\n";
  cout << "timestep_multiplier = " << d_timeinfo->delt_factor << "\n";
  cout << "delt_init = " << d_timeinfo->max_initial_delt << "\n";
  cout << "initial_delt_range = " << d_timeinfo->initial_delt_range << "\n";
  cout << "max_delt_increase = " << d_timeinfo->max_delt_increase << "\n";
  cout << "first = " << first << "\n";
  cout << "delt = " << delt << "\n";
  cout << "prev_delt = " << prev_delt << "\n";
#endif

  delt *= d_timeinfo->delt_factor;

  if (delt < d_timeinfo->delt_min) {
    proc0cout << "WARNING: raising delt from " << delt << " to minimum: " << d_timeinfo->delt_min << '\n';
    delt = d_timeinfo->delt_min;
  }

  if (!first && d_timeinfo->max_delt_increase < 1.e90 && delt > (1 + d_timeinfo->max_delt_increase) * prev_delt) {
    proc0cout << "WARNING (a): lowering delt from " << delt << " to maxmimum: " << (1 + d_timeinfo->max_delt_increase) * prev_delt
              << " (maximum increase of " << d_timeinfo->max_delt_increase << ")\n";
    delt = (1 + d_timeinfo->max_delt_increase) * prev_delt;
  }

  if (t <= d_timeinfo->initial_delt_range && delt > d_timeinfo->max_initial_delt) {
    proc0cout << "WARNING (b): lowering delt from " << delt << " to maximum: " << d_timeinfo->max_initial_delt
              << " (for initial timesteps)\n";
    delt = d_timeinfo->max_initial_delt;
  }

  if (delt > d_timeinfo->delt_max) {
    proc0cout << "WARNING (c): lowering delt from " << delt << " to maximum: " << d_timeinfo->delt_max << '\n';
    delt = d_timeinfo->delt_max;
  }

  // Clamp timestep to output/checkpoint.
  if (d_timeinfo->timestep_clamping && d_output) {
    double orig_delt = delt;
    double nextOutput = d_output->getNextOutputTime();
    double nextCheckpoint = d_output->getNextCheckpointTime();
    if (nextOutput != 0 && t + delt > nextOutput) {
      delt = nextOutput - t;
    }

    if (nextCheckpoint != 0 && t + delt > nextCheckpoint) {
      delt = nextCheckpoint - t;
    }

    if (delt != orig_delt) {
      proc0cout << "WARNING (d): lowering delt from " << orig_delt << " to " << delt << " to line up with output/checkpoint time\n";
    }
  }

  if (d_timeinfo->end_on_max_time && t + delt > d_timeinfo->maxTime) {
    delt = d_timeinfo->maxTime - t;
  }
}

//______________________________________________________________________
//
double
SimulationController::getWallTime( void )
{
  return d_wallTime;
}

//______________________________________________________________________
//
void
SimulationController::calcWallTime ( void )
{
  d_wallTime = Time::currentSeconds() - d_startTime;
}

//______________________________________________________________________
//
double
SimulationController::getStartTime ( void )
{
  return d_startTime;
}

void
SimulationController::calcStartTime ( void )
{
  d_startTime = Time::currentSeconds();
}

//______________________________________________________________________
//
void
SimulationController::setStartSimTime ( double t )
{
  d_startSimTime = t;
}

//______________________________________________________________________
//
void
SimulationController::initSimulationStatsVars ( void )
{
  // vars used to calculate standard deviation
  d_n = 0;
  d_wallTime = 0;
  d_prevWallTime = Time::currentSeconds();
  //d_sumOfWallTimes = 0; // sum of all walltimes
  //d_sumOfWallTimeSquares = 0; // sum all squares of walltimes
}

//______________________________________________________________________
//
void
SimulationController::printSimulationStats ( int timestep, double delt, double time )
{
  ReductionInfoMapper< SimulationState::RunTimeStat, double > &runTimeStats =
    d_sharedState->d_runTimeStats;

  // With the sum reduces, use double, since with memory it is possible that it will overflow
  double avg_memuse = runTimeStats.getAverage(SimulationState::SCIMemoryUsed);
  unsigned long max_memuse = static_cast<unsigned long>(runTimeStats.getMaximum(SimulationState::SCIMemoryUsed));
  int max_memuse_rank = runTimeStats.getRank(SimulationState::SCIMemoryUsed);

  double avg_highwater = runTimeStats.getAverage(SimulationState::SCIMemoryHighwater);
  unsigned long max_highwater =
      static_cast<unsigned long>(runTimeStats.getMaximum(SimulationState::SCIMemoryHighwater));

  // Sum up the average time for overhead related components. These
  // same values are used in SimulationState::getOverheadTime.
  double overhead_time =
    (runTimeStats.getAverage(SimulationState::CompilationTime) +
     runTimeStats.getAverage(SimulationState::RegriddingTime) +
     runTimeStats.getAverage(SimulationState::RegriddingCompilationTime) +
     runTimeStats.getAverage(SimulationState::RegriddingCopyDataTime) +
     runTimeStats.getAverage(SimulationState::LoadBalancerTime));

  // Sum up the average times for simulation components. These
  // same values are used in SimulationState::getTotalTime.
  double total_time =
    (overhead_time +
     runTimeStats.getAverage(SimulationState::TaskExecTime) +
     runTimeStats.getAverage(SimulationState::TaskLocalCommTime) +
     runTimeStats.getAverage(SimulationState::TaskGlobalCommTime) +
     runTimeStats.getAverage(SimulationState::TaskWaitCommTime) +
     runTimeStats.getAverage(SimulationState::TaskWaitThreadTime));

  // Calculate percentage of time spent in overhead.
  double percent_overhead = overhead_time / total_time;

  //set the overhead sample
  if( d_n > 2 ) { // Ignore the first 3 samples, they are not good samples.
    d_sharedState->overhead[d_sharedState->overheadIndex] = percent_overhead;
    // Increment the overhead index

    double overhead = 0;
    double weight = 0;

    int t = std::min( d_n - 2, OVERHEAD_WINDOW );
    //calcualte total weight by incrementing through the overhead sample array backwards and multiplying samples by the weights
    for( int i = 0; i < t; i++ ) {
      overhead += d_sharedState->overhead[(d_sharedState->overheadIndex+OVERHEAD_WINDOW-i)%OVERHEAD_WINDOW] * d_sharedState->overheadWeights[i];
      weight   += d_sharedState->overheadWeights[i];
    }

    d_sharedState->overheadAvg = overhead/weight;
    d_sharedState->overheadIndex =
      (d_sharedState->overheadIndex+1) % OVERHEAD_WINDOW;

    // Increase overhead size if needed.
  }

  // calculate mean/std dev
  double mean = 0;
  double walltime = d_wallTime-d_prevWallTime;

  if (d_n > 2) { // ignore times 0,1,2
    float alpha = 2.0 / (std::min(d_n-2,AVERAGE_WINDOW)+1);
    d_movingAverage = alpha*walltime + (1-alpha) * d_movingAverage;
    mean = d_movingAverage;
  }

  // Output timestep statistics...
  if (istats.active())
  {
    for (unsigned int i=0; i<runTimeStats.size(); i++)
    {
      SimulationState::RunTimeStat e = (SimulationState::RunTimeStat) i;

      if (runTimeStats[e] > 0)
      {
        DOUT(true, "rank: " << d_myworld->myrank() << " " << std::left << std::setw(19) << runTimeStats.getName(e)
	                          << " [" << runTimeStats.getUnits(e) << "]: " << runTimeStats[e]);
      }
    }
  }

  if( d_myworld->myrank() == 0 )
  {
    char walltime[96];

    if (d_n > 3)
    {
      sprintf( walltime, ", elap T = %6.2lf (mean: %6.2lf), ", d_wallTime, mean );
    }
    else {
      sprintf( walltime, ", elap T = %6.2lf,                ", d_wallTime );
    }

    std::ostringstream message;
    message << "Time="        << time
            << " (timestep "  << timestep
            << "), delT="     << delt
            << walltime;
    message << "Memory Use = ";

    if (avg_memuse == max_memuse && avg_highwater == max_highwater) {
      message << ProcessInfo::toHumanUnits((unsigned long) avg_memuse);
      if(avg_highwater) {
        message << "/" << ProcessInfo::toHumanUnits((unsigned long) avg_highwater);
      }
    }
    else {
      message << ProcessInfo::toHumanUnits((unsigned long) avg_memuse);
      if(avg_highwater) {
        message << "/" << ProcessInfo::toHumanUnits((unsigned long)avg_highwater);
      }
      message << " (avg), " << ProcessInfo::toHumanUnits(max_memuse);
      if(max_highwater) {
        message << "/" << ProcessInfo::toHumanUnits(max_highwater);
      }
      message << " (max on rank:" << max_memuse_rank << ")";
    }

    dbg << message.str() << "\n";
    dbg.flush();
    std::cout.flush();

    if (stats.active()) {
      stats << std::left << std::setw(19)  << "  Description                 Ave:            max:      mpi proc:    100*(1-ave/max) '% load imbalance'\n";

      for (unsigned int i=0; i<runTimeStats.size(); ++i)
      {
	      SimulationState::RunTimeStat e = (SimulationState::RunTimeStat) i;

	      if (runTimeStats.getMaximum(e) > 0)
	      {
	        stats << "  " << std::left << std::setw(19)<< runTimeStats.getName(e)
	      	<< "[" << runTimeStats.getUnits(e) << "]"
	      	<< " : " << std::setw(12) << runTimeStats.getAverage(e)
	      	<< " : " << std::setw(12) << runTimeStats.getMaximum(e)
	      	<< " : " << std::setw(10) << runTimeStats.getRank(e)
	      	<< " : " << std::setw(10)
	      	<< 100*(1-(runTimeStats.getAverage(e)/runTimeStats.getMaximum(e))) << "\n";
	      }
      }

      if( d_n > 2 && !std::isnan(d_sharedState->overheadAvg) )
      {
        stats << "  Time in overhead (%): " << d_sharedState->overheadAvg*100 <<  "\n";
      }
    }

    if ( d_n > 0 ) {

      double realSecondsNow = (d_wallTime - d_prevWallTime) / delt;
      double realSecondsAvg = (d_wallTime - d_startTime) / (time - d_startSimTime);

      dbgTime << "1 sim second takes ";

      dbgTime << std::left << std::showpoint << std::setprecision(3) << std::setw(4);

      if (realSecondsNow < SECONDS_PER_MINUTE) {
        dbgTime << realSecondsNow << " seconds (now), ";
      }
      else if (realSecondsNow < SECONDS_PER_HOUR) {
        dbgTime << realSecondsNow / SECONDS_PER_MINUTE << " minutes (now), ";
      }
      else if (realSecondsNow < SECONDS_PER_DAY) {
        dbgTime << realSecondsNow / SECONDS_PER_HOUR << " hours (now), ";
      }
      else if (realSecondsNow < SECONDS_PER_WEEK) {
        dbgTime << realSecondsNow / SECONDS_PER_DAY << " days (now), ";
      }
      else if (realSecondsNow < SECONDS_PER_YEAR) {
        dbgTime << realSecondsNow / SECONDS_PER_WEEK << " weeks (now), ";
      }
      else {
        dbgTime << realSecondsNow / SECONDS_PER_YEAR << " years (now), ";
      }

      dbgTime << std::setw(4);

      if (realSecondsAvg < SECONDS_PER_MINUTE) {
        dbgTime << realSecondsAvg << " seconds (avg) ";
      }
      else if (realSecondsAvg < SECONDS_PER_HOUR) {
        dbgTime << realSecondsAvg / SECONDS_PER_MINUTE << " minutes (avg) ";
      }
      else if (realSecondsAvg < SECONDS_PER_DAY) {
        dbgTime << realSecondsAvg / SECONDS_PER_HOUR << " hours (avg) ";
      }
      else if (realSecondsAvg < SECONDS_PER_WEEK) {
        dbgTime << realSecondsAvg / SECONDS_PER_DAY << " days (avg) ";
      }
      else if (realSecondsAvg < SECONDS_PER_YEAR) {
        dbgTime << realSecondsAvg / SECONDS_PER_WEEK << " weeks (avg) ";
      }
      else {
        dbgTime << realSecondsAvg / SECONDS_PER_YEAR << " years (avg) ";
      }

      dbgTime << "to calculate." << "\n";
    }

    d_prevWallTime = d_wallTime;
  }

  d_n++;

} // end printSimulationStats()


//______________________________________________________________________
//
void
SimulationController::getMemoryStats ( int timestep, bool create )
{
  unsigned long memUse, highwater, maxMemUse;
  d_scheduler->checkMemoryUse( memUse, highwater, maxMemUse );

  d_sharedState->d_runTimeStats[ SimulationState::SCIMemoryUsed ] =
    memUse;
  d_sharedState->d_runTimeStats[ SimulationState::SCIMemoryMaxUsed ] =
    maxMemUse;
  d_sharedState->d_runTimeStats[ SimulationState::SCIMemoryHighwater ] =
    highwater;

  if( ProcessInfo::isSupported(ProcessInfo::MEM_SIZE) )
    d_sharedState->d_runTimeStats[ SimulationState::MemoryUsed ] =
      ProcessInfo::getMemoryUsed();
  if( ProcessInfo::isSupported(ProcessInfo::MEM_RSS) )
    d_sharedState->d_runTimeStats[ SimulationState::MemoryResident ] =
      ProcessInfo::getMemoryResident();
}

//______________________________________________________________________
//
void
SimulationController::getPAPIStats( )
{
#ifdef USE_PAPI_COUNTERS
  int retp = PAPI_read(d_eventSet, d_eventValues);

  if (retp != PAPI_OK) {
    proc0cout << "Error: Cannot read PAPI event set!" << endl
              << "       Error code = " << retp << " (" << d_papiErrorCodes.find(retp)->second << ")" << endl;
    throw PapiInitializationError("PAPI read error. Unable to read hardware event set values.", __FILE__, __LINE__);
  }
  else {
    d_sharedState->d_runTimeStats[ SimulationState::TotalFlops ] =
      (double) d_eventValues[d_papiEvents.find(PAPI_FP_OPS)->second.eventValueIndex];
    d_sharedState->d_runTimeStats[ SimulationState::TotalVFlops ] =
      (double) d_eventValues[d_papiEvents.find(PAPI_DP_OPS)->second.eventValueIndex];
    d_sharedState->d_runTimeStats[ SimulationState::L2Misses ] =
      (double) d_eventValues[d_papiEvents.find(PAPI_L2_TCM)->second.eventValueIndex];
    d_sharedState->d_runTimeStats[ SimulationState::L3Misses ] =
      (double) d_eventValues[d_papiEvents.find(PAPI_L3_TCM)->second.eventValueIndex];
  }

  // zero the values in the hardware counter event set array
  retp = PAPI_reset(d_eventSet);

  if (retp != PAPI_OK) {
    proc0cout << "WARNNING: Cannot reset PAPI event set!" << endl
              << "          Error code = " << retp << " ("
	      << d_papiErrorCodes.find(retp)->second << ")" << endl;

    throw PapiInitializationError("PAPI reset error on hardware event set. "
				  "Unable to reset event set values.",
				  __FILE__, __LINE__);
  }
#endif
}

} // namespace Uintah
