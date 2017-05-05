/*
 * The MIT License
 *
 * Copyright (c) 1997-2017 The University of Utah
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

#include <Core/DataArchive/DataArchive.h>
#include <Core/Exceptions/InternalError.h>
#include <Core/Exceptions/PapiInitializationError.h>
#include <Core/Grid/Grid.h>
#include <Core/Grid/SimulationState.h>
#include <Core/Grid/SimulationTime.h>
#include <Core/Grid/Variables/VarTypes.h>
#include <Core/OS/Dir.h>
#include <Core/OS/ProcessInfo.h>
#include <Core/Parallel/Parallel.h>
#include <Core/Parallel/ProcessorGroup.h>
#include <Core/ProblemSpec/ProblemSpec.h>
#include <Core/Util/DebugStream.h>

#include <CCA/Ports/LoadBalancerPort.h>
#include <CCA/Ports/Output.h>
#include <CCA/Ports/ProblemSpecInterface.h>
#include <CCA/Ports/Regridder.h>
#include <CCA/Ports/Scheduler.h>
#include <CCA/Ports/SimulationInterface.h>

#include <CCA/Components/Schedulers/MPIScheduler.h>

#include <sci_defs/visit_defs.h>

#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <sys/param.h>
#include <vector>

#include <pwd.h>

#define SECONDS_PER_MINUTE        60.0
#define SECONDS_PER_HOUR        3600.0
#define SECONDS_PER_DAY        86400.0
#define SECONDS_PER_WEEK      604800.0
#define SECONDS_PER_YEAR    31536000.0

using namespace std;

static Uintah::DebugStream dbg(     "SimulationStats",            true  );
static Uintah::DebugStream dbgTime( "SimulationTimeStats",        false );
static Uintah::DebugStream simdbg(  "SimulationController",       false );
static Uintah::DebugStream stats(   "ComponentTimings",           false );
static Uintah::DebugStream istats(  "IndividualComponentTimings", false );
extern Uintah::DebugStream amrout;

namespace Uintah {

SimulationController::SimulationController( const ProcessorGroup * myworld,
                                                  bool             doAMR,
                                                  ProblemSpecP     pspec ) :
  UintahParallelComponent( myworld ), d_ups( pspec ), d_doAMR( doAMR )
{
  //initialize the overhead percentage
  overheadIndex = 0;
  
  for(int i=0; i<OVERHEAD_WINDOW; ++i) {
    double x = (double) i / (double) (OVERHEAD_WINDOW/2);
    overheadValues[i] = 0;
    overheadWeights[i]= 8.0 - x*x*x;
  }

  d_nSamples               = 0;

  d_delt                   = 0;
  d_prev_delt              = 0;

  d_simTime                = 0;
  d_startSimTime           = 0;
  
  d_restarting             = false;
  d_reduceUda              = false;
  d_doMultiTaskgraphing    = false;
  d_archive                = nullptr;
  d_sim                    = 0;

  d_grid_ps                = d_ups->findBlock( "Grid" );

  d_timeinfo    = scinew SimulationTime( d_ups );
  d_sharedState = scinew SimulationState( d_ups );

  d_sharedState->setSimulationTime( d_timeinfo );
  
#ifdef USE_PAPI_COUNTERS
  /*
   * Setup PAPI events to track.
   *
   * Here and in printSimulationStats() are the only places code needs
   * to be added for additional events to track. Everything is
   * parameterized and hopefully robust enough to handle unsupported
   * events on different architectures. Only supported events will
   * report stats in printSimulationStats().
   *
   * NOTE: All desired events may not be supported for a particular
   *       architecture and bad things, happen, e.g. misaligned
   *       event value array indices when an event can be queried
   *       but not added to an event set, hence the PapiEvent
   *       struct, map and logic in printSimulationStats().
   *
   *       On some platforms, errors about resource limitations may be
   *       encountered, and is why we limit this instrumentation to
   *       four events now (seems stable). At some point we will look
   *       into the cost of multiplexing with PAPI, which will allow a
   *       user to count more events than total physical counters by
   *       time sharing the existing counters. This comes at some loss
   *       in precision.
   *
   * PAPI_FP_OPS - floating point operations executed
   * PAPI_DP_OPS - floating point operations executed; optimized to count scaled double precision vector operations
   * PAPI_L2_TCM - level 2 total cache misses
   * PAPI_L3_TCM - level 3 total cache misses
   */
  d_papiEvents.insert(pair<int, PapiEvent>(PAPI_FP_OPS, PapiEvent("PAPI_FP_OPS", "FLOPS")));
  d_papiEvents.insert(pair<int, PapiEvent>(PAPI_DP_OPS, PapiEvent("PAPI_DP_OPS", "VFLOPS")));
  d_papiEvents.insert(pair<int, PapiEvent>(PAPI_L2_TCM, PapiEvent("PAPI_L2_TCM", "L2CacheMisses")));
  d_papiEvents.insert(pair<int, PapiEvent>(PAPI_L3_TCM, PapiEvent("PAPI_L3_TCM", "L3CacheMisses")));

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

  d_eventValues = scinew long long[d_papiEvents.size()];
  d_eventSet = PAPI_NULL;
  int retp = -1;

  // some PAPI boiler plate
  retp = PAPI_library_init(PAPI_VER_CURRENT);
  if (retp != PAPI_VER_CURRENT) {
    proc0cout << "Error: Cannot initialize PAPI library!\n"
              << "       Error code = " << retp << " (" << d_papiErrorCodes.find(retp)->second << ")\n";
    throw PapiInitializationError("PAPI library initialization error occurred. Check that your PAPI library can be initialized correctly.", __FILE__, __LINE__);
  }
  retp = PAPI_thread_init(pthread_self);
  if (retp != PAPI_OK) {
    if (d_myworld->myrank() == 0) {
      cout << "Error: Cannot initialize PAPI thread support!\n"
           << "       Error code = " << retp << " (" << d_papiErrorCodes.find(retp)->second << ")\n";
    }
    if (Parallel::getNumThreads() > 1) {
      throw PapiInitializationError("PAPI Pthread initialization error occurred. Check that your PAPI build supports Pthreads.", __FILE__, __LINE__);
    }
  }

  // query all the events to find that are supported, flag those that
  // are unsupported
  for (map<int, PapiEvent>::iterator iter=d_papiEvents.begin(); iter!=d_papiEvents.end(); iter++) {
    retp = PAPI_query_event(iter->first);
    if (retp != PAPI_OK) {
      proc0cout << "WARNNING: Cannot query PAPI event: " << iter->second.name << "!\n"
                << "          Error code = " << retp << " (" << d_papiErrorCodes.find(retp)->second << ")\n"
                << "          No stats will be printed for " << iter->second.simStatName << "\n";
    } else {
      iter->second.isSupported = true;
    }
  }

  // create a new empty PAPI event set
  retp = PAPI_create_eventset(&d_eventSet);
  if (retp != PAPI_OK) {
    proc0cout << "Error: Cannot create PAPI event set!\n"
              << "       Error code = " << retp << " (" << d_papiErrorCodes.find(retp)->second << ")\n";
    throw PapiInitializationError("PAPI event set creation error. Unable to create hardware counter event set.", __FILE__, __LINE__);
  }

  /* Iterate through PAPI events that are supported, flag those that
   *   cannot be added.  There are situations where an event may be
   *   queried but not added to an event set, this is the purpose of
   *   this block of code.
   */
  int index = 0;
  for (map<int, PapiEvent>::iterator iter = d_papiEvents.begin(); iter != d_papiEvents.end(); iter++) {
    if (iter->second.isSupported) {
      retp = PAPI_add_event(d_eventSet, iter->first);
      if (retp != PAPI_OK) { // this means the event queried OK but could not be added
        if (d_myworld->myrank() == 0) {
          cout << "WARNNING: Cannot add PAPI event: " << iter->second.name << "!\n"
               << "          Error code = " << retp << " (" << d_papiErrorCodes.find(retp)->second << ")\n"
               << "          No stats will be printed for " << iter->second.simStatName << "\n";
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
    proc0cout << "WARNNING: Cannot start PAPI event set!\n"
              << "          Error code = " << retp << " (" << d_papiErrorCodes.find(retp)->second << ")\n";
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
SimulationController::setReduceUdaFlags( const string & fromDir )
{
  d_doAMR       = false;
  d_reduceUda   = true;
  d_fromDir     = fromDir;
}

//______________________________________________________________________
//

void
SimulationController::doRestart( const string & restartFromDir, int timestep,
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
// Determines if the time step was the last one. 
bool
SimulationController::isLast( void )
{
  double walltime = walltimers.GetWallTime();

  // When using the wall clock time, rank 0 determines the time and
  // sends it to all other ranks.
  Uintah::MPI::Bcast( &walltime, 1, MPI_DOUBLE, 0, d_myworld->getComm() );

  return ( ( d_simTime >= d_timeinfo->maxTime ) ||
	   ( d_sharedState->getCurrentTopLevelTimeStep() >=
	     d_timeinfo->maxTimestep ) ||
	   ( d_timeinfo->max_wall_time > 0 &&
	     walltime >= d_timeinfo->max_wall_time ) );
}

//______________________________________________________________________
//
// Determines if the time step may be the last one. The simulation
// time, d_delt, and the time step are known. The only real unknown is
// the wall time for the simulation calculation. The best guess is
// based on the ExpMovingAverage of the previous time steps.
//
// MaybeLast should be called before any time step work is done.

bool
SimulationController::maybeLast( void )
{
  // The predicted time is a best guess at what the wall time will be
  // when the time step is finished. It is currently used only for
  // outputing and checkpointing. Both of which typically take much
  // longer than the simulation calculation.
  double walltime = walltimers.GetWallTime() +
    1.5 * walltimers.ExpMovingAverage().seconds();

  // When using the wall clock time, rank 0 determines the time and
  // sends it to all other ranks.
  Uintah::MPI::Bcast( &walltime, 1, MPI_DOUBLE, 0, d_myworld->getComm() );

  return ( (d_simTime+d_delt >= d_timeinfo->maxTime) ||

	   (d_sharedState->getCurrentTopLevelTimeStep() + 1 >=
	     d_timeinfo->maxTimestep) ||

	   (d_timeinfo->max_wall_time > 0 &&
	    walltime >= d_timeinfo->max_wall_time) );
}

//______________________________________________________________________
//

void
SimulationController::preGridSetup( void )
{
  d_output = dynamic_cast<Output*>(getPort("output"));
  if( !d_output ) {
    throw InternalError("dynamic_cast of 'd_output' failed!",
			__FILE__, __LINE__);
  }

  d_output->problemSetup( d_ups, d_sharedState.get_rep() );

  d_scheduler = dynamic_cast<Scheduler*>(getPort("scheduler"));
  d_scheduler->problemSetup(d_ups, d_sharedState);
    
  ProblemSpecP amr_ps = d_ups->findBlock("AMR");
  if( amr_ps ) {
    amr_ps->get( "doMultiTaskgraphing", d_doMultiTaskgraphing );
  }

  d_sim = dynamic_cast<SimulationInterface*>( getPort( "sim" ) );
  if( !d_sim ) {
    throw InternalError( "No simulation component", __FILE__, __LINE__ );
  }
#ifdef HAVE_VISIT
  if( d_sharedState->getVisIt() )
  {
    d_sharedState->d_debugStreams.push_back( &dbg );
    d_sharedState->d_debugStreams.push_back( &dbgTime );
    d_sharedState->d_debugStreams.push_back( &simdbg );
    d_sharedState->d_debugStreams.push_back( &stats );
    d_sharedState->d_debugStreams.push_back( &istats );
  }
#endif
} // end preGridSetup()

//______________________________________________________________________
//

void
SimulationController::gridSetup( void )
{
  if( d_restarting ) {
    // Create the DataArchive here, and store it, as we use it a few times...
    // We need to read the grid before ProblemSetup, and we can't load all
    // the data until after problemSetup, so we have to do a few 
    // different DataArchive operations

    Dir restartFromDir( d_fromDir );
    Dir checkpointRestartDir = restartFromDir.getSubdir( "checkpoints" );
    d_archive = scinew DataArchive( checkpointRestartDir.getName(),
                                    d_myworld->myrank(), d_myworld->size() );

    vector<int>    indices;
    vector<double> times;

    try {
      d_archive->queryTimesteps( indices, times );
    }
    catch( InternalError & ie ) {
      cerr << "\n";
      cerr << "An internal error was caught while trying to restart:\n";
      cerr << "\n";
      cerr << ie.message() << "\n";
      cerr << "This most likely means that the simulation UDA that you have specified\n";
      cerr << "to use for the restart does not have any checkpoint data in it.  Look\n";
      cerr << "in <uda>/checkpoints/ for timestep directories (t#####/) to verify.\n";
      cerr << "\n";
      Parallel::exitAll(1);
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
      ostringstream message;
      message << "Timestep " << d_restartTimestep << " not found";
      throw InternalError(message.str(), __FILE__, __LINE__);
    }

    // tsaad & bisaac: At this point, and during a restart, there not
    // legitimate load balancer. This means that the grid obtained
    // from the data archiver will global domain BCs on every MPI Rank
    // - i.e. every rank will have knowledge of ALL OTHER patches and
    // their boundary conditions.  This leads to a noticeable and
    // unacceptable increase in memory usage especially when hundreds
    // of boundaries (and boundary conditions) are present. That being
    // said, we query the grid WITHOUT requiring boundary
    // conditions. Once that is done, a legitimate load balancer will
    // be created later on - after which we use said balancer and
    // assign BCs to the grid.  NOTE the "false" argument below.
    d_currentGridP = d_archive->queryGrid( d_restartIndex, d_ups, false );
  }
  else /* if( !d_restarting ) */ {
    d_currentGridP = scinew Grid();
        
    d_sim->preGridProblemSetup( d_ups, d_currentGridP, d_sharedState );
    
    d_currentGridP->problemSetup( d_ups, d_myworld, d_doAMR );
  }

  if( d_currentGridP->numLevels() == 0 ) {
    throw InternalError("No problem (no levels in grid) specified.", __FILE__, __LINE__);
  }
   
  // Print out meta data
  if ( d_myworld->myrank() == 0 ) {
    d_currentGridP->printStatistics();
    amrout << "Restart grid\n" << *d_currentGridP.get_rep() << "\n";
  }

  // set the dimensionality of the problem.
  IntVector low, high, size;
  d_currentGridP->getLevel(0)->findCellIndexRange(low, high);
  size = high-low - d_currentGridP->getLevel(0)->getExtraCells()*IntVector(2,2,2);
  d_sharedState->setDimensionality(size[0] > 1, size[1] > 1, size[2] > 1);
} // end gridSetup()

//______________________________________________________________________
//

void
SimulationController::postGridSetup()
{
  // Set up regridder with initial information about grid.  Do before
  // sim - so that Switcher (being a sim) can reset the state of the
  // regridder
  d_regridder = dynamic_cast<Regridder*>( getPort("regridder") );

  if( d_regridder ) {
    d_regridder->problemSetup( d_ups, d_currentGridP, d_sharedState );
  }

  // Initialize load balancer.  Do the initialization here because the
  // dimensionality in the shared state is known, and that
  // initialization time is needed. In addition, do this step after
  // regridding as the minimum patch size that the regridder will
  // create will be known.
  d_lb = d_scheduler->getLoadBalancer();
  d_lb->problemSetup( d_ups, d_currentGridP, d_sharedState );

  ProblemSpecP restart_prob_spec_for_component = 0;

  if( d_restarting ) {
    // Do these before calling archive->restartInitialize, becasue
    // problemSetup creates VarLabels the DA needs.
    restart_prob_spec_for_component =
      d_archive->getTimestepDocForComponent( d_restartIndex );
  }

  // Pass the restart_prob_spec_for_component to the component's
  // problemSetup.  For restarting, pull the <MaterialProperties> from
  // the restart_prob_spec.  If it is not available, then we will pull
  // the properties from the d_ups instead.  Needs to be done before
  // DataArchive::restartInitialize.
  d_sim->problemSetup(d_ups, restart_prob_spec_for_component, d_currentGridP,
		      d_sharedState);

  if( d_restarting ) {
    simdbg << "Restarting... loading data\n";    
    d_archive->restartInitialize( d_restartIndex, d_currentGridP, d_scheduler->get_dw(1),
				  d_lb, &d_startSimTime );
      
    // Set the delt to what it was in the last simulation.  If in the last 
    // sim we were clamping delt based on the values of prevDelt, then
    // delt will be off if it doesn't match.
    d_delt = d_archive->getOldDelt( d_restartIndex );

    // Set the time step to the restart time step.
    d_sharedState->setCurrentTopLevelTimeStep( d_restartTimestep );

    // Tell the scheduler the generation of the re-started simulation.
    // (Add +1 because the scheduler will be starting on the next
    // timestep.)
    d_scheduler->setGeneration( d_restartTimestep + 1);
      
    // Check to see if the user has set a restart delt
    if (d_timeinfo->override_restart_delt != 0) {
      double delt = d_timeinfo->override_restart_delt;
      proc0cout << "Overriding restart delt with " << delt << "\n";
      d_scheduler->get_dw(1)->override( delt_vartype(delt),
					d_sharedState->get_delt_label() );
    }

    // This delete is an enigma... I think if it is called then memory
    // is not leaked, but sometimes if it it is called, then
    // everything segfaults...
    //
    // delete d_archive;
  }
  else /* if( !d_restarting ) */ {
    d_startSimTime = 0;
    
    d_delt = 0;
    
    d_sharedState->setCurrentTopLevelTimeStep( 0 );
  }
    
  // Finalize the shared state/materials
  d_sharedState->finalizeMaterials();
    
  // Done after the sim->problemSetup to get defaults into the
  // input.xml, which it writes along with index.xml
  d_output->initializeOutput(d_ups);

  if( d_restarting ) {
    Dir dir( d_fromDir );
    d_output->restartSetup( dir, 0, d_restartTimestep, d_startSimTime,
			    d_restartFromScratch, d_restartRemoveOldDir );
  }
} // end postGridSetup()

//______________________________________________________________________
//

void
SimulationController::getNextDeltaT( void )
{
  d_prev_delt = d_delt;

  // Retrieve the next delta T and adjust it based on timeinfo
  // parameters.
  DataWarehouse* newDW = d_scheduler->getLastDW();
  delt_vartype delt_var;
  newDW->get( delt_var, d_sharedState->get_delt_label() );
  d_delt = delt_var;

  // Adjust the delt
  d_delt *= d_timeinfo->delt_factor;
      
  // Check to see if the new delt is below the delt_min
  if( d_delt < d_timeinfo->delt_min ) {
    proc0cout << "WARNING: raising delt from " << d_delt;
    
    d_delt = d_timeinfo->delt_min;
    
    proc0cout << " to minimum: " << d_delt << '\n';
  }

  // Check to see if the new delt was increased too much over the
  // previous delt
  double delt_tmp = (1.0+d_timeinfo->max_delt_increase) * d_prev_delt;
  
  if( d_prev_delt > 0.0 &&
      d_timeinfo->max_delt_increase < 1.e90 &&
      d_delt > delt_tmp ) {
    proc0cout << "WARNING (a): lowering delt from " << d_delt;
    
    d_delt = delt_tmp;
    
    proc0cout << " to maxmimum: " << d_delt
              << " (maximum increase of " << d_timeinfo->max_delt_increase
              << ")\n";
  }

  // Check to see if the new delt exceeds the max_initial_delt
  if( d_simTime <= d_timeinfo->initial_delt_range &&
      d_delt > d_timeinfo->max_initial_delt ) {
    proc0cout << "WARNING (b): lowering delt from " << d_delt ;

    d_delt = d_timeinfo->max_initial_delt;

    proc0cout<< " to maximum: " << d_delt
	     << " (for initial timesteps)\n";
  }

  // Check to see if the new delt exceeds the delt_max
  if( d_delt > d_timeinfo->delt_max ) {
    proc0cout << "WARNING (c): lowering delt from " << d_delt;

    d_delt = d_timeinfo->delt_max;
    
    proc0cout << " to maximum: " << d_delt << '\n';
  }

  // Clamp delt to match the requested output and/or checkpoint times
  if( d_timeinfo->clamp_time_to_output ) {

    // Clamp to the output time
    double nextOutput = d_output->getNextOutputTime();
    if (nextOutput != 0 && d_simTime + d_delt > nextOutput) {
      proc0cout << "WARNING (d): lowering delt from " << d_delt;

      d_delt = nextOutput - d_simTime;

      proc0cout << " to " << d_delt
                << " to line up with output time\n";
    }

    // Clamp to the checkpoint time
    double nextCheckpoint = d_output->getNextCheckpointTime();
    if (nextCheckpoint != 0 && d_simTime + d_delt > nextCheckpoint) {
      proc0cout << "WARNING (d): lowering delt from " << d_delt;

      d_delt = nextCheckpoint - d_simTime;

      proc0cout << " to " << d_delt
                << " to line up with checkpoint time\n";
    }
  }
  
  // Clamp delt to the max end time,
  if (d_timeinfo->end_at_max_time &&
      d_simTime + d_delt > d_timeinfo->maxTime) {
    d_delt = d_timeinfo->maxTime - d_simTime;
  }

  // Write the new delt to the data warehouse
  newDW->override( delt_vartype(d_delt), d_sharedState->get_delt_label() );
}

//______________________________________________________________________
//

void
SimulationController::ReportStats( bool header /* = false */ )
{
  // Get and reduce the performace run time stats
  getMemoryStats();
  getPAPIStats();

  d_sharedState->d_runTimeStats.reduce(d_regridder &&
				       d_regridder->useDynamicDilation(),
				       d_myworld );

  d_sharedState->d_otherStats.reduce(d_regridder &&
				     d_regridder->useDynamicDilation(),
				     d_myworld );

  // Reduce the mpi run time stats.
  MPIScheduler * mpiScheduler =
    dynamic_cast<MPIScheduler*>( d_scheduler.get_rep() );
  
  if( mpiScheduler )
    mpiScheduler->mpi_info_.reduce( d_regridder &&
				    d_regridder->useDynamicDilation(),
				    d_myworld );

  // Print MPI statistics
  d_scheduler->printMPIStats();
  
  // Print the stats for this time step
  if( d_myworld->myrank() == 0 && header ) {
    dbg << std::endl;
    dbg << "Simulation and run time stats are reported "
	<< "at the end of each time step" << std::endl;
    dbg << "EMA == Wall time as an exponential moving average "
	<< "using a window of the last " << walltimers.getWindow()
	<< " time steps" << std::endl;

    dbg.flush();
    cout.flush();
  }
  
  ReductionInfoMapper< SimulationState::RunTimeStat, double > &runTimeStats =
    d_sharedState->d_runTimeStats;

  ReductionInfoMapper< unsigned int, double > &otherStats =
    d_sharedState->d_otherStats;

  // With the sum reduces, use double, since with memory it is possible that
  // it will overflow
  double        avg_memused =
    runTimeStats.getAverage( SimulationState::SCIMemoryUsed );
  unsigned long max_memused =
    runTimeStats.getMaximum( SimulationState::SCIMemoryUsed );
  int           max_memused_rank =
    runTimeStats.getRank( SimulationState::SCIMemoryUsed );

  double        avg_highwater =
    runTimeStats.getAverage( SimulationState::SCIMemoryHighwater );
  unsigned long max_highwater =
    runTimeStats.getMaximum( SimulationState::SCIMemoryHighwater );
  int           max_highwater_rank =
    runTimeStats.getRank( SimulationState::SCIMemoryHighwater );
    
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
  
  // Set the overhead percentage. Ignore the first sample as that is
  // for initalization.
  if( d_nSamples )
  {
    overheadValues[overheadIndex] = percent_overhead;

    double overhead = 0;
    double weight = 0;

    int t = min(d_nSamples, OVERHEAD_WINDOW);
    
    // Calcualte total weight by incrementing through the overhead
    // sample array backwards and multiplying samples by the weights
    for( int i=0; i<t; ++i )
    {
      unsigned int index = (overheadIndex-i+OVERHEAD_WINDOW) % OVERHEAD_WINDOW;
      overhead += overheadValues[index] * overheadWeights[i];
      weight += overheadWeights[i];
    }

    // Increment the overhead index
    overheadIndex = (overheadIndex+1) % OVERHEAD_WINDOW;

    d_sharedState->setOverheadAvg( overhead / weight );
  } 

  // Output timestep statistics...
  if (istats.active())
  {
    istats << "Run time performance stats" << std::endl;

    for (unsigned int i=0; i<runTimeStats.size(); i++)
    {
      SimulationState::RunTimeStat e = (SimulationState::RunTimeStat) i;
      
      if (runTimeStats[e] > 0)
      {
        istats << "rank: " << d_myworld->myrank() << " "
	       << left << setw(19) << runTimeStats.getName(e)
	       << " [" << runTimeStats.getUnits(e) << "]: "
	       << runTimeStats[e] << "\n";
      }
    }

    if( otherStats.size() )
      istats << "Other performance stats" << std::endl;
      
    for (unsigned int i=0; i<otherStats.size(); i++)
    {
      if (otherStats[i] > 0)
      {
        istats << "rank: " << d_myworld->myrank() << " "
	       << left << setw(19) << otherStats.getName(i)
	       << " [" << otherStats.getUnits(i) << "]: "
	       << otherStats[i] << "\n";
      }
    }
  } 

  // Update the moving average and get the wall time for this time step.
  Timers::nanoseconds timeStep = walltimers.updateExpMovingAverage();

  if( d_myworld->myrank() == 0 )
  {
    ostringstream message;
    message << left
	    << "Timestep "   << setw(8)
	    << d_sharedState->getCurrentTopLevelTimeStep()
	    << "Time="       << setw(12) << d_simTime
//	    << "delT="       << setw(12) << d_prev_delt
	    << "Next delT="  << setw(12) << d_delt

	    << "Wall Time = " << setw(10) << walltimers.GetWallTime()
	    // << "All Time steps= " << setw(12) << walltimers.TimeStep().seconds()
	    // << "Current Time Step= " << setw(12) << timeStep.seconds()
	    << "EMA="        << setw(12) << walltimers.ExpMovingAverage().seconds()
	    // << "In-situ Time = " << setw(12) << walltimers.InSitu().seconds()
      ;

    // Report on the memory used.
    if (avg_memused == max_memused && avg_highwater == max_highwater) {
      message << "Memory Use=" << setw(8)
	      << ProcessInfo::toHumanUnits((unsigned long) avg_memused);

      if(avg_highwater)
	message << "    Highwater Memory Use=" << setw(8)
		<< ProcessInfo::toHumanUnits((unsigned long) avg_highwater);
    }
    else {
      message << "Memory Used=" << setw(10)
	      << ProcessInfo::toHumanUnits((unsigned long) avg_memused)
	      << " (avg) " << setw(10)
	      << ProcessInfo::toHumanUnits(max_memused)
	      << " (max on rank:" << setw(6) << max_memused_rank << ")";

      if(avg_highwater)
	message << "    Highwater Memory Used=" << setw(10)
		<< ProcessInfo::toHumanUnits((unsigned long)avg_highwater)
		<< " (avg) " << setw(10)
		<< ProcessInfo::toHumanUnits(max_highwater)
		<< " (max on rank:" << setw(6) << max_highwater_rank << ")";
    }

    dbg << message.str() << "\n";
    dbg.flush();
    cout.flush();

    // Ignore the first sample as that is for initalization.
    if (stats.active() && d_nSamples) {

      stats << "Run time performance stats" << std::endl;
      
      stats << "  " << left
	    << setw(21) << "Description"
	    << setw(15) << "Units"
	    << setw(15) << "Average"
	    << setw(15) << "Maximum"
	    << setw(13) << "Rank"
	    << setw(13) << "100*(1-ave/max) '% load imbalance'"
	    << "\n";
      
      for (unsigned int i=0; i<runTimeStats.size(); ++i)
      {
	SimulationState::RunTimeStat e = (SimulationState::RunTimeStat) i;
	
	if (runTimeStats.getMaximum(e) > 0)
	{
	  stats << "  " << left
                << setw(21) << runTimeStats.getName(e)
		<< "[" << setw(10) << runTimeStats.getUnits(e) << "]"
		<< " : " << setw(12) << runTimeStats.getAverage(e)
		<< " : " << setw(12) << runTimeStats.getMaximum(e)
		<< " : " << setw(10) << runTimeStats.getRank(e)
		<< " : " << setw(10)
		<< 100.0 * (1.0 - (runTimeStats.getAverage(e) /
				   runTimeStats.getMaximum(e)))
		<< "\n";
	}
      }
      
      // Report the overhead percentage.
      if( !std::isnan(d_sharedState->getOverheadAvg()) ) {
        stats << "  Percentage of time spent in overhead : "
              << d_sharedState->getOverheadAvg()*100.0 <<  "\n";
      }

      if( otherStats.size() )
	stats << "Other performance stats" << std::endl;
      
      for (unsigned int i=0; i<otherStats.size(); ++i)
      {
	if (otherStats.getMaximum(i) > 0)
	{
	  stats << "  " << left
                << setw(21) << otherStats.getName(i)
		<< "[" << setw(10) << otherStats.getUnits(i) << "]"
		<< " : " << setw(12) << otherStats.getAverage(i)
		<< " : " << setw(12) << otherStats.getMaximum(i)
		<< " : " << setw(10) << otherStats.getRank(i)
		<< " : " << setw(10)
		<< 100.0 * (1.0 - (otherStats.getAverage(i) /
				   otherStats.getMaximum(i)))
		<< "\n";
	}
      }
    }
  
    // Ignore the first sample as that is for initalization.
    if (dbgTime.active() && d_nSamples ) {
      double realSecondsNow =
	timeStep.seconds() / d_delt;
      double realSecondsAvg =
	walltimers.TimeStep().seconds() / (d_simTime-d_startSimTime);

      dbgTime << "1 simulation second takes ";

      dbgTime << left << showpoint << setprecision(3) << setw(4);

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

      dbgTime << setw(4);

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
  }

  ++d_nSamples;

} // end printSimulationStats()

//______________________________________________________________________
//

void
SimulationController::getMemoryStats( bool create /* = false */ )
{
  unsigned long memUsed, highwater, maxMemUsed;
  d_scheduler->checkMemoryUse(memUsed, highwater, maxMemUsed);

  d_sharedState->d_runTimeStats[SimulationState::SCIMemoryUsed] = memUsed;
  d_sharedState->d_runTimeStats[SimulationState::SCIMemoryMaxUsed] = maxMemUsed;
  d_sharedState->d_runTimeStats[SimulationState::SCIMemoryHighwater] = highwater;

  if (ProcessInfo::isSupported(ProcessInfo::MEM_SIZE)) {
    d_sharedState->d_runTimeStats[SimulationState::MemoryUsed] = ProcessInfo::getMemoryUsed();
  }

  if (ProcessInfo::isSupported(ProcessInfo::MEM_RSS)) {
    d_sharedState->d_runTimeStats[SimulationState::MemoryResident] = ProcessInfo::getMemoryResident();
  }

  // Get memory stats for each proc if MALLOC_PERPROC is in the environment.
  if (getenv("MALLOC_PERPROC")) {
    ostream* mallocPerProcStream = nullptr;
    char* filenamePrefix = getenv("MALLOC_PERPROC");

    if (!filenamePrefix || strlen(filenamePrefix) == 0) {
      mallocPerProcStream = &dbg;
    }
    else {
      char filename[256];
      sprintf(filename, "%s.%d", filenamePrefix, d_myworld->myrank());

      if ( create ) {
        mallocPerProcStream = scinew ofstream(filename, ios::out | ios::trunc);
      }
      else {
        mallocPerProcStream = scinew ofstream(filename, ios::out | ios::app);
      }

      if( !mallocPerProcStream ) {
        delete mallocPerProcStream;
        mallocPerProcStream = &dbg;
      }
    }

    *mallocPerProcStream << "Proc " << d_myworld->myrank() << "   ";
    *mallocPerProcStream << "Timestep "
			 << d_sharedState->getCurrentTopLevelTimeStep() << "   ";

    if (ProcessInfo::isSupported(ProcessInfo::MEM_SIZE)) {
      *mallocPerProcStream << "Size " << ProcessInfo::getMemoryUsed() << "   ";
    }

    if (ProcessInfo::isSupported(ProcessInfo::MEM_RSS)) {
      *mallocPerProcStream << "RSS " << ProcessInfo::getMemoryResident() << "   ";
    }

    *mallocPerProcStream << "Sbrk " << (char*)sbrk(0) - d_scheduler->getStartAddr() << "   ";
#ifndef DISABLE_SCI_MALLOC
    *mallocPerProcStream << "Sci_Malloc_MemUsed " << memUsed << "   ";
    *mallocPerProcStream << "Sci_Malloc_MaxMemUsed " << maxMemUsed << "   ";
    *mallocPerProcStream << "Sci_Malloc_Highwater " << highwater;
#endif
    *mallocPerProcStream << "\n";

    if( mallocPerProcStream != &dbg ) {
      delete mallocPerProcStream;
    }
  }
}

//______________________________________________________________________
//
  
void
SimulationController::getPAPIStats( )
{
#ifdef USE_PAPI_COUNTERS
  int retp = PAPI_read(d_eventSet, d_eventValues);

  if (retp != PAPI_OK) {
    proc0cout << "Error: Cannot read PAPI event set!\n"
              << "       Error code = " << retp << " (" << d_papiErrorCodes.find(retp)->second << ")\n";
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
    proc0cout << "WARNNING: Cannot reset PAPI event set!\n"
              << "          Error code = " << retp << " ("
	      << d_papiErrorCodes.find(retp)->second << ")\n";

    throw PapiInitializationError( "PAPI reset error on hardware event set. "
                                   "Unable to reset event set values.",
                                   __FILE__, __LINE__ );
  }
#endif
}
  

//______________________________________________________________________
//
  
#ifdef HAVE_VISIT
bool
SimulationController::CheckInSitu( visit_simulation_data *visitSimData,
				   bool first )
{
    // If VisIt has been included into the build, check the lib sim
    // state to see if there is a connection and if so check to see if
    // anything needs to be done.
    if( d_sharedState->getVisIt() )
    {
      // Note this timer is used as a laptimer so the stop must come
      // after the lap time is taken so not to affect it.
      walltimers.TimeStep.stop();

      walltimers.InSitu.start();
  
      // Update all of the simulation grid and time dependent variables.
      visit_UpdateSimData( visitSimData, d_currentGridP,
			   d_simTime, d_prev_delt, d_delt,
			   first, isLast() );
      
      // Check the state - if the return value is true the user issued
      // a termination.
      if( visit_CheckState( visitSimData ) )
	return true;

      // This function is no longer used as last is now used in the
      // UpdateSimData which in turn will flip the state.
      
      // Check to see if at the last iteration. If so stop so the
      // user can have once last chance see the data.
      // if( visitSimData->stopAtLastTimeStep && last )
      // visit_EndLibSim( visitSimData );

      // The user may have adjusted delt so get it from the data
      // warehouse. If not then this call is a no-op.
      DataWarehouse* newDW = d_scheduler->getLastDW();
      delt_vartype delt_var;
      newDW->get( delt_var, d_sharedState->get_delt_label() );
      d_delt = delt_var;

      // Add the modified variable information into index.xml file.
      getOutput()->writeto_xml_files(visitSimData->modifiedVars);

      walltimers.InSitu.stop();

      // Note this timer is used as a laptimer.
      walltimers.TimeStep.start();
    }

    return false;
}
#endif

} // namespace Uintah
