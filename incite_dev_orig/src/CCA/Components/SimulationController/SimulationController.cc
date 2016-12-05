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


#include <CCA/Components/SimulationController/SimulationController.h>
#include <CCA/Ports/LoadBalancerPort.h>
#include <CCA/Ports/Output.h>
#include <CCA/Ports/ProblemSpecInterface.h>
#include <CCA/Ports/Regridder.h>
#include <CCA/Ports/Scheduler.h>
#include <CCA/Ports/SimulationInterface.h>

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
#include <Core/Util/Time.h>

#include <sci_defs/malloc_defs.h>
#include <sci_defs/papi_defs.h> // for PAPI performance counters
#include <sci_defs/visit_defs.h>

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


namespace {

Uintah::DebugStream dbg(     "SimulationStats",            true  );
Uintah::DebugStream dbgTime( "SimulationTimeStats",        false );
Uintah::DebugStream simdbg(  "SimulationController",       false );
Uintah::DebugStream stats(   "ComponentTimings",           false );
Uintah::DebugStream istats(  "IndividualComponentTimings", false );

}


namespace Uintah {

//______________________________________________________________________
//
SimulationController::SimulationController( const ProcessorGroup * myworld
                                          ,       bool             doAMR
                                          ,       ProblemSpecP     pspec
                                          )
  : UintahParallelComponent( myworld )
  , m_ups( pspec )
  , m_do_amr( doAMR )
{
  
  for (int i = 0; i < OVERHEAD_WINDOW; ++i) {
    double x = (double)i / (double)(OVERHEAD_WINDOW / 2);
    m_overhead_values[i] = 0;
    m_overhead_weights[i] = 8.0 - x * x * x;
  }

  m_grid_ps      = m_ups->findBlock( "Grid" );
  m_time_info    = scinew SimulationTime( m_ups );
  m_shared_state = scinew SimulationState( m_ups );
  m_shared_state->setSimulationTime( m_time_info );

  
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
  m_papi_events.insert(std::pair<int, PapiEvent>(PAPI_FP_OPS, PapiEvent("PAPI_FP_OPS", "FLOPS")));
  m_papi_events.insert(std::pair<int, PapiEvent>(PAPI_DP_OPS, PapiEvent("PAPI_DP_OPS", "VFLOPS")));
  m_papi_events.insert(std::pair<int, PapiEvent>(PAPI_L2_TCM, PapiEvent("PAPI_L2_TCM", "L2CacheMisses")));
  m_papi_events.insert(std::pair<int, PapiEvent>(PAPI_L3_TCM, PapiEvent("PAPI_L3_TCM", "L3CacheMisses")));

  // For meaningful error reporting - PAPI Version: 5.1.0 has 25 error return codes:
  m_papi_error_codes.insert(std::pair<int, std::string>( 0,  "No error"));
  m_papi_error_codes.insert(std::pair<int, std::string>(-1,  "Invalid argument"));
  m_papi_error_codes.insert(std::pair<int, std::string>(-2,  "Insufficient memory"));
  m_papi_error_codes.insert(std::pair<int, std::string>(-3,  "A System/C library call failed"));
  m_papi_error_codes.insert(std::pair<int, std::string>(-4,  "Not supported by substrate"));
  m_papi_error_codes.insert(std::pair<int, std::string>(-5,  "Access to the counters was lost or interrupted"));
  m_papi_error_codes.insert(std::pair<int, std::string>(-6,  "Internal error, please send mail to the developers"));
  m_papi_error_codes.insert(std::pair<int, std::string>(-7,  "Hardware event does not exist"));
  m_papi_error_codes.insert(std::pair<int, std::string>(-8,  "Hardware event exists, but cannot be counted due to counter resource limitations"));
  m_papi_error_codes.insert(std::pair<int, std::string>(-9,  "EventSet is currently not running"));
  m_papi_error_codes.insert(std::pair<int, std::string>(-10, "EventSet is currently counting"));
  m_papi_error_codes.insert(std::pair<int, std::string>(-11, "No such EventSet available"));
  m_papi_error_codes.insert(std::pair<int, std::string>(-12, "Event in argument is not a valid preset"));
  m_papi_error_codes.insert(std::pair<int, std::string>(-13, "Hardware does not support performance counters"));
  m_papi_error_codes.insert(std::pair<int, std::string>(-14, "Unknown error code"));
  m_papi_error_codes.insert(std::pair<int, std::string>(-15, "Permission level does not permit operation"));
  m_papi_error_codes.insert(std::pair<int, std::string>(-16, "PAPI hasn't been initialized yet"));
  m_papi_error_codes.insert(std::pair<int, std::string>(-17, "Component index isn't set"));
  m_papi_error_codes.insert(std::pair<int, std::string>(-18, "Not supported"));
  m_papi_error_codes.insert(std::pair<int, std::string>(-19, "Not implemented"));
  m_papi_error_codes.insert(std::pair<int, std::string>(-20, "Buffer size exceeded"));
  m_papi_error_codes.insert(std::pair<int, std::string>(-21, "EventSet domain is not supported for the operation"));
  m_papi_error_codes.insert(std::pair<int, std::string>(-22, "Invalid or missing event attributes"));
  m_papi_error_codes.insert(std::pair<int, std::string>(-23, "Too many events or attributes"));
  m_papi_error_codes.insert(std::pair<int, std::string>(-24, "Bad combination of features"));

  m_event_values = scinew long long[m_papi_events.size()];
  m_event_set = PAPI_NULL;
  int retp = -1;

  // some PAPI boiler plate
  retp = PAPI_library_init(PAPI_VER_CURRENT);
  if (retp != PAPI_VER_CURRENT) {
    proc0cout << "Error: Cannot initialize PAPI library!\n"
              << "       Error code = " << retp << " (" << m_papi_error_codes.find(retp)->second << ")\n";
    throw PapiInitializationError("PAPI library initialization error occurred. Check that your PAPI library can be initialized correctly.", __FILE__, __LINE__);
  }
  retp = PAPI_thread_init(pthread_self);
  if (retp != PAPI_OK) {
    if (d_myworld->myrank() == 0) {
      std::cout << "Error: Cannot initialize PAPI thread support!\n"
           << "       Error code = " << retp << " (" << m_papi_error_codes.find(retp)->second << ")\n";
    }
    if (Parallel::getNumThreads() > 1) {
      throw PapiInitializationError("PAPI Pthread initialization error occurred. Check that your PAPI build supports Pthreads.", __FILE__, __LINE__);
    }
  }

  // query all the events to find that are supported, flag those that
  // are unsupported
  for (std::map<int, PapiEvent>::iterator iter=m_papi_events.begin(); iter!=m_papi_events.end(); iter++) {
    retp = PAPI_query_event(iter->first);
    if (retp != PAPI_OK) {
      proc0cout << "WARNNING: Cannot query PAPI event: " << iter->second.name << "!\n"
                << "          Error code = " << retp << " (" << m_papi_error_codes.find(retp)->second << ")\n"
                << "          No stats will be printed for " << iter->second.simStatName << "\n";
    } else {
      iter->second.isSupported = true;
    }
  }

  // create a new empty PAPI event set
  retp = PAPI_create_eventset(&m_event_set);
  if (retp != PAPI_OK) {
    proc0cout << "Error: Cannot create PAPI event set!\n"
              << "       Error code = " << retp << " (" << m_papi_error_codes.find(retp)->second << ")\n";
    throw PapiInitializationError("PAPI event set creation error. Unable to create hardware counter event set.", __FILE__, __LINE__);
  }

  /* Iterate through PAPI events that are supported, flag those that
   *   cannot be added.  There are situations where an event may be
   *   queried but not added to an event set, this is the purpose of
   *   this block of code.
   */
  int index = 0;
  for (std::map<int, PapiEvent>::iterator iter = m_papi_events.begin(); iter != m_papi_events.end(); iter++) {
    if (iter->second.isSupported) {
      retp = PAPI_add_event(m_event_set, iter->first);
      if (retp != PAPI_OK) { // this means the event queried OK but could not be added
        if (d_myworld->myrank() == 0) {
          std::cout << "WARNNING: Cannot add PAPI event: " << iter->second.name << "!\n"
               << "          Error code = " << retp << " (" << m_papi_error_codes.find(retp)->second << ")\n"
               << "          No stats will be printed for " << iter->second.simStatName << "\n";
        }
        iter->second.isSupported = false;
      } else {
        iter->second.eventValueIndex = index;
        index++;
      }
    }
  }

  retp = PAPI_start(m_event_set);
  if (retp != PAPI_OK) {
    proc0cout << "WARNNING: Cannot start PAPI event set!\n"
              << "          Error code = " << retp << " (" << m_papi_error_codes.find(retp)->second << ")\n";
    throw PapiInitializationError("PAPI event set start error. Unable to start hardware counter event set.", __FILE__, __LINE__);
  }
#endif

} // end SimulationController CTOR

//______________________________________________________________________
//
SimulationController::~SimulationController()
{
  delete m_archive;
  delete m_time_info;

#ifdef USE_PAPI_COUNTERS
  delete d_eventValues;
#endif
}
  
//______________________________________________________________________
//
void
SimulationController::setReduceUdaFlags( const std::string & fromDir )
{
  m_do_amr       = false;
  m_reduce_uda   = true;
  m_from_dir     = fromDir;
}

//______________________________________________________________________
//
void
SimulationController::doRestart( const std::string & restartFromDir
                               ,       int           timestep
                               ,       bool          fromScratch
                               ,       bool          removeOldDir
                               )
{
  m_restarting          = true;
  m_from_dir             = restartFromDir;
  m_restart_timestep     = timestep;
  m_restart_from_scratch  = fromScratch;
  m_restart_remove_old_dir = removeOldDir;
}

//______________________________________________________________________
//
void
SimulationController::preGridSetup( void )
{
  m_output = dynamic_cast<Output*>(getPort("output"));
    
  if( !m_output ) {
    std::cout << "dynamic_cast of 'm_output' failed!\n";
    throw InternalError("dynamic_cast of 'm_output' failed!", __FILE__, __LINE__);
  }


  m_output->problemSetup( m_ups, m_shared_state.get_rep() );

  m_scheduler = dynamic_cast<Scheduler*>(getPort("scheduler"));
  m_scheduler->problemSetup(m_ups, m_shared_state);
    
  ProblemSpecP amr_ps = m_ups->findBlock("AMR");
  if( amr_ps ) {
    amr_ps->get( "doMultiTaskgraphing", m_do_multi_taskgraphing );
  }

  // TODO: is there a cleaner way of doing this and should it be in postGridSetup? - APH, 10/19/16
  // Find the radiation calculation frequency for Arches and RMCRT_Test components
  ProblemSpecP bm_ps = m_ups->get("calc_frequency", m_rad_calc_frequency);
  if (!bm_ps) {
    ProblemSpecP root_ps = m_ups->getRootNode();
    ProblemSpecP cfd_ps = root_ps->findBlock("CFD");
    if (cfd_ps) {
      ProblemSpecP arches_ps = cfd_ps->findBlock("ARCHES");
      ProblemSpecP rad_src_ps = arches_ps->findBlock("TransportEqns")->findBlock("Sources")->findBlock("src");
      // find the "divQ" src block for the radiation calculation frequency
      std::string src_name = "";
      rad_src_ps->getAttribute("label", src_name);
      while (src_name != "divQ") {
        rad_src_ps = rad_src_ps->findNextBlock("src");
        rad_src_ps->getAttribute("label", src_name);
      }
      rad_src_ps->getWithDefault("calc_frequency", m_rad_calc_frequency, 1);
    }
  }

#ifdef HAVE_VISIT
  if( m_shared_state->getVisIt() ) {
    m_shared_state->d_debugStreams.push_back( &dbg );
    m_shared_state->d_debugStreams.push_back( &dbgTime );
    m_shared_state->d_debugStreams.push_back( &simdbg );
    m_shared_state->d_debugStreams.push_back( &stats );
    m_shared_state->d_debugStreams.push_back( &istats );
  }
#endif
}

//______________________________________________________________________
//
GridP
SimulationController::gridSetup( void )
{
  GridP grid;

  if( m_restarting ) {
    // Create the DataArchive here, and store it, as we use it a few times...
    // We need to read the grid before ProblemSetup, and we can't load all
    // the data until after problemSetup, so we have to do a few 
    // different DataArchive operations

    Dir restartFromDir(m_from_dir);
    Dir checkpointRestartDir = restartFromDir.getSubdir("checkpoints");
    m_archive = scinew DataArchive(checkpointRestartDir.getName(), d_myworld->myrank(), d_myworld->size());

    std::vector<int>    indices;
    std::vector<double> times;

    try {
      m_archive->queryTimesteps( indices, times );
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
      Parallel::exitAll(1);
    }

    // Find the right time to query the grid
    if (m_restart_timestep == 0) {
      m_restart_index = 0;  // timestep == 0 means use the first timestep
      // reset m_restart_timestep to what it really is
      m_restart_timestep = indices[0];
    }
    else if (m_restart_timestep == -1 && indices.size() > 0) {
      m_restart_index = (unsigned int)(indices.size() - 1);
      // reset m_restart_timestep to what it really is
      m_restart_timestep = indices[indices.size() - 1];
    }
    else {
      for (int index = 0; index < (int)indices.size(); index++)
        if (indices[index] == m_restart_timestep) {
          m_restart_index = index;
          break;
        }
    }

    if (m_restart_index == (int)indices.size()) {
      // timestep not found
      std::ostringstream message;
      message << "Timestep " << m_restart_timestep << " not found";
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
    grid = m_archive->queryGrid( m_restart_index, m_ups, false );
  }
  else {
    grid = scinew Grid();
    
    m_sim = dynamic_cast<SimulationInterface*>( getPort( "sim" ) );
    if( !m_sim ) {
      throw InternalError( "No simulation component", __FILE__, __LINE__ );
    }
    
    m_sim->preGridProblemSetup( m_ups, grid, m_shared_state );
    
    grid->problemSetup( m_ups, d_myworld, m_do_amr );
  }
  
  if (grid->numLevels() == 0) {
    throw InternalError("No problem (no levels in grid) specified.", __FILE__, __LINE__);
  }

  // Print out meta data
  if (d_myworld->myrank() == 0) {
    grid->printStatistics();
  }

  // set the dimensionality of the problem.
  IntVector low, high, size;
  grid->getLevel(0)->findCellIndexRange(low, high);
  size = high - low - grid->getLevel(0)->getExtraCells() * IntVector(2, 2, 2);
  m_shared_state->setDimensionality(size[0] > 1, size[1] > 1, size[2] > 1);

  return grid;
}

//______________________________________________________________________
//
void
SimulationController::postGridSetup( GridP & grid, double & time )
{
  // Set up regridder with initial information about grid.  Do before
  // sim - so that Switcher (being a sim) can reset the state of the
  // regridder
  m_regridder = dynamic_cast<Regridder*>(getPort("regridder"));
  if (m_regridder) {
    m_regridder->problemSetup( m_ups, grid, m_shared_state );
  }
    
  // Initialize load balancer.  Do here since we have the
  // dimensionality in the shared state, and we want that at
  // initialization time. In addition do it after regridding since we
  // need to know the minimum patch size that the regridder will
  // create
  m_lb = m_scheduler->getLoadBalancer();
  m_lb->problemSetup( m_ups, grid, m_shared_state );

  // Initialize the CFD and/or MPM components
  m_sim = dynamic_cast<SimulationInterface*>(getPort("sim"));
  if( !m_sim ) {
    throw InternalError("No simulation component", __FILE__, __LINE__);
  }

  ProblemSpecP restart_prob_spec_for_component = nullptr;

  if( m_restarting ) {
    // Do these before calling archive->restartInitialize, since problemSetup creates VarLabels the DA needs.
    restart_prob_spec_for_component = m_archive->getTimestepDocForComponent( m_restart_index );
  }

  // Pass the restart_prob_spec_for_component to the Component's
  // problemSetup.  For restarting, pull the <MaterialProperties> from
  // the restart_prob_spec.  If it is not available, then we will pull
  // the properties from the m_ups instead.  Needs to be done before
  // DataArchive::restartInitialize
  m_sim->problemSetup(m_ups, restart_prob_spec_for_component, grid, m_shared_state);

  if( m_restarting ) {
    simdbg << "Restarting... loading data\n";    
    m_archive->restartInitialize( m_restart_index, grid, m_scheduler->get_dw(1), m_lb, &time );
      
    // Set prevDelt to what it was in the last simulation.  If in the last 
    // sim we were clamping delt based on the values of prevDelt, then
    // delt will be off if it doesn't match.
    m_prev_delt = m_archive->getOldDelt( m_restart_index );

    // Set the time step to the restart time step.
    m_shared_state->setCurrentTopLevelTimeStep( m_restart_timestep );

    // Tell the scheduler the generation of the re-started simulation.
    // (Add +1 because the scheduler will be starting on the next timestep.)
    m_scheduler->setGeneration( m_restart_timestep + 1 );
      
    // Check to see if the user has set a restart delt
    if (m_time_info->override_restart_delt != 0) {
      double newdelt = m_time_info->override_restart_delt;
      proc0cout << "Overriding restart delt with " << newdelt << "\n";
      m_scheduler->get_dw(1)->override( delt_vartype(newdelt), m_shared_state->get_delt_label() );
    }

//    // This delete is an enigma... I think if it is called then memory is not leaked, but sometimes if it
//    // it is called, then everything segfaults...
//
//     delete m_archive;

  }

  // Finalize the shared state/materials
  m_shared_state->finalizeMaterials();
    
  // Done after the sim->problemSetup to get defaults into the
  // input.xml, which it writes along with index.xml
  m_output->initializeOutput(m_ups);

  if( m_restarting ) {
    Dir dir(m_from_dir);
    m_output->restartSetup( dir, 0, m_restart_timestep, time, m_restart_from_scratch, m_restart_remove_old_dir );
  }
} // end postGridSetup()

//______________________________________________________________________
//
void
SimulationController::adjustDelT( double& delt, double prev_delt, double time )
{

#if 0
  proc0cout << "maxTime = "             << m_time_info->maxTime << "\n";
  proc0cout << "initTime = "            << m_time_info->initTime << "\n";
  proc0cout << "delt_min = "            << m_time_info->delt_min << "\n";
  proc0coutt << "delt_max = "           << m_time_info->delt_max << "\n";
  proc0cout << "timestep_multiplier = " << m_time_info->delt_factor << "\n";
  proc0cout << "delt_init = "           << m_time_info->max_initial_delt << "\n";
  proc0cout << "initial_delt_range = "  << m_time_info->initial_delt_range << "\n";
  proc0cout << "max_delt_increase = "   << m_time_info->max_delt_increase << "\n";
  proc0cout << "first = "               << first << "\n";
  proc0cout << "delt = "                << delt << "\n";
  proc0cout << "prev_delt = "           << prev_delt << "\n";
#endif

  delt *= m_time_info->delt_factor;
      
  // Check to see if delt is below the delt_min
  if( delt < m_time_info->delt_min ) {
    proc0cout << "WARNING: raising delt from " << delt << " to minimum: " << m_time_info->delt_min << '\n';
    delt = m_time_info->delt_min;
  }

  // Check to see if delt was increased too much over the previous delt
  if( prev_delt > 0.0 &&
      m_time_info->max_delt_increase < 1.e90 &&
      delt > (1.0 + m_time_info->max_delt_increase) * prev_delt) {
    proc0cout << "WARNING (a): lowering delt from " << delt 
              << " to maxmimum: " << (1.0 + m_time_info->max_delt_increase) * prev_delt
              << " (maximum increase of " << m_time_info->max_delt_increase
              << ")\n";
    delt = (1 + m_time_info->max_delt_increase) * prev_delt;
  }

  // Check to see if delt exceeds the max_initial_delt
  if( time <= m_time_info->initial_delt_range &&
      delt > m_time_info->max_initial_delt ) {
    proc0cout << "WARNING (b): lowering delt from " << delt 
              << " to maximum: " << m_time_info->max_initial_delt
              << " (for initial timesteps)\n";
    delt = m_time_info->max_initial_delt;
  }

  // Check to see if delt exceeds the delt_max
  if( delt > m_time_info->delt_max ) {
    proc0cout << "WARNING (c): lowering delt from " << delt
	      << " to maximum: " << m_time_info->delt_max << '\n';
    delt = m_time_info->delt_max;
  }

  // Clamp delt to match the requested output and/or checkpoint times
  if( m_time_info->timestep_clamping && m_output ) {
    double orig_delt = delt;
    double nextOutput     = m_output->getNextOutputTime();
    double nextCheckpoint = m_output->getNextCheckpointTime();

    // Clamp to the output time
    if (nextOutput != 0 && time + delt > nextOutput) {
      delt = nextOutput - time;
    }

    // Clamp to the checkpoint time
    if (nextCheckpoint != 0 && time + delt > nextCheckpoint) {
      delt = nextCheckpoint - time;
    }

    // Report if delt was changed.
    if (delt != orig_delt) {
      proc0cout << "WARNING (d): lowering delt from " << orig_delt 
                << " to " << delt
                << " to line up with output/checkpoint time\n";
    }
  }
  
  // Clamp delt to the max end time,
  if (m_time_info->end_on_max_time && time + delt > m_time_info->maxTime) {
    delt = m_time_info->maxTime - time;
  }
}

//______________________________________________________________________
//
void
SimulationController::initWallTimes( void )
{
  m_num_samples = 0;

  // vars used to calculate standard deviation
  m_start_wall_time = Time::currentSeconds();
  m_total_wall_time      = 0.0;
  m_total_exec_wall_time = 0.0;
  m_exec_wall_time       = 0.0;
  m_insitu_wall_time     = 0.0;
  m_exp_moving_average   = 0.0;
}

//______________________________________________________________________
//
void
SimulationController::calcTotalWallTime ( void )
{
  // Calculate the total wall time.
  m_total_wall_time += Time::currentSeconds() - m_start_wall_time;

  // Reset the start time for relative clocking.
  m_start_wall_time = Time::currentSeconds();
}

//______________________________________________________________________
//
void
SimulationController::calcExecWallTime ( void )
{
  // Calculate the execution wall times and update total wall time.
  m_exec_wall_time = Time::currentSeconds() - m_start_wall_time;
  m_total_exec_wall_time += m_exec_wall_time;
  m_total_wall_time      += m_exec_wall_time;

  // Reset the start time for relative clocking.
  m_start_wall_time = Time::currentSeconds();

  // Calculate the exponential moving average for this time step.
  // Multiplier: (2 / (Time periods + 1) )
  // EMA: {Close - EMA(previous day)} x multiplier + EMA(previous day).

  // Ignore the first sample as that is for initialization.
  if (m_num_samples) {
    double mult = 2.0 / (std::min(m_num_samples, AVERAGE_WINDOW) + 1);
    m_exp_moving_average = mult * m_exec_wall_time + (1.0 - mult) * m_exp_moving_average;
  }
  else
    m_exp_moving_average = m_exec_wall_time;
}

//______________________________________________________________________
//
void
SimulationController::calcInSituWallTime ( void )
{
  // Calculate the in-situ wall times and update total wall time.
  m_insitu_wall_time = Time::currentSeconds() - m_start_wall_time;
  m_total_wall_time += m_insitu_wall_time;

  // Reset the start time for relative clocking.
  m_start_wall_time = Time::currentSeconds();
}

//______________________________________________________________________
//
double
SimulationController::getTotalWallTime( void )
{
  return m_total_wall_time;
}

//______________________________________________________________________
//
double
SimulationController::getTotalExecWallTime( void )
{
  return m_total_exec_wall_time;
}

//______________________________________________________________________
//
double
SimulationController::getExecWallTime( void )
{
  return m_exec_wall_time;
}

//______________________________________________________________________
//
double
SimulationController::getInSituWallTime( void )
{
  return m_insitu_wall_time;
}

//______________________________________________________________________
//
void
SimulationController::setStartSimTime ( double time )
{
  m_start_sim_time = time;
}

//______________________________________________________________________
//
bool
SimulationController::isLast( double time )
{
  return ( ( time >= m_time_info->maxTime ) ||
	   ( m_shared_state->getCurrentTopLevelTimeStep() >=
	       m_time_info->maxTimestep ) ||
	   ( m_time_info->max_wall_time != 0 &&
	       m_total_wall_time >= m_time_info->max_wall_time ) );
}

//______________________________________________________________________
//
void
SimulationController::printSimulationStats( int    timestep,
					                                  double next_delt,
					                                  double prev_delt,
					                                  double time,
					                                  bool   header )
{
  if( d_myworld->myrank() == 0 && header ) {
    dbg << std::endl;
    dbg << "Simulation and run time stats are reported at the end of each time step" << std::endl;
    dbg << "Wall Time == Total wall time, including execution, stats, and in-situ" << std::endl;
    dbg << "EMA == Execution wall time as an exponential moving average using a window of " << AVERAGE_WINDOW << " time steps" << std::endl;

    dbg.flush();
    std::cout.flush();
  }
  
  ReductionInfoMapper< SimulationState::RunTimeStat, double > &runTimeStats = m_shared_state->d_runTimeStats;

  // With the sum reduces, use double, since with memory it is possible that
  // it will overflow
  double        avg_memused = runTimeStats.getAverage( SimulationState::SCIMemoryUsed );
  unsigned long max_memused = runTimeStats.getMaximum( SimulationState::SCIMemoryUsed );
  int           max_memused_rank = runTimeStats.getRank( SimulationState::SCIMemoryUsed );

  double        avg_highwater = runTimeStats.getAverage( SimulationState::SCIMemoryHighwater );
  unsigned long max_highwater = runTimeStats.getMaximum( SimulationState::SCIMemoryHighwater );
  int           max_highwater_rank = runTimeStats.getRank( SimulationState::SCIMemoryHighwater );
    
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
  
  // Set the overhead percentage. Ignore the first sample as that is for initialization.
  if (m_num_samples) {
    m_overhead_values[m_overhead_index] = percent_overhead;

    double overhead = 0;
    double weight = 0;

    int t = std::min(m_num_samples, OVERHEAD_WINDOW);

    // Calculate total weight by incrementing through the overhead
    // sample array backwards and multiplying samples by the weights
    for (int i = 0; i < t; ++i) {
      unsigned int index = (m_overhead_index - i + OVERHEAD_WINDOW) % OVERHEAD_WINDOW;
      overhead += m_overhead_values[index] * m_overhead_weights[i];
      weight += m_overhead_weights[i];
    }

    // Increment the overhead index
    m_overhead_index = (m_overhead_index + 1) % OVERHEAD_WINDOW;

    m_shared_state->setOverheadAvg(overhead / weight);
  } 

  // Output timestep statistics...
  if (istats.active()) {
    for (unsigned int i = 0; i < runTimeStats.size(); i++) {
      SimulationState::RunTimeStat e = (SimulationState::RunTimeStat)i;

      if (runTimeStats[e] > 0) {
        istats << "rank: " << d_myworld->myrank() << " " << std::left << std::setw(19) << runTimeStats.getName(e) << " ["
        << runTimeStats.getUnits(e) << "]: " << runTimeStats[e] << "\n";
      }
    }
  } 

  if( d_myworld->myrank() == 0 ) {
    std::ostringstream message;
    message << std::left
	    << "Timestep "  << std::setw(6) << timestep
	    << "Time="      << std::setw(12) << time
//	    << "delT="      << std::setw(12) << prev_delt
	    << "Next delT=" << std::setw(12) << next_delt
      
	    << "Wall Time= "      << std::setw(10) << m_total_wall_time
	    // << "Total Exec Wall Time=" << std::setw(10) << d_totalExecWallTime
	    // << "Exec Wall Time="       << std::setw(10) << d_execWallTime <<
	    // << "In-situ Wall Time="    << setw(10) << d_inSituWallTime
	    << "EMA="                   << std::setw(10) << m_exp_moving_average;

    // Report on the memory used.
    if (avg_memused == max_memused && avg_highwater == max_highwater) {
      message << "Memory Use=" << std::setw(8)
	      << ProcessInfo::toHumanUnits((unsigned long) avg_memused);

      if(avg_highwater)
	message << "    Highwater Memory Use=" << std::setw(8)
		<< ProcessInfo::toHumanUnits((unsigned long) avg_highwater);
    }
    else {
      message << "Memory Used=" << std::setw(8)
	      << ProcessInfo::toHumanUnits((unsigned long) avg_memused)
	      << " (avg) " << std::setw(10)
	      << ProcessInfo::toHumanUnits(max_memused)
	      << " (max on rank:" << std::setw(6) << max_memused_rank << ")";

      if(avg_highwater)
	message << "    Highwater Memory Used=" << std::setw(8)
		<< ProcessInfo::toHumanUnits((unsigned long)avg_highwater)
		<< " (avg) " << std::setw(8)
		<< ProcessInfo::toHumanUnits(max_highwater)
		<< " (max on rank:" << std::setw(6) << max_highwater_rank << ")";
    }

    dbg << message.str() << "\n";
    dbg.flush();
    std::cout.flush();

    // Ignore the first sample as that is for initialization.
    if (stats.active() && m_num_samples) {
	    stats << "  " << std::left
		        << std::setw(21) << "Description"
		        << std::setw(15) << "Units"
		        << std::setw(15) << "Average"
		        << std::setw(15) << "Maximum"
		        << std::setw(13) << "Rank"
		        << std::setw(13) << "100*(1-ave/max) '% load imbalance'"
		        << "\n";

      for (unsigned int i = 0; i < runTimeStats.size(); ++i) {
        SimulationState::RunTimeStat e = (SimulationState::RunTimeStat)i;

        if (runTimeStats.getMaximum(e) > 0) {
	        stats << "  " << std::left
                << std::setw(21) << runTimeStats.getName(e)
		            << "[" << std::setw(10) << runTimeStats.getUnits(e) << "]"
		            << " : " << std::setw(12) << runTimeStats.getAverage(e)
		            << " : " << std::setw(12) << runTimeStats.getMaximum(e)
		            << " : " << std::setw(10) << runTimeStats.getRank(e)
		            << " : " << std::setw(10)
		            << 100.0 * (1.0 - (runTimeStats.getAverage(e) / runTimeStats.getMaximum(e)))
		            << "\n";
        }
      }
      
      // Report the overhead percentage.
      if (!std::isnan(m_shared_state->getOverheadAvg())) {
        stats << "  Percentage of time spent in overhead : "
              << m_shared_state->getOverheadAvg() * 100.0
              << "\n";
      }
    }

    // Ignore the first sample as that is for initialization.
    if (dbgTime.active() && m_num_samples ) {
      double realSecondsNow = m_exec_wall_time / prev_delt;
      double realSecondsAvg = m_total_exec_wall_time / (time-m_start_sim_time);

      dbgTime << "1 simulation second takes ";

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
  }

  ++m_num_samples;

} // end printSimulationStats()


//______________________________________________________________________
//
void
SimulationController::getMemoryStats( int timestep, bool create /* = false */ )
{
  int my_rank = d_myworld->myrank();
  unsigned long memUse, highwater, maxMemUse;
  m_scheduler->checkMemoryUse(memUse, highwater, maxMemUse);

  m_shared_state->d_runTimeStats[SimulationState::SCIMemoryUsed] = memUse;
  m_shared_state->d_runTimeStats[SimulationState::SCIMemoryMaxUsed] = maxMemUse;
  m_shared_state->d_runTimeStats[SimulationState::SCIMemoryHighwater] = highwater;

  if (ProcessInfo::isSupported(ProcessInfo::MEM_SIZE)) {
    m_shared_state->d_runTimeStats[SimulationState::MemoryUsed] = ProcessInfo::getMemoryUsed();
  }

  if (ProcessInfo::isSupported(ProcessInfo::MEM_RSS)) {
    m_shared_state->d_runTimeStats[SimulationState::MemoryResident] = ProcessInfo::getMemoryResident();
  }

  // Get memory stats for each proc if MALLOC_PERPROC is in the environment.
  if (getenv("MALLOC_PERPROC")) {
   std::ostream* mallocPerProcStream = nullptr;
    char* filenamePrefix = getenv("MALLOC_PERPROC");

    if (!filenamePrefix || strlen(filenamePrefix) == 0) {
      mallocPerProcStream = &dbg;
    }
    else {
      char filename[256];
      sprintf(filename, "%s.%d", filenamePrefix, my_rank);

      if ( create ) {
        mallocPerProcStream = scinew std::ofstream(filename, std::ios::out | std::ios::trunc);
      }
      else {
        mallocPerProcStream = scinew std::ofstream(filename, std::ios::out | std::ios::app);
      }

      if( !mallocPerProcStream ) {
        delete mallocPerProcStream;
        mallocPerProcStream = &dbg;
      }
    }

    *mallocPerProcStream << "Proc " << my_rank << "   ";
    *mallocPerProcStream << "Timestep " << timestep << "   ";

    if (ProcessInfo::isSupported(ProcessInfo::MEM_SIZE)) {
      *mallocPerProcStream << "Size " << ProcessInfo::getMemoryUsed() << "   ";
    }

    if (ProcessInfo::isSupported(ProcessInfo::MEM_RSS)) {
      *mallocPerProcStream << "RSS " << ProcessInfo::getMemoryResident() << "   ";
    }

    *mallocPerProcStream << "Sbrk " << (char*)sbrk(0) - m_scheduler->getStartAddr() << "   ";
    
#ifndef DISABLE_SCI_MALLOC
    *mallocPerProcStream << "Sci_Malloc_Memuse "    << memUse << "   ";
    *mallocPerProcStream << "Sci_Malloc_MaxMemuse " << maxMemUse << "   ";
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
SimulationController::getPAPIStats( void )
{
#ifdef USE_PAPI_COUNTERS
  int retp = PAPI_read(m_event_set, m_event_values);

  if (retp != PAPI_OK) {
    proc0cout << "Error: Cannot read PAPI event set!\n"
              << "       Error code = " << retp << " (" << m_papi_error_codes.find(retp)->second << ")\n";
    throw PapiInitializationError("PAPI read error. Unable to read hardware event set values.", __FILE__, __LINE__);
  }
  else {
    m_shared_state->d_runTimeStats[ SimulationState::TotalFlops ]  = (double) m_event_values[m_papi_events.find(PAPI_FP_OPS)->second.eventValueIndex];
    m_shared_state->d_runTimeStats[ SimulationState::TotalVFlops ] = (double) m_event_values[m_papi_events.find(PAPI_DP_OPS)->second.eventValueIndex];
    m_shared_state->d_runTimeStats[ SimulationState::L2Misses ]    = (double) m_event_values[m_papi_events.find(PAPI_L2_TCM)->second.eventValueIndex];
    m_shared_state->d_runTimeStats[ SimulationState::L3Misses ]    = (double) m_event_values[m_papi_events.find(PAPI_L3_TCM)->second.eventValueIndex];
  }

  // zero the values in the hardware counter event set array
  retp = PAPI_reset(m_event_set);

  if (retp != PAPI_OK) {
    proc0cout << "WARNNING: Cannot reset PAPI event set!\n"
              << "          Error code = " << retp << " ("
	            << m_papi_error_codes.find(retp)->second << ")\n";

    throw PapiInitializationError( "PAPI reset error on hardware event set. Unable to reset event set values.",  __FILE__, __LINE__ );
  }
#endif
}
  
} // namespace Uintah
