/*
 * The MIT License
 *
 * Copyright (c) 1997-2018 The University of Utah
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

#include <CCA/Components/SimulationController/SimulationController.h>

#include <Core/DataArchive/DataArchive.h>
#include <Core/Exceptions/InternalError.h>
#include <Core/Grid/Grid.h>
#include <Core/Grid/SimulationState.h>
#include <Core/Grid/SimulationTime.h>
#include <Core/Grid/Variables/VarTypes.h>
#include <Core/OS/Dir.h>
#include <Core/OS/ProcessInfo.h>
#include <Core/Parallel/Parallel.h>
#include <Core/Parallel/ProcessorGroup.h>
#include <Core/ProblemSpec/ProblemSpec.h>
#include <Core/Util/DOUT.hpp>

#ifdef USE_PAPI_COUNTERS
  #include <Core/Exceptions/PapiInitializationError.h>
#endif

#include <CCA/Ports/LoadBalancer.h>
#include <CCA/Ports/Output.h>
#include <CCA/Ports/ProblemSpecInterface.h>
#include <CCA/Ports/Regridder.h>
#include <CCA/Ports/Scheduler.h>
#include <CCA/Ports/ApplicationInterface.h>

#include <CCA/Components/Schedulers/MPIScheduler.h>

#include <sci_defs/visit_defs.h>


#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <sstream>
#include <sys/param.h>
#include <vector>

#include <pwd.h>

#define SECONDS_PER_MINUTE        60.0
#define SECONDS_PER_HOUR        3600.0
#define SECONDS_PER_DAY        86400.0
#define SECONDS_PER_WEEK      604800.0
#define SECONDS_PER_YEAR    31536000.0


namespace {

Uintah::Dout g_sim_stats(         "SimulationStats",            "SimulationController", "sim stats debug stream", true  );
Uintah::Dout g_sim_mem_stats(     "SimulationMemStats",         "SimulationController", "memory stats debug stream", true  );
Uintah::Dout g_sim_time_stats(    "SimulationTimeStats",        "SimulationController", "stats time debug stream", false );
Uintah::Dout g_sim_ctrl_dbg(      "SimulationController",       "SimulationController", "general debug stream", false );
Uintah::Dout g_comp_timings(      "ComponentTimings",           "SimulationController", "aggregated component timings", false );
Uintah::Dout g_indv_comp_timings( "IndividualComponentTimings", "SimulationController", "individual component timings", false );

}

namespace Uintah {

SimulationController::SimulationController( const ProcessorGroup * myworld
                                          ,       ProblemSpecP     prob_spec
                                          )
  : UintahParallelComponent( myworld )
  , m_ups( prob_spec )
{
  //initialize the overhead percentage
  for( int i = 0; i < OVERHEAD_WINDOW; ++i ) {
    double x = (double) i / (double) (OVERHEAD_WINDOW/2);
    m_overhead_values[i]  = 0;
    m_overhead_weights[i] = 8.0 - x*x*x;
  }
  
  m_grid_ps = m_ups->findBlock( "Grid" );

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
   */

  // PAPI_FP_OPS - floating point operations executed
  m_papi_events.insert(std::pair<int, PapiEvent>(PAPI_FP_OPS, PapiEvent("PAPI_FP_OPS", TotalFlops)));

  // PAPI_DP_OPS - floating point operations executed; optimized to count scaled double precision vector operations
  m_papi_events.insert(std::pair<int, PapiEvent>(PAPI_DP_OPS, PapiEvent("PAPI_DP_OPS", TotalVFlops)));

  // PAPI_L2_TCM - level 2 total cache misses
  m_papi_events.insert(std::pair<int, PapiEvent>(PAPI_L2_TCM, PapiEvent("PAPI_L2_TCM", L2Misses)));

  // PAPI_L3_TCM - level 3 total cache misses
  m_papi_events.insert(std::pair<int, PapiEvent>(PAPI_L3_TCM, PapiEvent("PAPI_L3_TCM", L3Misses)));

  // PAPI_TLB_TL - Total translation lookaside buffer misses
  m_papi_events.insert(std::pair<int, PapiEvent>(PAPI_TLB_TL, PapiEvent("PAPI_TLB_TL", TLBMisses)));

  m_papi_event_values = scinew long long[m_papi_events.size()];
  m_papi_event_set    = PAPI_NULL;
  int retval          = PAPI_NULL;

  // initialize the PAPI library
  retval = PAPI_library_init(PAPI_VER_CURRENT);
  if (retval != PAPI_VER_CURRENT) {
    proc0cout << "Error: Cannot initialize PAPI library!\n"
              << "       Error code = " << retval << " (" << PAPI_strerror(retval) << ")\n";
    throw PapiInitializationError("PAPI library initialization error occurred. Check that your PAPI library can be initialized correctly.", __FILE__, __LINE__);
  }

  // initialize thread support in the PAPI library
  retval = PAPI_thread_init(pthread_self);
  if (retval != PAPI_OK) {
    proc0cout << "Error: Cannot initialize PAPI thread support!\n"
              << "       Error code = " << retval << " (" << PAPI_strerror(retval) << ")\n";
    if (Parallel::getNumThreads() > 1) {
      throw PapiInitializationError("PAPI Pthread initialization error occurred. Check that your PAPI build supports Pthreads.", __FILE__, __LINE__);
    }
  }

  // query all PAPI events - find which are supported, flag those that are unsupported
  for (std::map<int, PapiEvent>::iterator iter = m_papi_events.begin(); iter != m_papi_events.end(); ++iter) {
    retval = PAPI_query_event(iter->first);
    if (retval != PAPI_OK) {
      proc0cout << "WARNNING: Cannot query PAPI event: " << iter->second.m_name << "!\n"
                << "          Error code = " << retval << " (" << PAPI_strerror(retval)<< ")\n"
                << "          No stats will be printed for " << iter->second.m_sim_stat_name << std::endl;
    }
    else {
      iter->second.m_is_supported = true;
    }
  }

  // create a new empty PAPI event set
  retval = PAPI_create_eventset(&m_papi_event_set);
  if (retval != PAPI_OK) {
    proc0cout << "Error: Cannot create PAPI event set!\n"
              << "       Error code = " << retval << " (" << PAPI_strerror(retval)<< ")\n";
    throw PapiInitializationError("PAPI event set creation error. Unable to create hardware counter event set.", __FILE__, __LINE__);
  }

  // Iterate through supported PAPI events, flag those that cannot be added.
  //   There are situations where an event may be queried but not added to an event set.
  int index = 0;
  for (std::map<int, PapiEvent>::iterator iter = m_papi_events.begin(); iter != m_papi_events.end(); ++iter) {
    if (iter->second.m_is_supported) {
      retval = PAPI_add_event(m_papi_event_set, iter->first);
      if (retval != PAPI_OK) { // this means the event queried OK but could not be added
        proc0cout << "WARNNING: Cannot add PAPI event: " << iter->second.m_name << "!\n"
                  << "          Error code = " << retval << " (" << PAPI_strerror(retval) << ")\n"
                  << "          No stats will be printed for " << iter->second.m_sim_stat_name << std::endl;
        iter->second.m_is_supported = false;
      }
      else {
        iter->second.m_event_value_idx = index;
        index++;
      }
    }
  }

  // Start counting PAPI events
  retval = PAPI_start(m_papi_event_set);
  if (retval != PAPI_OK) {
    proc0cout << "   ERROR: Cannot start PAPI event set!\n"
              << "          Error code = " << retval << " (" << PAPI_strerror(retval)
              << ")" << std::endl;

    // PAPI_ENOCMP means component index isn't set because no tracked
    // events supported on this platform, otherwise something
    // potentially unreasonable happened, either way we should not
    // continue
    std::string error_message = "PAPI event set start error.";
    std::string specific_message = error_message + ((retval == PAPI_ENOCMP) ? "  None of the PAPI events tracked by Uintah are available on this platform. " : "")
                                                 + "Please recompile without PAPI enabled.";
    throw PapiInitializationError(specific_message, __FILE__, __LINE__);
  }
#endif

  std::string timeStr("seconds");
  std::string bytesStr("MBytes");

  m_runtime_stats.insert( CompilationTime,           std::string("Compilation"),           timeStr, 0 );
  m_runtime_stats.insert( RegriddingTime,            std::string("Regridding"),            timeStr, 0 );
  m_runtime_stats.insert( RegriddingCompilationTime, std::string("RegriddingCompilation"), timeStr, 0 );
  m_runtime_stats.insert( RegriddingCopyDataTime,    std::string("RegriddingCopyData"),    timeStr, 0 );
  m_runtime_stats.insert( LoadBalancerTime,          std::string("LoadBalancer"),          timeStr, 0 );

  m_runtime_stats.insert( TaskExecTime,              std::string("TaskExec"),              timeStr, 0 );
  m_runtime_stats.insert( TaskLocalCommTime,         std::string("TaskLocalComm"),         timeStr, 0 );
  m_runtime_stats.insert( TaskWaitCommTime,          std::string("TaskWaitCommTime"),      timeStr, 0 );
  m_runtime_stats.insert( TaskReduceCommTime,        std::string("TaskReduceCommTime"),    timeStr, 0 );
  m_runtime_stats.insert( TaskWaitThreadTime,        std::string("TaskWaitThread"),        timeStr, 0 );

  m_runtime_stats.insert( XMLIOTime,                 std::string("XMLIO"),                 timeStr, 0 );
  m_runtime_stats.insert( OutputIOTime,              std::string("OutputIO"),              timeStr, 0 );
  m_runtime_stats.insert( ReductionIOTime,           std::string("ReductionIO"),           timeStr, 0 );
  m_runtime_stats.insert( CheckpointIOTime,          std::string("CheckpointIO"),          timeStr, 0 );
  m_runtime_stats.insert( CheckpointReductionIOTime, std::string("CheckpointReductionIO"), timeStr, 0 );
  m_runtime_stats.insert( TotalIOTime,               std::string("TotalIO"),               timeStr, 0 );

  m_runtime_stats.insert( OutputIORate,              std::string("OutputIORate"),     "MBytes/sec", 0 );
  m_runtime_stats.insert( ReductionIORate,           std::string("ReductionIORate"),  "MBytes/sec", 0 );
  m_runtime_stats.insert( CheckpointIORate,          std::string("CheckpointIORate"), "MBytes/sec", 0 );
  m_runtime_stats.insert( CheckpointReducIORate,     std::string("CheckpointReducIORate"), "MBytes/sec", 0 );

  m_runtime_stats.insert( NumTasks,                  std::string("NumberOfTasks"), "tasks", 0 );
  m_runtime_stats.insert( NumPatches,                std::string("NumberOfPatches"), "patches", 0 );
  m_runtime_stats.insert( NumCells,                  std::string("NumberOfCells"), "cells", 0 );
  m_runtime_stats.insert( NumParticles,              std::string("NumberOfParticles"), "paticles", 0 );
  
  m_runtime_stats.insert( SCIMemoryUsed,             std::string("SCIMemoryUsed"),         bytesStr, 0 );
  m_runtime_stats.insert( SCIMemoryMaxUsed,          std::string("SCIMemoryMaxUsed"),      bytesStr, 0 );
  m_runtime_stats.insert( SCIMemoryHighwater,        std::string("SCIMemoryHighwater"),    bytesStr, 0 );
  m_runtime_stats.insert( MemoryUsed,                std::string("MemoryUsed"),            bytesStr, 0 );
  m_runtime_stats.insert( MemoryResident,            std::string("MemoryResident"),        bytesStr, 0 );

#ifdef USE_PAPI_COUNTERS
  m_runtime_stats.insert( TotalFlops,  std::string("TotalFlops") , "FLOPS" , 0 );
  m_runtime_stats.insert( TotalVFlops, std::string("TotalVFlops"), "FLOPS" , 0 );
  m_runtime_stats.insert( L2Misses,    std::string("L2Misses")   , "misses", 0 );
  m_runtime_stats.insert( L3Misses,    std::string("L3Misses")   , "misses", 0 );
  m_runtime_stats.insert( TLBMisses,   std::string("TLBMisses")  , "misses", 0 );
#endif

#ifdef HAVE_VISIT
  m_visitSimData = scinew visit_simulation_data();
#endif

} // end SimulationController constructor

//______________________________________________________________________
//
SimulationController::~SimulationController()
{
  if (m_restart_archive) {
    delete m_restart_archive;
  }

#ifdef USE_PAPI_COUNTERS
  delete m_papi_event_values;
#endif

#ifdef HAVE_VISIT
  delete m_visitSimData;
#endif
}
  
//______________________________________________________________________
//
void
SimulationController::setPostProcessFlags()
{
  m_post_process_uda = true;
}

//______________________________________________________________________
//
void
SimulationController::getComponents( void )
{
  m_application = dynamic_cast<ApplicationInterface*>( getPort( "application" ) );

  if( !m_application ) {
    throw InternalError("dynamic_cast of 'm_app' failed!", __FILE__, __LINE__);
  }

  m_loadBalancer = dynamic_cast<LoadBalancer*>( getPort("load balancer") );

  if( !m_loadBalancer ) {
    throw InternalError("dynamic_cast of 'm_loadBalancer' failed!", __FILE__, __LINE__);
  }

  m_output = dynamic_cast<Output*>( getPort("output") );

  if( !m_output ) {
    throw InternalError("dynamic_cast of 'm_output' failed!", __FILE__, __LINE__);
  }

  m_regridder = dynamic_cast<Regridder*>( getPort("regridder") );

  if( m_application->isDynamicRegridding() && !m_regridder ) {
    throw InternalError("dynamic_cast of 'm_regridder' failed!", __FILE__, __LINE__);
  }

  m_scheduler = dynamic_cast<Scheduler*>( getPort("scheduler") );

  if( !m_scheduler ) {
    throw InternalError("dynamic_cast of 'm_scheduler' failed!", __FILE__, __LINE__);
  }
}

//______________________________________________________________________
//
void
SimulationController::releaseComponents( void )
{
  releasePort( "application" );
  releasePort( "load balancer" );
  releasePort( "output" );
  releasePort( "regridder" );
  releasePort( "scheduler" );
 
  m_application  = nullptr;
  m_loadBalancer = nullptr;
  m_output       = nullptr;
  m_regridder    = nullptr;
  m_scheduler    = nullptr;
}

//______________________________________________________________________
//

void
SimulationController::doRestart( const std::string & restartFromDir
                               ,       int           timeStep
                               ,       bool          fromScratch
                               ,       bool          removeOldDir
                               )
{
  m_restarting             = true;
  m_from_dir               = restartFromDir;
  m_restart_timestep       = timeStep;
  m_restart_from_scratch   = fromScratch;
  m_restart_remove_old_dir = removeOldDir;
}

//______________________________________________________________________
//
void
SimulationController::restartArchiveSetup( void )
{
  // Set up the restart archive now as it is need by the output.
  if( m_restarting ) {
    // Create the DataArchive here, and store it, as it is used a few
    // times. The grid needs to be read before the ProblemSetup, and
    // not all of the data can be read until after ProblemSetup, so
    // DataArchive operations are needed.

    Dir restartFromDir( m_from_dir );
    Dir checkpointRestartDir = restartFromDir.getSubdir( "checkpoints" );

    m_restart_archive = scinew DataArchive( checkpointRestartDir.getName(),
                                            d_myworld->myRank(),
                                            d_myworld->nRanks() );

    std::vector<int>    indices;
    std::vector<double> times;

    try {
      m_restart_archive->queryTimesteps( indices, times );
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
      m_restart_index    = 0;          // timestep == 0 means use the first timestep
      m_restart_timestep = indices[0]; // reset m_restart_timestep to what it really is
    }
    else if (m_restart_timestep == -1 && indices.size() > 0) {
      m_restart_index    = (unsigned int)(indices.size() - 1);
      m_restart_timestep = indices[indices.size() - 1]; // reset m_restart_timestep to what it really is
    }
    else {
      for (int index = 0; index < (int)indices.size(); index++)
        if (indices[index] == m_restart_timestep) {
          m_restart_index = index;
          break;
        }
    }
      
    if (m_restart_index == (int) indices.size()) {
      // timestep not found
      std::ostringstream message;
      message << "Time step " << m_restart_timestep << " not found";
      throw InternalError(message.str(), __FILE__, __LINE__);
    }

    // Do this call before calling DataArchive::restartInitialize,
    // because problemSetup() creates VarLabels the DataArchive needs.
    m_restart_ps =
      m_restart_archive->getTimestepDocForComponent( m_restart_index );
  }
}

//______________________________________________________________________
//
void
SimulationController::outputSetup( void )
{  
  // Set up the output - needs to be done before the application is setup.
  m_output->setRuntimeStats( &m_runtime_stats );

  m_output->problemSetup( m_ups, m_restart_ps,
                          m_application->getSimulationStateP() );
}

//______________________________________________________________________
//
void
SimulationController::gridSetup( void )
{
  // Set up the grid.
  if( m_restarting ) {
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
    m_current_gridP =
      m_restart_archive->queryGrid( m_restart_index, m_ups, false );
  }
  else /* if( !m_restarting ) */ {
    m_current_gridP = scinew Grid();

    // The call to preGridProblemSetup() by the simulation interface
    // allows for a call to grid->setExtraCells() to be made before
    // the grid problemSetup() so that if there are no extra cells
    // defined in the ups file, all levels will use the grid extra
    // cell value.

    // For instance, Wasatch does not allow users to specify extra
    // cells through the input file. Instead, Wasatch wants to specify
    // it internally. This call gives the option to do just that
    // though it does not follow normal paradigm of calling
    // problemSetup immediately after a component or other object is
    // created.
    m_application->preGridProblemSetup( m_ups, m_current_gridP );

    // Now that the simulation interface has made its changes do the
    // normal grid problemSetup()
    m_current_gridP->problemSetup( m_ups, d_myworld, m_application->isAMR() );
  }

  if( m_current_gridP->numLevels() == 0 ) {
    throw InternalError("No problem (no levels in grid) specified.", __FILE__, __LINE__);
  }

  // Print out meta data
  if ( d_myworld->myRank() == 0 ) {
    m_current_gridP->printStatistics();
//    DOUT(true, "Restart grid\n" << *m_current_gridP.get_rep());
  }
}

//______________________________________________________________________
//
void
SimulationController::regridderSetup( void )
{
  // Set up the regridder.

  // Do this step before fully setting up the application interface so that the
  // Switcher (being an application) can reset the state of the regridder.
  if( m_regridder ) {
    m_regridder->problemSetup( m_ups, m_current_gridP, m_application->getSimulationStateP() );
  }
}

//______________________________________________________________________
//
void
SimulationController::schedulerSetup( void )
{
  // Now that the grid is completely set up, set up the scheduler.
  m_scheduler->setRuntimeStats( &m_runtime_stats );

  m_scheduler->problemSetup( m_ups, m_application->getSimulationStateP() );

  // Additional set up calls.
  m_scheduler->setInitTimestep( true );
  m_scheduler->setRestartInitTimestep( m_restarting );
  m_scheduler->initialize( 1, 1 );
  m_scheduler->clearTaskMonitoring();

  m_scheduler->advanceDataWarehouse( m_current_gridP, true );
}

//______________________________________________________________________
//
void
SimulationController::loadBalancerSetup( void )
{
  // Set up the load balancer.
  m_loadBalancer->setRuntimeStats( &m_runtime_stats );

  //  Set the dimensionality of the problem.
  IntVector low, high, size;
  m_current_gridP->getLevel(0)->findCellIndexRange(low, high);

  size = high - low - m_current_gridP->getLevel(0)->getExtraCells()*IntVector(2,2,2);
  
  m_loadBalancer->setDimensionality(size[0] > 1, size[1] > 1, size[2] > 1);
 
  // In addition, do this step after regridding setup as the minimum
  // patch size that the regridder will create will be known.
  m_loadBalancer->problemSetup( m_ups, m_current_gridP, m_application->getSimulationStateP() );
}

//______________________________________________________________________
//
void
SimulationController::applicationSetup( void )
{
  // Pass the m_restart_ps to the component's problemSetup.  For
  // restarting, pull the <MaterialProperties> from the m_restart_ps.
  // If the properties are not available, then pull the properties
  // from the m_ups instead.  This step needs to be done before
  // DataArchive::restartInitialize.
  m_application->problemSetup(m_ups, m_restart_ps, m_current_gridP);

  // Finalize the shared state/materials
  m_application->getSimulationStateP()->finalizeMaterials();
}


//______________________________________________________________________
//
void
SimulationController::timeStateSetup()
{
  // Restarting so initialize time state using the archive data.
  if( m_restarting ) {

    DOUT(g_sim_ctrl_dbg, "Restarting... loading data");

    double simTimeStart;
    
    m_restart_archive->restartInitialize( m_restart_index,
                                          m_current_gridP,
                                          m_scheduler->get_dw(1),
                                          m_loadBalancer,
                                          &simTimeStart );

    // Set the time step to the restart time step which is immediately
    // written to the DW.
    m_application->setTimeStep( m_restart_timestep );

    // Set the simulation time to the restart simulation time which is
    // immediately written to the DW.
    m_application->setSimTimeStart( simTimeStart );

    // Set the next delta T which is immediately written to the DW.

    // Note the old delta T is a default and normally would not be
    // used.
    m_application->setNextDelT( m_restart_archive->getOldDelt( m_restart_index ) );

    // Tell the scheduler the generation of the re-started simulation.
    // (Add +1 because the scheduler will be starting on the next
    // time step.)
    m_scheduler->setGeneration( m_restart_timestep + 1 );

    // This delete is an enigma. If it is called then memory is not
    // leaked, but sometimes if is called, then everything segfaults.
    // delete m_restart_archive;
  }
  else
  {
    // Set the time step to 0 which is immediately written to the DW.
    m_application->setTimeStep( 0 );

    // Set the simulation time to 0 which is immediately written to the DW.
    m_application->setSimTimeStart( 0 );
  }
}

//______________________________________________________________________
//
void
SimulationController::finalSetup()
{
  // This step is done after the call to m_application->problemSetup to get
  // the defaults set by the simulation interface into the input.xml,
  // which the output writes along with index.xml
  m_output->initializeOutput(m_ups, m_current_gridP );

  // This step is done after the output is initialized so that global
  // reduction output vars are copied to the new uda. Further, it must
  // be called after timeStateSetup() is call so that checkpoints are
  // copied to the new uda as well.
  if( m_restarting ) {
    Dir dir( m_from_dir );
    m_output->restartSetup( dir, 0, m_restart_timestep,
                            m_application->getSimTimeStart(),
                            m_restart_from_scratch, m_restart_remove_old_dir );
  }

  // Miscellaneous initializations.
  ProblemSpecP amr_ps = m_ups->findBlock("AMR");
  if( amr_ps ) {
    amr_ps->get( "doMultiTaskgraphing", m_do_multi_taskgraphing );
  }
}

//______________________________________________________________________
//
void SimulationController::ResetStats( void )
{
  m_runtime_stats.reset( 0 );
  m_application->resetApplicationStats( 0 );
}

//______________________________________________________________________
//
void
SimulationController::ScheduleReportStats( bool header )
{
  Task* task = scinew Task("SimulationController::ReportStats",
                           this, &SimulationController::ReportStats, header);
  
  task->setType(Task::OncePerProc);

  // Require delta T so that the task gets scheduled
  // correctly. Otherwise the scheduler/taskgraph will toss an error :
  // Caught std exception: map::at: key not found
  task->requires(Task::NewDW, m_application->getDelTLabel() );

  m_scheduler->addTask(task,
                       m_loadBalancer->getPerProcessorPatchSet(m_current_gridP),
                       m_application->getSimulationStateP()->allMaterials() );

  // std::cerr << "*************" << __FUNCTION__ << "  " << __LINE__ << "  " << header << std::endl;
}

void
SimulationController::ReportStats(const ProcessorGroup*,
                                  const PatchSubset*,
                                  const MaterialSubset*,
                                        DataWarehouse*,
                                        DataWarehouse*,
                                        bool header )
{
  // Get and reduce the performance runtime stats
  getMemoryStats();
  getPAPIStats();

  // Reductions are only need if these are true.
  if( (m_regridder && m_regridder->useDynamicDilation()) ||
      g_sim_mem_stats || g_comp_timings || g_indv_comp_timings ) {

    m_runtime_stats.reduce( m_regridder && m_regridder->useDynamicDilation(), d_myworld );
  
    m_application->reduceApplicationStats( m_regridder && m_regridder->useDynamicDilation(), d_myworld );

    // Reduce the MPI runtime stats.
    MPIScheduler * mpiScheduler = dynamic_cast<MPIScheduler*>( m_scheduler.get_rep() );

    if( mpiScheduler ) {
      mpiScheduler->mpi_info_.reduce( m_regridder && m_regridder->useDynamicDilation(), d_myworld );
    }
  }
  
  // Update the moving average and get the wall time for this time step.
  Timers::nanoseconds timeStep = m_wall_timers.updateExpMovingAverage();

  // Print the stats for this time step
  if( d_myworld->myRank() == 0 && g_sim_stats ) {
    std::ostringstream message;

    if( header )
      message << std::endl
              << "Simulation and run time stats are reported "
              << "at the end of each time step" << std::endl
              << "EMA == Wall time as an exponential moving average "
              << "using a window of the last " << m_wall_timers.getWindow()
              << " time steps" << std::endl;

    message << std::left
            << "Timestep "   << std::setw(8)  << m_application->getTimeStep()
            << "Time="       << std::setw(12) << m_application->getSimTime()
            // << "delT="       << std::setw(12) << m_application->getDelT()
            << "Next delT="  << std::setw(12) << m_application->getNextDelT()

            << "Wall Time=" << std::setw(10) << m_wall_timers.GetWallTime()
            // << "All Time steps= " << std::setw(12) << m_wall_timers.TimeStep().seconds()
            // << "Current Time Step= " << std::setw(12) << timeStep.seconds()
            << "EMA="        << std::setw(12) << m_wall_timers.ExpMovingAverage().seconds()
           // << "In-situ Time = " << std::setw(12) << walltimers.InSitu().seconds()
      ;

    // Report on the memory used.
    if( g_sim_mem_stats ) {
      // With the sum reduces, use double, since with memory it is possible that
      // it will overflow
      double        avg_memused      = m_runtime_stats.getAverage( SCIMemoryUsed );
      unsigned long max_memused      = m_runtime_stats.getMaximum( SCIMemoryUsed );
      int           max_memused_rank = m_runtime_stats.getRank(    SCIMemoryUsed );
      
      double        avg_highwater      = m_runtime_stats.getAverage( SCIMemoryHighwater );
      unsigned long max_highwater      = m_runtime_stats.getMaximum( SCIMemoryHighwater );
      int           max_highwater_rank = m_runtime_stats.getRank(    SCIMemoryHighwater );
      
      if (avg_memused == max_memused && avg_highwater == max_highwater) {
        message << "Memory Use=" << std::setw(8)
                << ProcessInfo::toHumanUnits((unsigned long) avg_memused);

        if(avg_highwater)
          message << "    Highwater Memory Use=" << std::setw(8)
                  << ProcessInfo::toHumanUnits((unsigned long) avg_highwater);
      }
      else {
        message << "Memory Used=" << std::setw(10)
                << ProcessInfo::toHumanUnits((unsigned long) avg_memused)
                << " (avg) " << std::setw(10)
                << ProcessInfo::toHumanUnits(max_memused)
                << " (max on rank: " << std::setw(6) << max_memused_rank << ")";

        if (avg_highwater)
          message << "    Highwater Memory Used=" << std::setw(10)
                  << ProcessInfo::toHumanUnits((unsigned long)avg_highwater)
                  << " (avg) " << std::setw(10)
                  << ProcessInfo::toHumanUnits(max_highwater)
                  << " (max on rank: " << std::setw(6) << max_highwater_rank << ")";
      }
    }
    else {
      double  memused   = m_runtime_stats[SCIMemoryUsed];
      double  highwater = m_runtime_stats[SCIMemoryHighwater];
      
      message << "Memory Use=" << std::setw(8)
              << ProcessInfo::toHumanUnits((unsigned long) memused );

      if(highwater)
        message << "    Highwater Memory Use=" << std::setw(8)
                << ProcessInfo::toHumanUnits((unsigned long) highwater);

      message << " (on rank 0 only)";
    }

    DOUT(true, message.str());
    std::cout.flush();
  }
  
  // Report on the simulation time used.
  if( d_myworld->myRank() == 0 && g_sim_time_stats && m_num_samples )
  {
    // Ignore the first sample as that is for initialization.
    std::ostringstream message;

    double realSecondsNow = timeStep.seconds() / m_application->getDelT();
    double realSecondsAvg = m_wall_timers.TimeStep().seconds() /
      (m_application->getSimTime()-m_application->getSimTimeStart());
    
    message << "1 simulation second takes ";
    
    message << std::left << std::showpoint << std::setprecision(3) << std::setw(4);
    
    if (realSecondsNow < SECONDS_PER_MINUTE) {
      message << realSecondsNow << " seconds (now), ";
    }
    else if (realSecondsNow < SECONDS_PER_HOUR) {
      message << realSecondsNow / SECONDS_PER_MINUTE << " minutes (now), ";
    }
    else if (realSecondsNow < SECONDS_PER_DAY) {
      message << realSecondsNow / SECONDS_PER_HOUR << " hours (now), ";
    }
    else if (realSecondsNow < SECONDS_PER_WEEK) {
      message << realSecondsNow / SECONDS_PER_DAY << " days (now), ";
    }
    else if (realSecondsNow < SECONDS_PER_YEAR) {
      message << realSecondsNow / SECONDS_PER_WEEK << " weeks (now), ";
    }
    else {
      message << realSecondsNow / SECONDS_PER_YEAR << " years (now), ";
    }

    message << std::setw(4);

    if (realSecondsAvg < SECONDS_PER_MINUTE) {
      message << realSecondsAvg << " seconds (avg) ";
    }
    else if (realSecondsAvg < SECONDS_PER_HOUR) {
      message << realSecondsAvg / SECONDS_PER_MINUTE << " minutes (avg) ";
    }
    else if (realSecondsAvg < SECONDS_PER_DAY) {
      message << realSecondsAvg / SECONDS_PER_HOUR << " hours (avg) ";
    }
    else if (realSecondsAvg < SECONDS_PER_WEEK) {
      message << realSecondsAvg / SECONDS_PER_DAY << " days (avg) ";
    }
    else if (realSecondsAvg < SECONDS_PER_YEAR) {
      message << realSecondsAvg / SECONDS_PER_WEEK << " weeks (avg) ";
    }
    else {
      message << realSecondsAvg / SECONDS_PER_YEAR << " years (avg) ";
    }

    DOUT(true, message.str());
    std::cout.flush();
  }

  // Sum up the average time for overhead related components. These
  // same values are used in SimulationState::getOverheadTime.
  double overhead_time =
    (m_runtime_stats.getAverage(CompilationTime)           +
     m_runtime_stats.getAverage(RegriddingTime)            +
     m_runtime_stats.getAverage(RegriddingCompilationTime) +
     m_runtime_stats.getAverage(RegriddingCopyDataTime)    +
     m_runtime_stats.getAverage(LoadBalancerTime));

  // Sum up the average times for simulation components. These
  // same values are used in SimulationState::getTotalTime.
  double total_time =
    (overhead_time +
     m_runtime_stats.getAverage(TaskExecTime)       +
     m_runtime_stats.getAverage(TaskLocalCommTime)  +
     m_runtime_stats.getAverage(TaskWaitCommTime)   +
     m_runtime_stats.getAverage(TaskReduceCommTime) +
     m_runtime_stats.getAverage(TaskWaitThreadTime));
  
    // Calculate percentage of time spent in overhead.
  double percent_overhead = overhead_time / total_time;
  
  double overheadAverage = 0;

  // Set the overhead percentage. Ignore the first sample as that is
  // for initialization.
  if (m_num_samples) {
    m_overhead_values[m_overhead_index] = percent_overhead;

    double overhead = 0;
    double weight = 0;

    int sample_size = std::min(m_num_samples, OVERHEAD_WINDOW);

    // Calculate total weight by incrementing through the overhead
    // sample array backwards and multiplying samples by the weights
    for (int i = 0; i < sample_size; ++i) {
      unsigned int index = (m_overhead_index - i + OVERHEAD_WINDOW) % OVERHEAD_WINDOW;
      overhead += m_overhead_values[index] * m_overhead_weights[i];
      weight   += m_overhead_weights[i];
    }

    // Increment the overhead index
    m_overhead_index = (m_overhead_index + 1) % OVERHEAD_WINDOW;

    overheadAverage = overhead / weight;

    if( m_regridder ) {
      m_regridder->setOverheadAverage(overheadAverage);
    }
  }

  // Infrastructure per proc runtime performance stats
  if (g_indv_comp_timings) {

    if (m_runtime_stats.size()) {
      std::ostringstream message;
      message << "Per proc runtime performance stats" << std::endl;

      for (unsigned int i = 0; i < m_runtime_stats.size(); ++i) {
        if (m_runtime_stats[i] > 0) {
          message << "  " << std::left
                  << "rank: " << std::setw(5) << d_myworld->myRank() << " "
                  << std::left << std::setw(19) << m_runtime_stats.getName(i) << " ["
                  << m_runtime_stats.getUnits(i) << "]: " << m_runtime_stats[i] << "\n";
        }
      }

      DOUT(true, message.str());
    }

    // Application per proc runtime performance stats    
    if (m_application->getApplicationStats().size()) {
      std::ostringstream message;
      message << "Application per proc performance stats" << std::endl;

      for (unsigned int i = 0; i < m_application->getApplicationStats().size(); ++i) {
        if (m_application->getApplicationStats()[i] > 0) {
          message << "  " << std::left
                  << "rank: " << std::setw(5) << d_myworld->myRank() << " "
                  << std::left << std::setw(19) << m_application->getApplicationStats().getName(i) << " ["
                  << m_application->getApplicationStats().getUnits(i) << "]: " << m_application->getApplicationStats()[i] << "\n";
        }
      }
      
      DOUT(true, message.str());
    }
  } 

  // Average proc runtime performance stats
  if( d_myworld->myRank() == 0 && g_comp_timings && m_num_samples)
  {
    // Ignore the first sample as that is for initialization.
    std::ostringstream message;

    if (m_runtime_stats.size()) {
      message << "Runtime performance stats" << std::endl;
      
      message << "  " << std::left
              << std::setw(21) << "Description"
              << std::setw(15) << "Units"
              << std::setw(15) << "Average"
              << std::setw(15) << "Maximum"
              << std::setw(13) << "Rank"
              << std::setw(13) << "100*(1-ave/max) '% load imbalance'"
              << std::endl;
        
      for (unsigned int i=0; i<m_runtime_stats.size(); ++i) {
        if( m_runtime_stats.getMaximum(i) != 0.0 )
          message << "  " << std::left
                  << std::setw(21) << m_runtime_stats.getName(i)
                  << "[" << std::setw(10) << m_runtime_stats.getUnits(i) << "]"
                  << " : " << std::setw(12) << m_runtime_stats.getAverage(i)
                  << " : " << std::setw(12) << m_runtime_stats.getMaximum(i)
                  << " : " << std::setw(10) << m_runtime_stats.getRank(i)
                  << " : " << std::setw(10)
                  << (m_runtime_stats.getMaximum(i) != 0.0 ? (100.0 * (1.0 - (m_runtime_stats.getAverage(i) / m_runtime_stats.getMaximum(i)))) : 0)
                  << std::endl;
      }
    }
    
    // Report the overhead percentage.
    if( !std::isnan(overheadAverage) ) {
      message << "  Percentage of time spent in overhead : " << overheadAverage * 100.0 << "\n";
    }

    message << std::endl;
    
    // Report the application stats.
    if( m_application->getApplicationStats().size() ) {
      message << "Application performance stats" << std::endl;
      
      message << "  " << std::left
              << std::setw(21) << "Description"
              << std::setw(15) << "Units"
              << std::setw(15) << "Average"
              << std::setw(15) << "Maximum"
              << std::setw(13) << "Rank"
              << std::setw(13) << "100*(1-ave/max) '% load imbalance'"
              << std::endl;

      for (unsigned int i=0; i<m_application->getApplicationStats().size(); ++i) {
        if( m_application->getApplicationStats().getMaximum(i) != 0.0 )
          message << "  " << std::left
                  << std::setw(21) << m_application->getApplicationStats().getName(i)
                  << "["   << std::setw(10) << m_application->getApplicationStats().getUnits(i) << "]"
                  << " : " << std::setw(12) << m_application->getApplicationStats().getAverage(i)
                  << " : " << std::setw(12) << m_application->getApplicationStats().getMaximum(i)
                  << " : " << std::setw(10) << m_application->getApplicationStats().getRank(i)
                  << " : " << std::setw(10)
                  << (m_application->getApplicationStats().getMaximum(i) != 0.0 ? (100.0 * (1.0 - (m_application->getApplicationStats().getAverage(i) / m_application->getApplicationStats().getMaximum(i)))) : 0)
                  << std::endl;
      }

      message << std::endl;
    }

    DOUT(true, message.str());
  }
  
  ++m_num_samples;
  
} // end printSimulationStats()

//______________________________________________________________________
//

void
SimulationController::getMemoryStats( bool create /* = false */ )
{
  unsigned long memUsed;
  unsigned long highwater;
  unsigned long maxMemUsed;

  m_scheduler->checkMemoryUse(memUsed, highwater, maxMemUsed);

  m_runtime_stats[SCIMemoryUsed]      = memUsed;
  m_runtime_stats[SCIMemoryMaxUsed]   = maxMemUsed;
  m_runtime_stats[SCIMemoryHighwater] = highwater;

  if (ProcessInfo::isSupported(ProcessInfo::MEM_SIZE)) {
    m_runtime_stats[MemoryUsed] = ProcessInfo::getMemoryUsed();
  }

  if (ProcessInfo::isSupported(ProcessInfo::MEM_RSS)) {
    m_runtime_stats[MemoryResident] = ProcessInfo::getMemoryResident();
  }

  // Get memory stats for each proc if MALLOC_PERPROC is in the environment.
  if (getenv("MALLOC_PERPROC")) {
    std::ostream* mallocPerProcStream = nullptr;
    char* filenamePrefix = getenv("MALLOC_PERPROC");

    // provide a default filename if none provided
    if (!filenamePrefix || strlen(filenamePrefix) == 0) {
      filenamePrefix = (char*)"malloc.log";
    }

    char filename[256];
    sprintf(filename, "%s.%d", filenamePrefix, d_myworld->myRank());

    if (create) {
      mallocPerProcStream = scinew std::ofstream(filename, std::ios::out | std::ios::trunc);
    }
    else {
      mallocPerProcStream = scinew std::ofstream(filename, std::ios::out | std::ios::app);
    }

    *mallocPerProcStream << "Proc "     << d_myworld->myRank()  << "   ";
    *mallocPerProcStream << "TimeStep " << m_application->getTimeStep() << "   ";

    if (ProcessInfo::isSupported(ProcessInfo::MEM_SIZE)) {
      *mallocPerProcStream << "Size " << ProcessInfo::getMemoryUsed() << "   ";
    }

    if (ProcessInfo::isSupported(ProcessInfo::MEM_RSS)) {
      *mallocPerProcStream << "RSS " << ProcessInfo::getMemoryResident() << "   ";
    }

    *mallocPerProcStream << "Sbrk " << (char*)sbrk(0) - m_scheduler->getStartAddr() << "   ";

#ifndef DISABLE_SCI_MALLOC
    *mallocPerProcStream << "Sci_Malloc_MemUsed " << memUsed << "   ";
    *mallocPerProcStream << "Sci_Malloc_MaxMemUsed " << maxMemUsed << "   ";
    *mallocPerProcStream << "Sci_Malloc_Highwater " << highwater;
#endif

    *mallocPerProcStream << std::endl;

    if (mallocPerProcStream) {
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
  int retval = PAPI_read(m_papi_event_set, m_papi_event_values);

  if (retval != PAPI_OK) {
    proc0cout << "   Error: Cannot read PAPI event set!\n"
              << "       Error code = " << retval << " (" << PAPI_strerror(retval) << ")\n";
    throw PapiInitializationError("PAPI read error. Unable to read hardware event set values.", __FILE__, __LINE__);
  }
  else {
    // query all PAPI events - find which are supported, flag those that are unsupported
    for (std::map<int, PapiEvent>::iterator iter = m_papi_events.begin(); iter != m_papi_events.end(); ++iter) {
      if (iter->second.m_is_supported) {
        m_runtime_stats[ iter->second.m_sim_stat_name ] =
                static_cast<double>(m_papi_event_values[m_papi_events.find(iter->first)->second.m_event_value_idx]);
      }
    }
  }

  // zero the values in the hardware counter event set array
  retval = PAPI_reset(m_papi_event_set);
  if (retval != PAPI_OK) {
    proc0cout << "WARNNING: Cannot reset PAPI event set!\n"
              << "          Error code = " << retval << " ("
              << PAPI_strerror(retval) << ")\n";
    throw PapiInitializationError( "PAPI reset error on hardware event set, unable to reset event set values.", __FILE__, __LINE__ );
  }
#endif
}
  
//______________________________________________________________________
//
void
SimulationController::ScheduleCheckInSitu( bool first )
{
#ifdef HAVE_VISIT
  if( getVisIt() ) {

    Task* task = scinew Task("SimulationController::CheckInSitu",
                             this, &SimulationController::CheckInSitu, first);
    
    task->setType(Task::OncePerProc);

    // Require delta T so that the task gets scheduled
    // correctly. Otherwise the scheduler/taskgraph will toss an error
    // : Caught std exception: map::at: key not found
    task->requires(Task::NewDW, m_application->getDelTLabel() );

    m_scheduler->addTask(task,
                         m_loadBalancer->getPerProcessorPatchSet(m_current_gridP),
                         m_application->getSimulationStateP()->allMaterials() );

    // std::cerr << "*************" << __FUNCTION__ << "  " << __LINE__ << "  " << first << std::endl;
  }
#endif      
}

//______________________________________________________________________
//
void
SimulationController::CheckInSitu(const ProcessorGroup*,
                                  const PatchSubset*,
                                  const MaterialSubset*,
                                        DataWarehouse*,
                                        DataWarehouse*,
                                        bool first)
{
#ifdef HAVE_VISIT
  // If VisIt has been included into the build, check the lib sim
  // state to see if there is a connection and if so check to see if
  // anything needs to be done.
  if( getVisIt() ) {
    // Note this timer is used as a laptimer so the stop must come
    // after the lap time is taken so not to affect it.
    m_wall_timers.TimeStep.stop();

    m_wall_timers.InSitu.start();

    // Update all of the simulation grid and time dependent variables.
    visit_UpdateSimData( m_visitSimData, m_current_gridP,
                         m_application->getSimTime(),
                         m_application->getTimeStep(),
                         m_application->getDelT(),
                         m_application->getNextDelT(),
                         first,
                         m_application->isLastTimeStep(m_wall_timers.GetWallTime()) );

    // Check the state - if the return value is true the user issued
    // a termination.
    if( visit_CheckState( m_visitSimData ) ) {
      m_application->mayEndSimulation(true);
    }

    // std::cerr << "*************" << __FUNCTION__ << "  " << __LINE__ << "  "
    //        << first << "  "
    //        << m_application->getSimTime() << "  "
    //        << m_application->getTimeStep() << "  "
    //        << std::endl;

    // This function is no longer used as last is now used in the
    // UpdateSimData which in turn will flip the state.
      
    // Check to see if at the last iteration. If so stop so the
    // user can have once last chance see the data.
    // if( m_visitSimData->stopAtLastTimeStep && last )
    // visit_EndLibSim( m_visitSimData );

    // Add the modified variable information into index.xml file.
    m_output->writeto_xml_files(m_visitSimData->modifiedVars);

    m_wall_timers.InSitu.stop();

    // Note this timer is used as a laptimer.
    m_wall_timers.TimeStep.start();
  }
#endif
}

} // namespace Uintah
