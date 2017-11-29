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
#include <CCA/Ports/ApplicationInterface.h>

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
                                                  ProblemSpecP     prob_spec )
  : UintahParallelComponent( myworld ), d_ups( prob_spec )
{
  //initialize the overhead percentage
  overheadIndex = 0;
  for( int i = 0; i < OVERHEAD_WINDOW; ++i ) {
    double x = (double) i / (double) (OVERHEAD_WINDOW/2);
    overheadValues[i]  = 0;
    overheadWeights[i] = 8.0 - x*x*x;
  }

  d_nSamples               = 0;
  d_recompileTaskGraph     = false;
  
  d_restarting             = false;
  d_postProcessUda         = false;
  d_do_multi_taskgraphing  = false;
  d_restart_archive        = nullptr;
  d_app                    = 0;

  d_grid_ps                = d_ups->findBlock( "Grid" );

  d_restart_ps             = 0;
  
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
  m_papi_events.insert(pair<int, PapiEvent>(PAPI_FP_OPS, PapiEvent("PAPI_FP_OPS", SimulationState::TotalFlops)));

  // PAPI_DP_OPS - floating point operations executed; optimized to count scaled double precision vector operations
  m_papi_events.insert(pair<int, PapiEvent>(PAPI_DP_OPS, PapiEvent("PAPI_DP_OPS", SimulationState::TotalVFlops)));

  // PAPI_L2_TCM - level 2 total cache misses
  m_papi_events.insert(pair<int, PapiEvent>(PAPI_L2_TCM, PapiEvent("PAPI_L2_TCM", SimulationState::L2Misses)));

  // PAPI_L3_TCM - level 3 total cache misses
  m_papi_events.insert(pair<int, PapiEvent>(PAPI_L3_TCM, PapiEvent("PAPI_L3_TCM", SimulationState::L3Misses)));

  // PAPI_TLB_TL - Total translation lookaside buffer misses
  m_papi_events.insert(pair<int, PapiEvent>(PAPI_TLB_TL, PapiEvent("PAPI_TLB_TL", SimulationState::TLBMisses)));

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

    // PAPI_ENOCMP means component index isn't set because no tracked events supported on this platform,
    // otherwise something potentially unreasonable happened, either way we should not continue
    std::string error_message = "PAPI event set start error.";
    std::string specific_message = error_message + ((retval == PAPI_ENOCMP) ? "  None of the PAPI events tracked by Uintah are available on this platform. " : "")
                                                 + "Please recompile without PAPI enabled.";
    throw PapiInitializationError(specific_message, __FILE__, __LINE__);
  }
#endif

  std::string timeStr("seconds");
  std::string bytesStr("MBytes");
    
  d_runTimeStats.insert( CompilationTime,           std::string("Compilation"),           timeStr, 0 );
  d_runTimeStats.insert( RegriddingTime,            std::string("Regridding"),            timeStr, 0 );
  d_runTimeStats.insert( RegriddingCompilationTime, std::string("RegriddingCompilation"), timeStr, 0 );
  d_runTimeStats.insert( RegriddingCopyDataTime,    std::string("RegriddingCopyData"),    timeStr, 0 );
  d_runTimeStats.insert( LoadBalancerTime,          std::string("LoadBalancer"),          timeStr, 0 );

  d_runTimeStats.insert( TaskExecTime,       std::string("TaskExec"),           timeStr, 0 );
  d_runTimeStats.insert( TaskLocalCommTime,  std::string("TaskLocalComm"),      timeStr, 0 );
  d_runTimeStats.insert( TaskWaitCommTime,   std::string("TaskWaitCommTime"),   timeStr, 0 );
  d_runTimeStats.insert( TaskReduceCommTime, std::string("TaskReduceCommTime"), timeStr, 0 );
  d_runTimeStats.insert( TaskWaitThreadTime, std::string("TaskWaitThread"),     timeStr, 0 );

  d_runTimeStats.insert( XMLIOTime,          std::string("XMLIO"),            timeStr, 0 );
  d_runTimeStats.insert( OutputIOTime,       std::string("OutputIO"),         timeStr, 0 );
  d_runTimeStats.insert( ReductionIOTime,    std::string("ReductionIO"),      timeStr, 0 );
  d_runTimeStats.insert( CheckpointIOTime,   std::string("CheckpointIO"),     timeStr, 0 );
  d_runTimeStats.insert( CheckpointReductionIOTime, std::string("CheckpointReductionIO"),     timeStr, 0 );
  d_runTimeStats.insert( TotalIOTime,        std::string("TotalIO"),          timeStr, 0 );

  d_runTimeStats.insert( OutputIORate,       std::string("OutputIORate"),     "MBytes/sec", 0 );
  d_runTimeStats.insert( ReductionIORate,    std::string("ReductionIORate"),  "MBytes/sec", 0 );
  d_runTimeStats.insert( CheckpointIORate,   std::string("CheckpointIORate"), "MBytes/sec", 0 );
  d_runTimeStats.insert( CheckpointReducIORate, std::string("CheckpointReducIORate"), "MBytes/sec", 0 );

  d_runTimeStats.insert( SCIMemoryUsed,      std::string("SCIMemoryUsed"),      bytesStr, 0 );
  d_runTimeStats.insert( SCIMemoryMaxUsed,   std::string("SCIMemoryMaxUsed"),   bytesStr, 0 );
  d_runTimeStats.insert( SCIMemoryHighwater, std::string("SCIMemoryHighwater"), bytesStr, 0 );
  d_runTimeStats.insert( MemoryUsed,         std::string("MemoryUsed"),         bytesStr, 0 );
  d_runTimeStats.insert( MemoryResident,     std::string("MemoryResident"),     bytesStr, 0 );

#ifdef USE_PAPI_COUNTERS
  d_runTimeStats.insert( TotalFlops,  std::string("TotalFlops") , "FLOPS" , 0 );
  d_runTimeStats.insert( TotalVFlops, std::string("TotalVFlops"), "FLOPS" , 0 );
  d_runTimeStats.insert( L2Misses,    std::string("L2Misses")   , "misses", 0 );
  d_runTimeStats.insert( L3Misses,    std::string("L3Misses")   , "misses", 0 );
  d_runTimeStats.insert( TLBMisses,   std::string("TLBMisses")  , "misses", 0 );
#endif

  d_runTimeStats.validate( MAX_TIMING_STATS );

  ResetStats();

#ifdef HAVE_VISIT
  d_doVisIt = false;
#endif     
} // end SimulationController constructor

//______________________________________________________________________
//
SimulationController::~SimulationController()
{
  delete d_restart_archive;

#ifdef USE_PAPI_COUNTERS
  delete m_papi_event_values;
#endif
}
  
//______________________________________________________________________
//
void
SimulationController::setPostProcessFlags( const string & fromDir )
{
  d_postProcessUda   = true;
  d_fromDir          = fromDir;
}

//______________________________________________________________________
//
void
SimulationController::getComponents( void )
{
  d_app = dynamic_cast<ApplicationInterface*>( getPort( "application" ) );

  if( !d_app ) {
    throw InternalError("dynamic_cast of 'd_app' failed!",
                        __FILE__, __LINE__);
  }

  d_scheduler = dynamic_cast<Scheduler*>( getPort("scheduler") );

  if( !d_scheduler ) {
    throw InternalError("dynamic_cast of 'd_scheduler' failed!",
                        __FILE__, __LINE__);
  }

  d_regridder = dynamic_cast<Regridder*>( getPort("regridder") );

  if( d_app->isDynamicRegridding() && !d_regridder ) {
    throw InternalError("dynamic_cast of 'd_regridder' failed!",
                        __FILE__, __LINE__);
  }

  d_output = dynamic_cast<Output*>( getPort("output") );

  if( !d_output ) {
    throw InternalError("dynamic_cast of 'd_output' failed!",
                        __FILE__, __LINE__);
  }
}

//______________________________________________________________________
//
void
SimulationController::releaseComponents( void )
{
  releasePort( "application" );
  releasePort( "load balancer" );
  releasePort( "regridder" );
  releasePort( "output" );

  d_app       = nullptr;
  d_scheduler = nullptr;
  d_regridder = nullptr;
  d_output    = nullptr;
}

//______________________________________________________________________
//

void
SimulationController::doRestart( const string & restartFromDir, int timeStep,
                                 bool fromScratch, bool removeOldDir )
{
  d_restarting          = true;
  d_fromDir             = restartFromDir;
  d_restartTimeStep     = timeStep;
  d_restartFromScratch  = fromScratch;
  d_restartRemoveOldDir = removeOldDir;
}

//______________________________________________________________________
//
void
SimulationController::restartArchiveSetup( void )
{
  // Set up the restart archive now as it is need by the output.
  if( d_restarting ) {
    // Create the DataArchive here, and store it, as it is used a few
    // times. The grid needs to be read before the ProblemSetup, and
    // not all of the data can be read until after ProblemSetup, so
    // DataArchive operations are needed.

    Dir restartFromDir( d_fromDir );
    Dir checkpointRestartDir = restartFromDir.getSubdir( "checkpoints" );

    d_restart_archive = scinew DataArchive( checkpointRestartDir.getName(),
                                            d_myworld->myRank(),
                                            d_myworld->nRanks() );

    vector<int>    indices;
    vector<double> times;

    try {
      d_restart_archive->queryTimesteps( indices, times );
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
    if (d_restartTimeStep == 0) {
      d_restartIndex = 0; // timestep == 0 means use the first timestep
      // reset d_restartTimestep to what it really is
      d_restartTimeStep = indices[0];
    }
    else if (d_restartTimeStep == -1 && indices.size() > 0) {
      d_restartIndex = (unsigned int)(indices.size() - 1); 
      // reset d_restartTimeStep to what it really is
      d_restartTimeStep = indices[indices.size() - 1];
    }
    else {
      for (int index = 0; index < (int)indices.size(); index++)
        if (indices[index] == d_restartTimeStep) {
          d_restartIndex = index;
          break;
        }
    }
      
    if (d_restartIndex == (int) indices.size()) {
      // timestep not found
      ostringstream message;
      message << "Time step " << d_restartTimeStep << " not found";
      throw InternalError(message.str(), __FILE__, __LINE__);
    }

    // Do this call before calling DataArchive::restartInitialize,
    // because problemSetup() creates VarLabels the DataArchive needs.
    d_restart_ps =
      d_restart_archive->getTimestepDocForComponent( d_restartIndex );
  }
}

//______________________________________________________________________
//
void
SimulationController::outputSetup( void )
{
  // Set up the output now as it may be needed by simulation
  // interface.
  d_output->problemSetup( d_ups, d_restart_ps, d_app->getSimulationStateP() );

  d_output->setRunTimeStats( &d_runTimeStats );
  
  // Note: there are other downstream calls to d_output to complete
  // the setup. See finialSetup().
}

//______________________________________________________________________
//
void
SimulationController::schedulerSetup( void )
{

  // Set up the scheduler now as it will be needed by simulation
  // interface.

  d_scheduler->problemSetup( d_ups, d_app->getSimulationStateP() );

  // Additional set up calls.
  d_scheduler->setInitTimestep( true );
  d_scheduler->setRestartInitTimestep( d_restarting );
  d_scheduler->initialize( 1, 1 );
  d_scheduler->clearTaskMonitoring();

  d_scheduler->setRunTimeStats( &d_runTimeStats );
  
  // Note: there are other downstream calls to d_scheduler to complete
  // the setup. See outOfSyncSetup().
}

//______________________________________________________________________
//
void
SimulationController::applicationSetup( void )
{
  d_app->getComponents();

  // Set up the application as it may change the inital grid
  // setup.

  // Note: normally problemSetup would be called here but the grid is
  // needed and it has not yet been created. Further the simulation
  // controller needs the regridder which obviously needs the grid.

  // Further the application may need to change the grid
  // before it is setup.

  // As such, get the application, create the grid, let the
  // simulation interafce make its changes to the grid, setup the
  // grid, then set up the simulation interface. The simulation
  // interface call to problemSetup is done in outOfSyncSetup().
}

//______________________________________________________________________
//
void
SimulationController::gridSetup( void )
{
  // Set up the grid.
  if( d_restarting ) {
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
    d_currentGridP =
      d_restart_archive->queryGrid( d_restartIndex, d_ups, false );
  }
  else /* if( !d_restarting ) */ {
    d_currentGridP = scinew Grid();

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
    d_app->preGridProblemSetup( d_ups, d_currentGridP );

    // Now that the simulation interface has made its changes do the
    // normal grid problemSetup()
    d_currentGridP->problemSetup( d_ups, d_myworld, d_app->isAMR() );
  }

  if( d_currentGridP->numLevels() == 0 ) {
    throw InternalError("No problem (no levels in grid) specified.",
                        __FILE__, __LINE__);
  }

  // Print out meta data
  if ( d_myworld->myRank() == 0 ) {
    d_currentGridP->printStatistics();
    amrout << "Restart grid\n" << *d_currentGridP.get_rep() << "\n";
  }
}

//______________________________________________________________________
//
void
SimulationController::regridderSetup( void )
{
  // Set up the regridder.

  // Do this step before fully setting up the simulation interface so
  // that Switcher (being a simulation interface) can reset the state
  // of the regridder. See outOfSyncSetup().
  if( d_regridder ) {
    d_regridder->problemSetup( d_ups, d_currentGridP, d_app->getSimulationStateP() );
  }
}

//______________________________________________________________________
//
void
SimulationController::loadBalancerSetup( void )
{
  // Set up the load balancer.
  d_lb = d_scheduler->getLoadBalancer();

  //  Set the dimensionality of the problem.
  IntVector low, high, size;
  d_currentGridP->getLevel(0)->findCellIndexRange(low, high);

  size = high - low -
    d_currentGridP->getLevel(0)->getExtraCells()*IntVector(2,2,2);
  
  d_lb->setDimensionality(size[0] > 1, size[1] > 1, size[2] > 1);
 
  // In addition, do this step after regridding setup as the minimum
  // patch size that the regridder will create will be known.
  d_lb->problemSetup( d_ups, d_currentGridP, d_app->getSimulationStateP() );

  d_lb->setRunTimeStats( &d_runTimeStats );
}

//______________________________________________________________________
//
void
SimulationController::outOfSyncSetup()
{
  // Complete the setup of the application and scheduler that
  // could not be completed until the grid was setup.
  
  // The simulation interface was initialized earlier because the it
  // was needed to possibly set the grid's extra cells before the
  // grid's ProblemSetup was called (it can not be done afterwards).

  // Do this step after setting up the regridder so that Switcher
  // (being a simulation interface) can reset the state of the
  // regridder.

  // Pass the d_restart_ps to the component's problemSetup.  For
  // restarting, pull the <MaterialProperties> from the d_restart_ps.
  // If the properties are not available, then pull the properties
  // from the d_ups instead.  This step needs to be done before
  // DataArchive::restartInitialize.
  d_app->problemSetup(d_ups, d_restart_ps, d_currentGridP);

  // The scheduler was setup earlier because the simulation interface
  // needed it.

  // Complete the setup of the scheduler.
  d_scheduler->advanceDataWarehouse( d_currentGridP, true );
}

//______________________________________________________________________
//
void
SimulationController::timeStateSetup()
{
  // Set up the time state.

  // Restarting so initialize time state using the archive data.
  if( d_restarting ) {
    simdbg << "Restarting... loading data\n";

    double simTimeStart;
    
    d_restart_archive->restartInitialize( d_restartIndex,
                                          d_currentGridP,
                                          d_scheduler->get_dw(1),
                                          d_lb,
                                          &simTimeStart );

    // Set the time step to the restart time step.
    d_app->setTimeStep( d_restartTimeStep );

    // Set the simulation time to the restart simulation time.
    d_app->setSimTimeStart( simTimeStart );

    // Set the delta T to the restart delta T.
    d_app->restartDelT( d_restart_archive->getOldDelt( d_restartIndex ) );
    
    // Tell the scheduler the generation of the re-started simulation.
    // (Add +1 because the scheduler will be starting on the next
    // time step.)
    d_scheduler->setGeneration( d_restartTimeStep + 1 );
      
    // This delete is an enigma. If it is called then memory is not
    // leaked, but sometimes if is called, then everything segfaults.
    // delete d_restart_archive;
  }
  else /* if( !d_restarting ) */ {
    d_app->setTimeStep( 0 );
    d_app->setSimTimeStart( 0 );
    d_app->setDelT( 0 );    
  }
}

//______________________________________________________________________
//
void
SimulationController::finalSetup()
{
  // Finalize the shared state/materials
  d_app->getSimulationStateP()->finalizeMaterials();

  // The output was initalized earlier because the simulation
  // interface needed it. 
  
  // This step is done after the call to d_app->problemSetup to get
  // the defaults set by the simulation interface into the input.xml,
  // which the output writes along with index.xml
  d_output->initializeOutput(d_ups);

  // This step is done after the output is initalized so that global
  // reduction ouput vars are copied to the new uda. Further, it must
  // be called after timeStateSetup() is call so that checkpoints are
  // copied to the new uda as well.
  if( d_restarting ) {
    Dir dir( d_fromDir );
    d_output->restartSetup( dir, 0, d_restartTimeStep,
			    d_app->getSimTimeStart(),
                            d_restartFromScratch, d_restartRemoveOldDir );
  }

  // Miscellaneous initializations.
  ProblemSpecP amr_ps = d_ups->findBlock("AMR");
  if( amr_ps ) {
    amr_ps->get( "doMultiTaskgraphing", d_do_multi_taskgraphing );
  }

#ifdef HAVE_VISIT
  if( getVisIt() ) {
    d_app->getDebugStreams().push_back( &dbg );
    d_app->getDebugStreams().push_back( &dbgTime );
    d_app->getDebugStreams().push_back( &simdbg );
    d_app->getDebugStreams().push_back( &stats );
    d_app->getDebugStreams().push_back( &istats );
  }
#endif
}

//______________________________________________________________________
//
void SimulationController::ResetStats( void )
{
  d_runTimeStats.reset( 0 );  
  d_otherStats.reset( 0 );  
}

//______________________________________________________________________
//

void
SimulationController::ReportStats( bool header /* = false */ )
{
  // Get and reduce the performance runtime stats
  getMemoryStats();
  getPAPIStats();

  d_runTimeStats.reduce(d_regridder &&
			d_regridder->useDynamicDilation(),
			d_myworld );
  
  d_otherStats.reduce(d_regridder &&
		      d_regridder->useDynamicDilation(),
		      d_myworld );

  // Reduce the MPI runtime stats.
  MPIScheduler * mpiScheduler = dynamic_cast<MPIScheduler*>( d_scheduler.get_rep() );

  if( mpiScheduler )
    mpiScheduler->mpi_info_.reduce( d_regridder &&
                                    d_regridder->useDynamicDilation(),
                                    d_myworld );
  
  // Print the stats for this time step
  if( d_myworld->myRank() == 0 && header ) {
    dbg << std::endl;
    dbg << "Simulation and run time stats are reported "
        << "at the end of each time step" << std::endl;
    dbg << "EMA == Wall time as an exponential moving average "
        << "using a window of the last " << walltimers.getWindow()
        << " time steps" << std::endl;

    dbg.flush();
    cout.flush();
  }
  
  // With the sum reduces, use double, since with memory it is possible that
  // it will overflow
  double        avg_memused      = d_runTimeStats.getAverage( SCIMemoryUsed );
  unsigned long max_memused      = d_runTimeStats.getMaximum( SCIMemoryUsed );
  int           max_memused_rank = d_runTimeStats.getRank( SCIMemoryUsed );

  double        avg_highwater      = d_runTimeStats.getAverage( SCIMemoryHighwater );
  unsigned long max_highwater      = d_runTimeStats.getMaximum( SCIMemoryHighwater );
  int           max_highwater_rank = d_runTimeStats.getRank( SCIMemoryHighwater );
    
  // Sum up the average time for overhead related components. These
  // same values are used in SimulationState::getOverheadTime.
  double overhead_time =
    (d_runTimeStats.getAverage(CompilationTime)           +
     d_runTimeStats.getAverage(RegriddingTime)            +
     d_runTimeStats.getAverage(RegriddingCompilationTime) +
     d_runTimeStats.getAverage(RegriddingCopyDataTime)    +
     d_runTimeStats.getAverage(LoadBalancerTime));

  // Sum up the average times for simulation components. These
  // same values are used in SimulationState::getTotalTime.
  double total_time =
    (overhead_time +
     d_runTimeStats.getAverage(TaskExecTime)       +
     d_runTimeStats.getAverage(TaskLocalCommTime)  +
     d_runTimeStats.getAverage(TaskWaitCommTime) +
     d_runTimeStats.getAverage(TaskReduceCommTime)   +
     d_runTimeStats.getAverage(TaskWaitThreadTime));
  
    // Calculate percentage of time spent in overhead.
  double percent_overhead = overhead_time / total_time;
  
  double overheadAverage = 0;

  // Set the overhead percentage. Ignore the first sample as that is
  // for initalization.
  if (d_nSamples) {
    overheadValues[overheadIndex] = percent_overhead;

    double overhead = 0;
    double weight = 0;

    int sample_size = min(d_nSamples, OVERHEAD_WINDOW);

    // Calculate total weight by incrementing through the overhead
    // sample array backwards and multiplying samples by the weights
    for (int i = 0; i < sample_size; ++i) {
      unsigned int index = (overheadIndex - i + OVERHEAD_WINDOW) % OVERHEAD_WINDOW;
      overhead += overheadValues[index] * overheadWeights[i];
      weight   += overheadWeights[i];
    }

    // Increment the overhead index
    overheadIndex = (overheadIndex + 1) % OVERHEAD_WINDOW;

    overheadAverage = overhead / weight;

    if( d_regridder )
      d_regridder->setOverheadAverage(overheadAverage);
  } 

  // Output time step statistics...
  if (istats.active()) {
    istats << "Run time performance stats" << std::endl;

    for (unsigned int i = 0; i < d_runTimeStats.size(); i++) {
      RunTimeStatsEnum e = (RunTimeStatsEnum) i;

      if (d_runTimeStats[e] > 0) {
        istats << "rank: " << d_myworld->myRank() << " "
	       << left << setw(19) << d_runTimeStats.getName(e) << " ["
               << d_runTimeStats.getUnits(e) << "]: " << d_runTimeStats[e] << "\n";
      }
    }

    if (d_otherStats.size())
      istats << "Other performance stats" << std::endl;

    for (unsigned int i = 0; i < d_otherStats.size(); i++) {
      if (d_otherStats[i] > 0) {
        istats << "rank: " << d_myworld->myRank() << " "
	       << left << setw(19) << d_otherStats.getName(i) << " ["
               << d_otherStats.getUnits(i) << "]: " << d_otherStats[i] << "\n";
      }
    }
  } 

  // Update the moving average and get the wall time for this time step.
  Timers::nanoseconds timeStep = walltimers.updateExpMovingAverage();

  if( d_myworld->myRank() == 0 )
  {
    ostringstream message;
    message << left
            << "Timestep "   << setw(8)  << d_app->getTimeStep()
            << "Time="       << setw(12) << d_app->getSimTime()
//          << "delT="       << setw(12) << d_app->getPrevDelT()
            << "Next delT="  << setw(12) << d_app->getDelT()

            << "Wall Time=" << setw(10) << walltimers.GetWallTime()
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
              << " (max on rank: " << setw(6) << max_memused_rank << ")";

      if (avg_highwater)
        message << "    Highwater Memory Used=" << setw(10)
                << ProcessInfo::toHumanUnits((unsigned long)avg_highwater)
                << " (avg) " << setw(10)
                << ProcessInfo::toHumanUnits(max_highwater)
                << " (max on rank: " << setw(6) << max_highwater_rank << ")";
    }

    dbg << message.str() << "\n";
    dbg.flush();
    cout.flush();

    // Ignore the first sample as that is for initialization.
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
      
      for (unsigned int i=0; i<d_runTimeStats.size(); ++i)
      {
        RunTimeStatsEnum e = (RunTimeStatsEnum) i;
        
        if (d_runTimeStats.getMaximum(e) > 0)
        {
          stats << "  " << left
                << setw(21) << d_runTimeStats.getName(e)
                << "[" << setw(10) << d_runTimeStats.getUnits(e) << "]"
                << " : " << setw(12) << d_runTimeStats.getAverage(e)
                << " : " << setw(12) << d_runTimeStats.getMaximum(e)
                << " : " << setw(10) << d_runTimeStats.getRank(e)
                << " : " << setw(10)
                << 100.0 * (1.0 - (d_runTimeStats.getAverage(e) /
                                   d_runTimeStats.getMaximum(e)))
                << "\n";
        }
      }
      
      // Report the overhead percentage.
      if( !std::isnan(overheadAverage) ) {
        stats << "  Percentage of time spent in overhead : "
              << overheadAverage * 100.0 <<  "\n";
      }

      if( d_otherStats.size() )
        stats << "Other performance stats" << std::endl;
      
      for (unsigned int i=0; i<d_otherStats.size(); ++i)
      {
        if (d_otherStats.getMaximum(i) > 0)
        {
          stats << "  " << left
                << setw(21) << d_otherStats.getName(i)
                << "["   << setw(10) << d_otherStats.getUnits(i) << "]"
                << " : " << setw(12) << d_otherStats.getAverage(i)
                << " : " << setw(12) << d_otherStats.getMaximum(i)
                << " : " << setw(10) << d_otherStats.getRank(i)
                << " : " << setw(10)
                << 100.0 * (1.0 - (d_otherStats.getAverage(i) / d_otherStats.getMaximum(i)))
                << "\n";
        }
      }
    }
  
    // Ignore the first sample as that is for initialization.
    if (dbgTime.active() && d_nSamples ) {
      double realSecondsNow =
        timeStep.seconds() / d_app->getDelT();
      double realSecondsAvg = walltimers.TimeStep().seconds() / (d_app->getSimTime()-d_app->getSimTimeStart());

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

  d_runTimeStats[SCIMemoryUsed] = memUsed;
  d_runTimeStats[SCIMemoryMaxUsed] = maxMemUsed;
  d_runTimeStats[SCIMemoryHighwater] = highwater;

  if (ProcessInfo::isSupported(ProcessInfo::MEM_SIZE)) {
    d_runTimeStats[MemoryUsed] = ProcessInfo::getMemoryUsed();
  }

  if (ProcessInfo::isSupported(ProcessInfo::MEM_RSS)) {
    d_runTimeStats[MemoryResident] = ProcessInfo::getMemoryResident();
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
      sprintf(filename, "%s.%d", filenamePrefix, d_myworld->myRank());

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

    *mallocPerProcStream << "Proc " << d_myworld->myRank() << "   ";
    *mallocPerProcStream << "TimeStep " << d_app->getTimeStep() << "   ";

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
        d_runTimeStats[ iter->second.m_sim_stat_name ] =
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
  
#ifdef HAVE_VISIT
bool
SimulationController::CheckInSitu( visit_simulation_data * visitSimData,
                                   bool                    first )
{
  // If VisIt has been included into the build, check the lib sim
  // state to see if there is a connection and if so check to see if
  // anything needs to be done.
  if( getVisIt() ) {
    // Note this timer is used as a laptimer so the stop must come
    // after the lap time is taken so not to affect it.
    walltimers.TimeStep.stop();

    walltimers.InSitu.start();

    // Update all of the simulation grid and time dependent variables.
    visit_UpdateSimData( visitSimData, d_currentGridP,
			 d_app->getSimTime(),
			 d_app->getTimeStep(),
			 d_app->getPrevDelT(),
			 d_app->getDelT(),
                         first,
			 d_app->isLastTimeStep(walltimers.GetWallTime()) );

    // Check the state - if the return value is true the user issued
    // a termination.
    if( visit_CheckState( visitSimData ) ) {
      return true;
    }

    // This function is no longer used as last is now used in the
    // UpdateSimData which in turn will flip the state.
      
    // Check to see if at the last iteration. If so stop so the
    // user can have once last chance see the data.
    // if( visitSimData->stopAtLastTimeStep && last )
    // visit_EndLibSim( visitSimData );

    // The user may have adjusted delT so get it from the data
    // warehouse. If not then this call is a no-op.
    DataWarehouse* newDW = d_scheduler->getLastDW();
    delt_vartype delT_var;
    newDW->get( delT_var, d_app->getDelTLabel() );
    d_app->setDelT( delT_var );

    // Add the modified variable information into index.xml file.
    getOutput()->writeto_xml_files(d_app->getTimeStep(),
				   visitSimData->modifiedVars);

    walltimers.InSitu.stop();

    // Note this timer is used as a laptimer.
    walltimers.TimeStep.start();
  }

  return false;
}
#endif

} // namespace Uintah
