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

#include <CCA/Components/SimulationController/AMRSimulationController.h>

#include <CCA/Components/PostProcessUda/PostProcess.h>
#include <CCA/Components/Regridder/PerPatchVars.h>
#include <CCA/Ports/DataWarehouse.h>
#include <CCA/Ports/LoadBalancerPort.h>
#include <CCA/Ports/Output.h>
#include <CCA/Ports/ProblemSpecInterface.h>
#include <CCA/Ports/Regridder.h>
#include <CCA/Ports/Scheduler.h>
#include <CCA/Ports/ApplicationInterface.h>

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
#include <Core/Util/DebugStream.h>

#ifdef HAVE_CUDA
#  include <CCA/Components/Schedulers/GPUGridVariableInfo.h>
#endif

#include <sci_defs/malloc_defs.h>
#include <sci_defs/gperftools_defs.h>

#include <iostream>
#include <iomanip>
#include <ctime>

using namespace Uintah;

namespace {

DebugStream amrout(      "AMR"                    , false);
DebugStream dbg(         "AMRSimulationController", false);
DebugStream dbg_barrier( "MPIBarriers"            , false);
DebugStream dbg_dwmem(   "LogDWMemory"            , false);
DebugStream gprofile(    "CPUProfiler"            , false);
DebugStream gheapprofile("HeapProfiler"           , false);
DebugStream gheapchecker("HeapChecker"            , false);

}

AMRSimulationController::AMRSimulationController( const ProcessorGroup * myworld
                                                ,       ProblemSpecP     pspec
                                                )
  :  SimulationController( myworld, pspec )
{
}

//______________________________________________________________________
//
void
AMRSimulationController::run()
{
  MALLOC_TRACE_TAG_SCOPE("AMRSimulationController::run()");

  getComponents();

  bool first = true;

  // If VisIt has been included into the build, initialize the lib sim
  // so that a user can connect to the simulation via VisIt.
#ifdef HAVE_VISIT
  visit_simulation_data visitSimData;

  if( getVisIt() ) {
    visitSimData.simController = this;
    visitSimData.runMode = getVisIt();

    // Running with VisIt so add in the variables that the user can
    // modify.
    // variable 1 - Must start with the component name and have NO
    // spaces in the var name.
    ApplicationInterface::interactiveVar var;
    var.name     = "Scrub-Data-Warehouse";
    var.type     = Uintah::TypeDescription::bool_type;
    var.value    = (void *) &(m_scrub_datawarehouse);
    var.range[0] = 0;
    var.range[1] = 1;
    var.modifiable = true;
    var.recompile  = false;
    var.modified   = false;
    m_app->getStateVars().push_back( var );

    m_app->getDebugStreams().push_back( &amrout );
    m_app->getDebugStreams().push_back( &dbg );
    m_app->getDebugStreams().push_back( &dbg_barrier );
    m_app->getDebugStreams().push_back( &dbg_dwmem );
    m_app->getDebugStreams().push_back( &gprofile );
    m_app->getDebugStreams().push_back( &gheapprofile );
    m_app->getDebugStreams().push_back( &gheapchecker );

    visit_InitLibSim( &visitSimData );
  }
#endif


#ifdef USE_GPERFTOOLS
  if (gprofile.active()){
    char gprofname[512];
    sprintf(gprofname, "cpuprof-rank%d", d_myworld->myRank());
    ProfilerStart(gprofname);
  }
  if (gheapprofile.active()){
    char gheapprofname[512];
    sprintf(gheapprofname, "heapprof-rank%d", d_myworld->myRank());
    HeapProfilerStart(gheapprofname);
  }

  HeapLeakChecker * heap_checker=nullptr;
  if (gheapchecker.active()){
    if (!gheapprofile.active()){
      char gheapchkname[512];
      sprintf(gheapchkname, "heapchk-rank%d", d_myworld->myRank());
      heap_checker= new HeapLeakChecker(gheapchkname);
    } else {
      std::cout<< "HEAPCHECKER: Cannot start with heapprofiler" << std::endl;
    }
  }
#endif


  // ____________________________________________________________________
  // Begin the zero time step. Which is either initialization or restart.
  
  // Start the wall timer for the initialization time step
  m_wall_timers.TimeStep.reset( true );

  
  //---------------------------------------------------------------------
  // The order of the setup is important. See the individual component
  // setup calls for more details.
  // --------------------------------------------------------------------

  // Setup the restart archive first as the output needs it.
  restartArchiveSetup();

  // Setup the output as the application interface needs it.
  outputSetup();

  // Setup the grid using the restart archive and application interface.
  gridSetup();

  // Setup the regridder using the grid.
  regridderSetup();

  // Setup the scheduler.
  schedulerSetup();

  // Setup the load balancer using the grid.
  loadBalancerSetup();

  // Setup the application using the restart archive and under the
  // hood the output and scheduler.
  applicationSetup();

  // Setup the time state using the restart archive, grid, scheduler,
  // and load balancer.
  timeStateSetup();

  // Setup the final bits including the output.
  finalSetup();

  // Once the grid is set up pass it on to the GPU.
#ifdef HAVE_CUDA
  GpuUtilities::assignPatchesToGpus( m_current_gridP );
#endif

  //  postProcessUda - needs to be done after the time info is read but
  //  before the initial time step.
  if( m_post_process_uda ) {
    Dir fromDir( m_from_dir );
    m_output->postProcessUdaSetup( fromDir );
    m_app->getSimulationTime()->m_delt_factor = 1;
    m_app->getSimulationTime()->m_delt_min    = 0;
    m_app->getSimulationTime()->m_delt_max    = 1e99;
    m_app->getSimulationTime()->m_init_time = static_cast<PostProcessUda*>(m_app)->getInitialTime();
    m_app->getSimulationTime()->m_max_time  = static_cast<PostProcessUda*>(m_app)->getMaxTime();
    m_app->getSimulationTime()->m_max_delt_increase = 1e99;
    m_app->getSimulationTime()->m_max_initial_delt  = 1e99;
  }

  // Setup, compile, and run the taskgraph for the initialization time step
  doInitialTimeStep();
  
  // Finalize the application system vars including getting the next
  // delta T - Do this before reporting stats or the in-situ so the
  // new delT is available.
  m_app->finalizeSystemVars( m_scheduler );
  
  // Report all of the stats before doing any possible in-situ work
  // as that effects the lap timer for the time steps.
  ReportStats( first );

  // If compiled with VisIt check the in-situ status for work.
#ifdef HAVE_VISIT
  if( CheckInSitu( &visitSimData, first ) )
    exit(0);
#endif      

  // Update the profiler weights
  m_lb->finalizeContributions(m_current_gridP);
  m_lb->resetCostForecaster();

  // Done with all the initialization.
  m_scheduler->setInitTimestep(false);

  // ____________________________________________________________________
  // End the zero time step. Which is either initialization or restart.
  
#ifndef DISABLE_SCI_MALLOC
  AllocatorSetDefaultTagLineNumber( m_app->getTimeStep() );
#endif

#ifdef HAVE_PIDX
  // Setup for PIDX
  static bool pidx_need_to_recompile = false;
  static bool pidx_restore_nth_rank  = false;
  static int  pidx_requested_nth_rank = -1;
  
  if( m_output->savingAsPIDX() ) {
    if( pidx_requested_nth_rank == -1 ) {
      pidx_requested_nth_rank = m_lb->getNthRank();

      if( pidx_requested_nth_rank > 1 ) {
        proc0cout << "Input file requests output to be saved by every "
                  << pidx_requested_nth_rank << "th processor.\n"
                  << "  - However, setting output to every processor "
                  << "until a checkpoint is reached." << std::endl;
        m_lb->setNthRank( 1 );
        m_lb->possiblyDynamicallyReallocate( m_current_gridP, LoadBalancerPort::regrid );
        m_output->setSaveAsPIDX();
      }
    }
  }
#endif

  // Set the timer for the main loop. This timer is sync'ed with the
  // simulation time to get a measurement of the simulation to wall time.
  m_wall_timers.TimeStep.reset( true );
    
  // The main loop where the specified application problem is solved.
  while( !m_app->isLastTimeStep( m_wall_timers.GetWallTime() ) ) {

    // Perform a bunch of housekeeping operations at the top of the
    // loop. Performing them here asures that everything is ready
    // after the inital time step. It also reduces duplicate code.

    // Before any work is done including incrementing the time step
    // check to see if this iteration may be the last one. The
    // DataArchiver uses it for determining whether to output or
    // checkpoint the last time step. Maybelast uses the wall time and
    // is sync'd across all ranks.

    // The predicted time is a best guess at what the wall time will be
    // when the time step is finished. It is currently used only for
    // outputing and checkpointing. Both of which typically take much
    // longer than the simulation calculation.
    double walltime = m_wall_timers.GetWallTime() + 1.5 * m_wall_timers.ExpMovingAverage().seconds();

    m_output->maybeLastTimestep( m_app->maybeLastTimeStep( walltime ) );
    
    // Set the current wall time for this rank (i.e. this value is NOT
    // sync'd across all ranks). The Data Archive uses it for
    // determining when to output or checkpoint.
    m_output->setElapsedWallTime( m_wall_timers.GetWallTime() );
    
    // Get the next output checkpoint time step. This step is not done
    // in beginOutputTimestep because the original values are needed
    // to compare with if there is a timestep restart so it is
    // performed here. At this point the time step, sim time, and all
    // wall time are all in sync.

    m_output->findNext_OutputCheckPointTimestep( m_app->getTimeStep(),
						                                     m_app->getSimTime(),
						                                     m_app->getDelT(),
						                                     first && m_restarting );

    // Reset the runtime performance stats
    ResetStats();
    
    // Reset memory use tracking variable
    m_scheduler->resetMaxMemValue();
    
    // Clear the task monitoring.
    m_scheduler->clearTaskMonitoring();
    
    // Increment (by one) the current time step number so components
    // know what time step they are on.
    m_app->incrementTimeStep( m_current_gridP );

    // Ready for the next time step. 

#ifdef USE_GPERFTOOLS
    if (gheapprofile.active()){
      char heapename[512];
      sprintf(heapename, "Timestep %d", m_app->getTimeStep());
      HeapProfilerDump(heapename);
    }
#endif
     
    MALLOC_TRACE_TAG_SCOPE( "AMRSimulationController::run()::control loop" );

    if (dbg_barrier.active()) {
      for (int i = 0; i < 5; ++i) {
        m_barrier_times[i] = 0;
      }
    }

#ifdef HAVE_PIDX
    bool pidx_checkpointing = false;

    if( m_output->savingAsPIDX() ) {

      pidx_checkpointing =
        // Output the checkpoint based on the simulation time.
        ( (m_output->getCheckpointInterval() > 0 &&
           (m_app->getSimTime()+m_app->getDelT()) >= m_output->getNextCheckpointTime()) ||

          // Output the checkpoint based on the time step interval.
          (m_output->getCheckpointTimestepInterval() > 0 &&
           m_app->getTimeStep() == m_output->getNextCheckpointTimestep()) ||

          // Output the checkpoint based on the being the last time step.
          (m_app->getSimulationTime()->m_max_wall_time > 0 && m_ouput->maybeLast()) );

      // When using the wall clock time for checkpoints, rank 0
      // determines the wall time and sends it to all other ranks.
      if( m_output->getCheckpointWalltimeInterval() > 0 ) {
        double tmp_time = -1;

        if( Parallel::getMPIRank() == 0 ) {
          tmp_time = m_output->getElapsedWallTime();
        }
        Uintah::MPI::Bcast( &tmp_time, 1, MPI_DOUBLE, 0, d_myworld->getComm() );

        if( tmp_time >= m_output->getNextCheckpointWalltime() )
          pidx_checkpointing = true;    
      }

      
      // Checkpointing
      if( pidx_checkpointing ) {

        if( pidx_requested_nth_rank > 1 ) {
          proc0cout << "This is a checkpoint time step (" << m_app->getTimeStep()
                    << ") - need to recompile with nth proc set to: "
                    << pidx_requested_nth_rank << std::endl;

          m_lb->setNthRank( pidx_requested_nth_rank );
          m_lb->possiblyDynamicallyReallocate( m_current_gridP, LoadBalancerPort::regrid );
          m_output->setSaveAsUDA();
          pidx_need_to_recompile = true;
        }
      }

      // Output
      if( ( m_output->getOutputTimestepInterval() > 0 &&
            m_app->getTimeStep() == m_output->getNextOutputTimestep() ) ||
          ( m_output->getOutputInterval() > 0         &&
            ( m_app->getSimTime() + m_app->getDelT() ) >= m_output->getNextOutputTime() ) ) {

        proc0cout << "This is an output time step: " << m_app->getTimeStep() << "\n";

        if( pidx_need_to_recompile ) { // If this is also a checkpoint time step
          proc0cout << "   Postposing as this is also a checkpoint time step...\n";
          m_output->postponeNextOutputTimestep();
        }
      }
    }
#endif

    // Regridding
    if (m_regridder) {
      // If not the first time step or restarting check for regridding
      if ((!first || m_restarting) && m_regridder->needsToReGrid(m_current_gridP)) {
        proc0cout << " Need to regrid." << std::endl;
        doRegridding( false );
      }
      // Covers single-level regridder case (w/ restarts)
      else if (m_regridder->doRegridOnce() && m_regridder->isAdaptive()) {
        proc0cout << " Regridding once." << std::endl;
        m_scheduler->setRestartInitTimestep( false );
        doRegridding( false );
        m_regridder->setAdaptivity( false );
      }
    }

    // Compute number of dataWarehouses - multiplies by the time
    // refinement ratio for each level.
    int totalFine = 1;

    if (!m_app->isLockstepAMR()) {
      for (int i = 1; i < m_current_gridP->numLevels(); ++i) {
        totalFine *= m_current_gridP->getLevel(i)->getRefinementRatioMaxDim();
      }
    }
     
    if (dbg_dwmem.active()) {
      // Remember, this isn't logged if DISABLE_SCI_MALLOC is set (so
      // usually in optimized mode this will not be run.)
      m_scheduler->logMemoryUse();
      std::ostringstream fn;
      fn << "alloc." << std::setw(5) << std::setfill('0') << d_myworld->myRank() << ".out";
      std::string filename(fn.str());

#ifndef DISABLE_SCI_MALLOC
      DumpAllocator(DefaultAllocator(), filename.c_str());
#endif

    }

    if (dbg_barrier.active()) {
      m_barrier_timer.reset( true);
      Uintah::MPI::Barrier(d_myworld->getComm());
      m_barrier_times[2] += m_barrier_timer().seconds();
    }

    // This step is a hack but it is the only way to get a new grid
    // from postProcessUda and needs to be done before
    // advanceDataWarehouse is called.
    if (m_post_process_uda) {
      m_current_gridP = static_cast<PostProcessUda*>(m_app)->getGrid();
    }

    // After one step (either time step or initialization) and the
    // updating of delta T finalize the old time step, e.g. finalize
    // and advance the Datawarehouse
    m_scheduler->advanceDataWarehouse( m_current_gridP );

#ifndef DISABLE_SCI_MALLOC
    AllocatorSetDefaultTagLineNumber( m_app->getTimeStep() );
#endif

    bool nr = needRecompile();
    
#ifdef HAVE_PIDX
    nr = (nr || pidx_need_to_recompile || pidx_restore_nth_rank);
#endif         

    if( nr || first ) {
    
      // Recompile taskgraph, re-assign BCs, reset recompile flag.      
      if (nr) {
        m_current_gridP->assignBCS(m_grid_ps, m_lb);
        m_current_gridP->performConsistencyCheck();
        m_recompile_taskgraph = false;
      }
    
#ifdef HAVE_PIDX
      if( pidx_requested_nth_rank > 1 ) {      
        if( pidx_restore_nth_rank ) {
          proc0cout << "This is the time step following a checkpoint - "
                    << "need to put the task graph back with a recompile - "
                    << "setting nth output to 1\n";
          m_lb->setNthRank( 1 );
          m_lb->possiblyDynamicallyReallocate( m_current_gridP, LoadBalancerPort::regrid );
          m_output->setSaveAsPIDX();
          pidx_restore_nth_rank = false;
        }
        
        if( pidx_need_to_recompile ) {
          // Don't need to recompile on the next time step as it will
          // happen on this one.  However, the nth rank value will
          // need to be restored after this time step, so set
          // pidx_restore_nth_rank to true.
          pidx_need_to_recompile = false;
          pidx_restore_nth_rank = true;
        }
      }
#endif

      m_scheduler->setRestartInitTimestep( false );
      recompile( totalFine );
    }
    else {
      // This is not correct if we have switched to a different
      // component, since the delT will be wrong
      m_output->finalizeTimestep( m_app->getTimeStep(),
		                              m_app->getSimTime(),
			                            m_app->getDelT(),
			                            m_current_gridP,
			                            m_scheduler,
			                            0 );
    }

    if( dbg_barrier.active() ) {
      m_barrier_timer.reset( true );
      Uintah::MPI::Barrier( d_myworld->getComm() );
      m_barrier_times[3] += m_barrier_timer().seconds();
    }

    // Execute the current time step, restarting if necessary.
    executeTimeStep( totalFine );
      
    // If debugging, output the barrier times.
    if( dbg_barrier.active() ) {
      m_barrier_timer.reset( true );
      Uintah::MPI::Barrier( d_myworld->getComm() );
      m_barrier_times[4] += m_barrier_timer().seconds();

      double avg[5];
      Uintah::MPI::Reduce( m_barrier_times, avg, 5, MPI_DOUBLE, MPI_SUM, 0, d_myworld->getComm() );

      if( d_myworld->myRank() == 0 ) {
        std::cout << "Barrier Times: ";
        for( int i = 0; i < 5; ++i ) {
          avg[i] /= d_myworld->nRanks();
          std::cout << avg[i] << " ";
        }
        std::cout << "\n";
      }
    }

#ifdef HAVE_PIDX
    // For PIDX only save timestep.xml when checkpointing.  Normal
    // time step dumps using PIDX do not need to write the xml
    // information.      
    if( !m_output->savingAsPIDX() || (m_output->savingAsPIDX() && pidx_checkpointing) )
#endif    
    {
      // If PIDX is not being used write timestep.xml for both
      // checkpoints and time step dumps.
      m_output->writeto_xml_files( m_app->getTimeStep(),
				                           m_app->getSimTime(),
				                           m_app->getDelT(),
				                           m_current_gridP );
    }
    
    // Finalize the application system vars including getting the next
    // delta T - Do this before reporting stats or the in-situ so the
    // new delT is available.
    m_app->finalizeSystemVars( m_scheduler );
  
    // Report all of the stats before doing any possible in-situ work
    // as that affects the lap timer for the time steps.
    ReportStats( false );

    // If compiled with VisIt check the in-situ status for work.
#ifdef HAVE_VISIT
    if( CheckInSitu( &visitSimData, false ) ) {
      break;
    }
#endif      

    // Update the profiler weights
    m_lb->finalizeContributions(m_current_gridP);

    // Done with the first time step.
    if( first ) {
      m_scheduler->setRestartInitTimestep( false );
      first = false;
    }
  } // end while main time loop (time is not up, etc)
  
  // m_ups->releaseDocument();

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

//______________________________________________________________________
//
void
AMRSimulationController::doInitialTimeStep()
{
  MALLOC_TRACE_TAG_SCOPE( "AMRSimulationController::doInitialTimeStep()" );

  m_scheduler->mapDataWarehouse(Task::OldDW, 0);
  m_scheduler->mapDataWarehouse(Task::NewDW, 1);
  m_scheduler->mapDataWarehouse(Task::CoarseOldDW, 0);
  m_scheduler->mapDataWarehouse(Task::CoarseNewDW, 1);
  
  Timers::Simple taskGraphTimer;          // Task graph time

  if( m_restarting ) {

    // for dynamic lb's, set up restart patch config
    m_lb->possiblyDynamicallyReallocate( m_current_gridP, LoadBalancerPort::restart );

    // tsaad & bisaac: At this point, during a restart, a grid does
    // NOT have knowledge of the boundary conditions.  (See other
    // comments in SimulationController.cc for why that is the
    // case). Here, and given a legitimate load balancer, we can
    // assign the BCs to the grid in an efficient manner.
    m_current_gridP->assignBCS( m_grid_ps, m_lb );
    m_current_gridP->performConsistencyCheck();

    m_app->restartInitialize();

    for (int i = m_current_gridP->numLevels() - 1; i >= 0; i--) {
      m_app->scheduleRestartInitialize( m_current_gridP->getLevel(i), m_scheduler );
    }

    // Initialize the system var (time step and simulation time)
    m_app->scheduleInitializeSystemVars( m_current_gridP,
					                               m_lb->getPerProcessorPatchSet(m_current_gridP),
					                               m_scheduler );
  
    taskGraphTimer.reset( true );
    m_scheduler->compile();
    taskGraphTimer.stop();

    m_runtime_stats[ CompilationTime ] += taskGraphTimer().seconds();

    proc0cout << "Done with taskgraph compile (" << taskGraphTimer().seconds() << " seconds)\n";

    // No scrubbing for initial step
    m_scheduler->get_dw( 1 )->setScrubbing( DataWarehouse::ScrubNone );
    m_scheduler->execute();

    // Now we know we're done with any additions to the new DW - finalize it
    m_scheduler->get_dw( 1 )->finalize();

    if (m_regridder && m_regridder->isAdaptive()) {
      // On restart:
      //   we must set up the tasks (but not compile) so we can have the
      //   initial OldDW Requirements in order to regrid straight away
      for( int i = m_current_gridP->numLevels() - 1; i >= 0; i-- ) {
        m_app->scheduleTimeAdvance(m_current_gridP->getLevel(i), m_scheduler);
      }
    }

    // Monitoring tasks must be scheduled last!!
    for (int i = m_current_gridP->numLevels() - 1; i >= 0; i--) {
      m_scheduler->scheduleTaskMonitoring(m_current_gridP->getLevel(i));
    }
  }
  else /* if( !m_restarting ) */ {
    // for dynamic lb's, set up initial patch config
    m_lb->possiblyDynamicallyReallocate( m_current_gridP, LoadBalancerPort::init );
    
    m_current_gridP->assignBCS( m_grid_ps, m_lb );
    m_current_gridP->performConsistencyCheck();

    bool needNewLevel = false;

    do {
      proc0cout << "\nCompiling initialization taskgraph...\n";

      // Initialize the CFD and/or MPM data
      for( int i = m_current_gridP->numLevels() - 1; i >= 0; i-- ) {
        m_app->scheduleInitialize(m_current_gridP->getLevel(i), m_scheduler);

        if( m_regridder ) {
          // So we can initially regrid
          m_regridder->scheduleInitializeErrorEstimate(m_current_gridP->getLevel(i));
          m_app->scheduleInitialErrorEstimate(m_current_gridP->getLevel(i), m_scheduler);
          
          // We don't use error estimates if we don't make another
          // level, so don't dilate.
          if( i < m_regridder->maxLevels() - 1 ) {
            m_regridder->scheduleDilation(m_current_gridP->getLevel(i),
					  m_app->isLockstepAMR());
          }
        }
      }

      // Compute the next time step.
      scheduleComputeStableTimeStep();
  
      // NOTE ARS - FIXME before the output so the values can be saved.
      // Monitoring tasks must be scheduled last!!
      for (int i = 0; i < m_current_gridP->numLevels(); i++) {
        m_scheduler->scheduleTaskMonitoring(m_current_gridP->getLevel(i));
      }

      // Output tasks
      const bool recompile = true;
      m_output->finalizeTimestep( m_app->getTimeStep(),
				                          m_app->getSimTime(),
				                          m_app->getDelT(),
				                          m_current_gridP,
				                          m_scheduler,
				                          recompile) ;

      m_output->sched_allOutputTasks( m_app->getDelT(),
                                      m_current_gridP,
                                      m_scheduler, recompile );

      // Initialize the system var (time step and simulation time).
      // Must be done after the output.
      m_app->scheduleInitializeSystemVars( m_current_gridP, m_lb->getPerProcessorPatchSet(m_current_gridP), m_scheduler );

      taskGraphTimer.reset( true );
      m_scheduler->compile();
      taskGraphTimer.stop();

      m_runtime_stats[ CompilationTime ] += taskGraphTimer().seconds();

      proc0cout << "Done with taskgraph compile (" << taskGraphTimer().seconds() << " seconds)\n";

      // No scrubbing for initial step
      m_scheduler->get_dw(1)->setScrubbing(DataWarehouse::ScrubNone);
      m_scheduler->execute();

      needNewLevel = ( m_regridder && m_regridder->isAdaptive() &&
                       m_current_gridP->numLevels() < m_regridder->maxLevels() &&
                       doRegridding(true) );

      if ( needNewLevel ) {
        m_scheduler->initialize( 1, 1 );
        m_scheduler->advanceDataWarehouse( m_current_gridP, true );
      }

    } while ( needNewLevel );

    m_output->writeto_xml_files( 0, 0, 0, m_current_gridP );
  }

} // end doInitialTimeStep()

//______________________________________________________________________
//
void
AMRSimulationController::executeTimeStep( int totalFine )
{
  MALLOC_TRACE_TAG_SCOPE("AMRSimulationController::executeTimeStep()");

  // If the time step needs to be restarted, this loop will execute multiple times.
  bool success = false;

  int tg_index = m_app->computeTaskGraphIndex();

  bool restartable = m_app->restartableTimeSteps();
  m_scheduler->setRestartable(restartable);

  // Execute at least once.
  while (!success) {
    m_app->adjustDelTForAllLevels( m_scheduler, m_current_gridP, totalFine );

    // Standard data warehouse scrubbing.
    if (m_scrub_datawarehouse && m_lb->getNthRank() == 1) {
      if (restartable) {
        m_scheduler->get_dw(0)->setScrubbing(DataWarehouse::ScrubNonPermanent);
      }
      else {
        m_scheduler->get_dw(0)->setScrubbing(DataWarehouse::ScrubComplete);
      }
      // The other data warehouse as well as those for other levels.
      for (int i = 1; i <= totalFine; ++i) {
        m_scheduler->get_dw(i)->setScrubbing(DataWarehouse::ScrubNonPermanent);
      }
    }
    // If not scrubbing or getNthRank requires the variables after
    // they would have been scrubbed so turn off all scrubbing.
    else {  //if( !m_scrub_datawarehouse || m_lb->getNthRank() > 1 )
      for (int i = 0; i <= totalFine; ++i) {
        m_scheduler->get_dw(i)->setScrubbing(DataWarehouse::ScrubNone);
      }
    }

    if (m_do_multi_taskgraphing) {
      subCycleExecute(0, totalFine, 0, true);
    }
    // TG index set by component that requested temporal scheduling
    //   (multiple primary task graphs) this is passed to
    //   scheduler->execute(), default index is 0
    else {
      int iteration = (m_last_recompile_timeStep == m_app->getTimeStep()) ? 0 : 1;
      m_scheduler->execute(tg_index, iteration);
    }

    //  If time step has been restarted adjust the delta T and restart.
    if (m_scheduler->get_dw(totalFine)->timestepRestarted()) {
      ASSERT(restartable);

      for (int i = 1; i <= totalFine; ++i) {
        m_scheduler->replaceDataWarehouse(i, m_current_gridP);
      }

      // Recompute the delta T.
      m_app->recomputeTimeStep();

      // Re-evaluate the outputting and checkpointing.
      m_output->reevaluate_OutputCheckPointTimestep(m_app->getSimTime(), m_app->getDelT());

      success = false;
    }
    else {
      if (m_scheduler->get_dw(1)->timestepAborted()) {
        throw InternalError("Execution aborted, cannot restart time step\n", __FILE__, __LINE__);
      }

      success = true;
    }
  } 
}

//______________________________________________________________________
//
bool
AMRSimulationController::doRegridding( bool initialTimeStep )
{
  MALLOC_TRACE_TAG_SCOPE("AMRSimulationController::doRegridding()");

  Timers::Simple regriddingTimer;         // Regridding time

  regriddingTimer.start();

  bool retVal = false;

  if( !initialTimeStep ) {
    proc0cout << "______________________________________________________________________\n";
  }
    
  GridP oldGrid = m_current_gridP;
  m_current_gridP = m_regridder->regrid(oldGrid.get_rep());
  
  if(dbg_barrier.active()) {
    m_barrier_timer.reset( true );
    Uintah::MPI::Barrier(d_myworld->getComm());
    m_barrier_times[0] += m_barrier_timer().seconds();
  }
  
  regriddingTimer.stop();

  m_runtime_stats[ RegriddingTime ] += regriddingTimer().seconds();
  
  m_app->setRegridTimeStep(false);

  int lbstate = initialTimeStep ? LoadBalancerPort::init : LoadBalancerPort::regrid;

  if (m_current_gridP != oldGrid) {
    m_app->setRegridTimeStep(true);
     
    m_lb->possiblyDynamicallyReallocate(m_current_gridP, lbstate);

    if(dbg_barrier.active()) {
      m_barrier_timer.reset( true );
      Uintah::MPI::Barrier(d_myworld->getComm());
      m_barrier_times[1] += m_barrier_timer().seconds();
    }
    
    m_current_gridP->assignBCS( m_grid_ps, m_lb );
    m_current_gridP->performConsistencyCheck();

    //__________________________________
    //  output regridding stats
    if (d_myworld->myRank() == 0) {

      std::cout << "  REGRIDDING:";

      // amrout << "---------- OLD GRID ----------" << std::endl << *(oldGrid.get_rep());
      for (int i = 0; i < m_current_gridP->numLevels(); i++) {
        std::cout << " Level " << i << " has " << m_current_gridP->getLevel(i)->numPatches() << " patches...";
      }
      std::cout << "\n";

      if (amrout.active()) {
        amrout << "---------- NEW GRID ----------\n" << "Grid has " << m_current_gridP->numLevels() << " level(s)\n";

        for( int levelIndex = 0; levelIndex < m_current_gridP->numLevels(); levelIndex++ ) {
          LevelP level = m_current_gridP->getLevel( levelIndex );

          amrout << "  Level " << level->getID() << ", indx: " << level->getIndex() << " has " << level->numPatches() << " patch(es)\n";

          for( Level::patch_iterator patchIter = level->patchesBegin(); patchIter < level->patchesEnd(); patchIter++ ) {
            const Patch* patch = *patchIter;
            amrout << "(Patch " << patch->getID() << " proc " << m_lb->getPatchwiseProcessorAssignment(patch) << ": box="
                   << patch->getExtraBox() << ", lowIndex=" << patch->getExtraCellLowIndex() << ", highIndex="
                   << patch->getExtraCellHighIndex() << ")\n";
          }
        }
      }
    }  // rank 0

    Timers::Simple schedulerTimer;

    if( !initialTimeStep ) {
      schedulerTimer.start();
      m_scheduler->scheduleAndDoDataCopy( m_current_gridP, m_app );
      schedulerTimer.stop();
    }
    
    proc0cout << "done regridding ("
              << regriddingTimer().seconds() + schedulerTimer().seconds()
              << " seconds, "
              << "regridding took " << regriddingTimer().seconds()
              << " seconds";
      
    if (!initialTimeStep) {
      proc0cout << ", scheduling and copying took " << schedulerTimer().seconds() << " seconds)\n";
    }
    else {
      proc0cout << ")\n";
    }
    
    retVal = true;
  }  // grid != oldGrid

  if (!initialTimeStep)
    proc0cout << "______________________________________________________________________\n";
  
  return retVal;
}

//______________________________________________________________________
//
bool
AMRSimulationController::needRecompile()
{
  MALLOC_TRACE_TAG_SCOPE("AMRSimulationController::needRecompile()");

  // Currently, m_output, m_sim, m_lb, m_regridder can request a recompile
  bool recompile =
       m_output->needRecompile(m_app->getSimTime(), m_app->getDelT(), m_current_gridP) ||
       m_app->needRecompile(m_app->getSimTime(), m_app->getDelT(), m_current_gridP)    ||
       m_lb->needRecompile(m_app->getSimTime(), m_app->getDelT(), m_current_gridP)     ||
       m_recompile_taskgraph;
  
  if (m_regridder) {
    recompile |= m_regridder->needRecompile(m_app->getSimTime(), m_app->getDelT(), m_current_gridP);
  }

#ifdef HAVE_VISIT
  // Check all of the component variables that might require the task
  // graph to be recompiled.

  // ARS - Should this check be on the component level?
  for( unsigned int i=0; i<m_app->getUPSVars().size(); ++i )
  {
    ApplicationInterface::interactiveVar &var = m_app->getUPSVars()[i];

    if( var.modified && var.recompile )
    {
      recompile = true;
    }
  }
#endif
  
  return recompile;
}

//______________________________________________________________________
//
void
AMRSimulationController::recompile( int totalFine )
{
  MALLOC_TRACE_TAG_SCOPE("AMRSimulationController::Recompile()");

  Timers::Simple taskGraphTimer;

  taskGraphTimer.start();

  proc0cout << "Compiling taskgraph...\n";

  m_last_recompile_timeStep = m_app->getTimeStep();

  m_scheduler->initialize( 1, totalFine );
  m_scheduler->fillDataWarehouses( m_current_gridP );

  // Set up new DWs, DW mappings.
  m_scheduler->clearMappings();
  m_scheduler->mapDataWarehouse( Task::OldDW, 0 );
  m_scheduler->mapDataWarehouse( Task::NewDW, totalFine );
  m_scheduler->mapDataWarehouse( Task::CoarseOldDW, 0 );
  m_scheduler->mapDataWarehouse( Task::CoarseNewDW, totalFine );
  
  int my_rank = d_myworld->myRank();
  if( m_do_multi_taskgraphing ) {
    for (int i = 0; i < m_current_gridP->numLevels(); i++) {
      // taskgraphs 0-numlevels-1
      if( i > 0 ) {
        // we have the first one already
        m_scheduler->addTaskGraph(Scheduler::NormalTaskGraph);
      }
      dbg << my_rank << "   Creating level " << i << " tg\n";

      m_app->scheduleTimeAdvance(m_current_gridP->getLevel(i), m_scheduler);
    }

    for (int i = 0; i < m_current_gridP->numLevels(); i++) {
      if (m_app->isAMR() && m_current_gridP->numLevels() > 1) {
        dbg << my_rank << "   Doing Intermediate TG level " << i << " tg\n";
        // taskgraphs numlevels-2*numlevels-1
        m_scheduler->addTaskGraph(Scheduler::IntermediateTaskGraph);
      }

      // schedule a coarsen from the finest level to this level
      for( int j = m_current_gridP->numLevels() - 2; j >= i; j-- ) {
        dbg << my_rank << "   schedule coarsen on level " << j << "\n";
        m_app->scheduleCoarsen(m_current_gridP->getLevel(j), m_scheduler);
      }

      m_app->scheduleFinalizeTimestep(m_current_gridP->getLevel(i), m_scheduler);

      // schedule a refineInterface from this level to the finest level
      for (int j = i; j < m_current_gridP->numLevels(); j++) {
        if (j != 0) {
          dbg << my_rank << "   schedule RI on level " << j << " for tg "
              << i << " coarseold " << (j == i) << " coarsenew " << true << "\n";
          m_app->scheduleRefineInterface( m_current_gridP->getLevel(j), m_scheduler, j == i, true );
        }
      }
    }
    // for the final error estimate and stable timestep tasks
    m_scheduler->addTaskGraph(Scheduler::IntermediateTaskGraph);
  }
  else /* if ( !m_do_multi_taskgraphing ) */ {
    subCycleCompile( 0, totalFine, 0, 0 );

    m_scheduler->clearMappings();
    m_scheduler->mapDataWarehouse( Task::OldDW, 0 );
    m_scheduler->mapDataWarehouse( Task::NewDW, totalFine );
  }

  // If regridding schedule error estimates
  for( int i = m_current_gridP->numLevels() - 1; i >= 0; i-- ) {
    dbg << my_rank << "   final TG " << i << "\n";

    if( m_regridder ) {
      m_regridder->scheduleInitializeErrorEstimate(m_current_gridP->getLevel(i));
      m_app->scheduleErrorEstimate(m_current_gridP->getLevel(i), m_scheduler);

      if (i < m_regridder->maxLevels() - 1) { // we don't use error estimates if we don't make another level, so don't dilate
        m_regridder->scheduleDilation(m_current_gridP->getLevel(i),  m_app->isLockstepAMR());
      }
    }
  }

  // After all tasks are done schedule the on-the-fly and other analysis.
  for (int i = 0; i < m_current_gridP->numLevels(); i++) {
    m_app->scheduleAnalysis(m_current_gridP->getLevel(i), m_scheduler);
  }

  // Compute the next time step.
  scheduleComputeStableTimeStep();
  
  // NOTE ARS - FIXME before the output so the values can be saved.
  // Monitoring tasks must be scheduled last!!
  for (int i = 0; i < m_current_gridP->numLevels(); i++) {
    m_scheduler->scheduleTaskMonitoring(m_current_gridP->getLevel(i));
  }
  
  // Output tasks
  m_output->finalizeTimestep( m_app->getTimeStep(),
			                        m_app->getSimTime(),
			                        m_app->getDelT(),
			                        m_current_gridP,
			                        m_scheduler,
			                        true );

  m_output->sched_allOutputTasks(m_app->getDelT(), m_current_gridP, m_scheduler, true);

  // Update the system var (time step and simulation time). Must be
  // done after the output.
  m_app->scheduleUpdateSystemVars( m_current_gridP,
                                   m_lb->getPerProcessorPatchSet(m_current_gridP),
				                           m_scheduler );

  m_scheduler->compile();

  taskGraphTimer.stop();

  m_runtime_stats[ CompilationTime ] += taskGraphTimer().seconds();

  proc0cout << "Done with taskgraph re-compile (" << taskGraphTimer().seconds() << " seconds)\n";
} // end recompile()

//______________________________________________________________________
//
void
AMRSimulationController::subCycleCompile( int startDW,
                                          int dwStride,
                                          int numLevel,
                                          int step )
{
  MALLOC_TRACE_TAG_SCOPE("AMRSimulationController::subCycleCompile()");
  //amrout << "Start AMRSimulationController::subCycleCompile, level=" << numLevel << '\n';
  // We are on (the fine) level numLevel
  LevelP fineLevel = m_current_gridP->getLevel(numLevel);
  LevelP coarseLevel;
  int coarseStartDW;
  int coarseDWStride;
  int numCoarseSteps; // how many steps between this level and the coarser
  int numFineSteps;   // how many steps between this level and the finer
  if (numLevel > 0) {
    numCoarseSteps = m_app->isLockstepAMR() ? 1 : fineLevel->getRefinementRatioMaxDim();
    coarseLevel = m_current_gridP->getLevel(numLevel-1);
    coarseDWStride = dwStride * numCoarseSteps;
    coarseStartDW = (startDW/coarseDWStride)*coarseDWStride;
  }
  else {
    coarseDWStride = dwStride;
    coarseStartDW = startDW;
    numCoarseSteps = 0;
  }
  
  ASSERT(dwStride > 0 && numLevel < m_current_gridP->numLevels())
  m_scheduler->clearMappings();
  m_scheduler->mapDataWarehouse(Task::OldDW, startDW);
  m_scheduler->mapDataWarehouse(Task::NewDW, startDW+dwStride);
  m_scheduler->mapDataWarehouse(Task::CoarseOldDW, coarseStartDW);
  m_scheduler->mapDataWarehouse(Task::CoarseNewDW, coarseStartDW+coarseDWStride);

  m_app->scheduleTimeAdvance(fineLevel, m_scheduler);

  if( m_app->isAMR() ) {
    if( numLevel + 1 < m_current_gridP->numLevels() ) {
      numFineSteps = m_app->isLockstepAMR() ? 1 : fineLevel->getFinerLevel()->getRefinementRatioMaxDim();
      int newStride = dwStride / numFineSteps;

      for( int substep = 0; substep < numFineSteps; substep++ ) {
        subCycleCompile(startDW + substep * newStride, newStride, numLevel + 1, substep);
      }

      // Coarsen and then refine_CFI at the end of the W-cycle
      m_scheduler->clearMappings();
      m_scheduler->mapDataWarehouse( Task::OldDW, 0 );
      m_scheduler->mapDataWarehouse( Task::NewDW, startDW + dwStride );
      m_scheduler->mapDataWarehouse( Task::CoarseOldDW, startDW );
      m_scheduler->mapDataWarehouse( Task::CoarseNewDW, startDW + dwStride );
      m_app->scheduleCoarsen( fineLevel, m_scheduler );
    }
  }

  m_scheduler->clearMappings();
  m_scheduler->mapDataWarehouse(Task::OldDW, startDW);
  m_scheduler->mapDataWarehouse(Task::NewDW, startDW+dwStride);
  m_scheduler->mapDataWarehouse(Task::CoarseOldDW, coarseStartDW);
  m_scheduler->mapDataWarehouse(Task::CoarseNewDW, coarseStartDW+coarseDWStride);
  m_app->scheduleFinalizeTimestep(fineLevel, m_scheduler);

  // do refineInterface after the freshest data we can get; after the
  // finer level's coarsen completes do all the levels at this point
  // in time as well, so all the coarsens go in order, and then the
  // refineInterfaces
  if (m_app->isAMR() && (step < numCoarseSteps -1 || numLevel == 0)) {
    
    for (int i = fineLevel->getIndex(); i < fineLevel->getGrid()->numLevels(); i++) {
      if (i == 0) {
        continue;
      }
      if (i == fineLevel->getIndex() && numLevel != 0) {
        m_scheduler->mapDataWarehouse(Task::CoarseOldDW, coarseStartDW);
        m_scheduler->mapDataWarehouse(Task::CoarseNewDW, coarseStartDW+coarseDWStride);
        m_app->scheduleRefineInterface(fineLevel, m_scheduler, true, true);
      }
      else {
        // look in the NewDW all the way down
        m_scheduler->mapDataWarehouse(Task::CoarseOldDW, 0);
        m_scheduler->mapDataWarehouse(Task::CoarseNewDW, startDW+dwStride);
        m_app->scheduleRefineInterface(fineLevel->getGrid()->getLevel(i), m_scheduler, false, true);
      }
    }
  }
}

//______________________________________________________________________
//
void
AMRSimulationController::subCycleExecute( int startDW,
                                          int dwStride,
                                          int levelNum,
                                          bool rootCycle )
{
  MALLOC_TRACE_TAG_SCOPE("AMRSimulationController::subCycleExecutue()");

  // there are 2n+1 taskgraphs, n for the basic timestep, n for intermediate 
  // timestep work, and 1 for the errorEstimate and stableTimestep, where n
  // is the number of levels.
  
  // amrout << "Start AMRSimulationController::subCycleExecute, level=" << numLevel << '\n';
  // We are on (the fine) level numLevel
  int numSteps;
  if (levelNum == 0 || m_app->isLockstepAMR()) {
    numSteps = 1;
  }
  else {
    numSteps = m_current_gridP->getLevel(levelNum)->getRefinementRatioMaxDim();
  }
  
  int newDWStride = dwStride/numSteps;

  DataWarehouse::ScrubMode oldScrubbing = (/*m_lb->isDynamic() ||*/ m_app->restartableTimeSteps()) ?
    DataWarehouse::ScrubNonPermanent : DataWarehouse::ScrubComplete;

  int curDW = startDW;
  for( int step = 0; step < numSteps; step++ ) {
  
    if( step > 0 ) {
      curDW += newDWStride; // can't increment at the end, or the FINAL tg for L0 will use the wrong DWs
    }

    m_scheduler->clearMappings();
    m_scheduler->mapDataWarehouse(Task::OldDW, curDW);
    m_scheduler->mapDataWarehouse(Task::NewDW, curDW+newDWStride);
    m_scheduler->mapDataWarehouse(Task::CoarseOldDW, startDW);
    m_scheduler->mapDataWarehouse(Task::CoarseNewDW, startDW+dwStride);

    // we really only need to pass in whether the current DW is mapped to 0 or not
    // TODO - fix inter-Taskgraph scrubbing
    m_scheduler->get_dw(curDW)->setScrubbing(oldScrubbing); // OldDW
    m_scheduler->get_dw(curDW+newDWStride)->setScrubbing(DataWarehouse::ScrubNonPermanent); // NewDW
    m_scheduler->get_dw(startDW)->setScrubbing(oldScrubbing); // CoarseOldDW
    m_scheduler->get_dw(startDW+dwStride)->setScrubbing(DataWarehouse::ScrubNonPermanent); // CoarseNewDW
    
    // we need to unfinalize because execute finalizes all new DWs,
    // and we need to write into them still (even if we finalized only
    // the NewDW in execute, we will still need to write into that DW)
    m_scheduler->get_dw(curDW+newDWStride)->unfinalize();

    // iteration only matters if it's zero or greater than 0
    int iteration = curDW + (m_last_recompile_timeStep == m_app->getTimeStep() ? 0 : 1);
    
    if (dbg.active()) {
      dbg << d_myworld->myRank() << "   Executing TG on level " << levelNum << " with old DW " << curDW << "="
          << m_scheduler->get_dw(curDW)->getID() << " and new " << curDW + newDWStride << "="
          << m_scheduler->get_dw(curDW + newDWStride)->getID() << "CO-DW: " << startDW << " CNDW " << startDW + dwStride
          << " on iteration " << iteration << "\n";
    }
    
    m_scheduler->execute(levelNum, iteration);
    
    if( levelNum + 1 < m_current_gridP->numLevels() ) {
      ASSERT(newDWStride > 0);
      subCycleExecute(curDW, newDWStride, levelNum + 1, false);
    }
 
    if (m_app->isAMR() && m_current_gridP->numLevels() > 1 &&
        (step < numSteps-1 || levelNum == 0)) {
      // Since the execute of the intermediate is time-based,
      // execute the intermediate TG relevant to this level, if we are in the 
      // middle of the subcycle or at the end of level 0.
      // the end of the cycle will be taken care of by the parent level sybcycle
      m_scheduler->clearMappings();
      m_scheduler->mapDataWarehouse(Task::OldDW, curDW);
      m_scheduler->mapDataWarehouse(Task::NewDW, curDW+newDWStride);
      m_scheduler->mapDataWarehouse(Task::CoarseOldDW, startDW);
      m_scheduler->mapDataWarehouse(Task::CoarseNewDW, startDW+dwStride);

      m_scheduler->get_dw(curDW)->setScrubbing(oldScrubbing); // OldDW
      m_scheduler->get_dw(curDW+newDWStride)->setScrubbing(DataWarehouse::ScrubNonPermanent); // NewDW
      m_scheduler->get_dw(startDW)->setScrubbing(oldScrubbing); // CoarseOldDW
      m_scheduler->get_dw(startDW+dwStride)->setScrubbing(DataWarehouse::ScrubNonPermanent); // CoarseNewDW

      if (dbg.active()) {
        dbg << d_myworld->myRank() << "   Executing INT TG on level " << levelNum << " with old DW " << curDW << "="
            << m_scheduler->get_dw(curDW)->getID() << " and new " << curDW + newDWStride << "="
            << m_scheduler->get_dw(curDW + newDWStride)->getID() << " CO-DW: " << startDW << " CNDW " << startDW + dwStride
            << " on iteration " << iteration << "\n";
      }
      m_scheduler->get_dw(curDW+newDWStride)->unfinalize();
      m_scheduler->execute(levelNum+m_current_gridP->numLevels(), iteration);
    }
    
    if (curDW % dwStride != 0) {
      //the currentDW(old datawarehouse) should no longer be needed - in the case of NonPermanent OldDW scrubbing
      m_scheduler->get_dw(curDW)->clear();
    }
  }

  if( levelNum == 0 ) {
    // execute the final TG
    if (dbg.active()) {
      dbg << d_myworld->myRank() << "   Executing Final TG on level " << levelNum << " with old DW " << curDW << " = "
          << m_scheduler->get_dw(curDW)->getID() << " and new " << curDW + newDWStride << " = "
          << m_scheduler->get_dw(curDW + newDWStride)->getID() << std::endl;
    }

    m_scheduler->get_dw(curDW + newDWStride)->unfinalize();
    m_scheduler->execute(m_scheduler->getNumTaskGraphs() - 1, 1);
  }
} // end subCycleExecute()

//______________________________________________________________________
//
void
AMRSimulationController::scheduleComputeStableTimeStep()
{
  MALLOC_TRACE_TAG_SCOPE("AMRSimulationController::scheduleComputeStableTimeStep()");

  // Schedule the application to compute the next time step on a per
  // patch basis.
  for (int i = 0; i < m_current_gridP->numLevels(); i++) {
    m_app->scheduleComputeStableTimeStep(m_current_gridP->getLevel(i), m_scheduler);
  }

  // Schedule the reduction of the time step and other variables on a
  // per patch basis to a per rank basis.
  m_app->scheduleReduceSystemVars( m_current_gridP,
				                           m_lb->getPerProcessorPatchSet(m_current_gridP),
				                           m_scheduler);
}
