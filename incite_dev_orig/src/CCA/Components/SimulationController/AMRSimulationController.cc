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

#include <CCA/Components/SimulationController/AMRSimulationController.h>
#include <CCA/Components/ReduceUda/UdaReducer.h>
#include <CCA/Components/Regridder/PerPatchVars.h>
#include <CCA/Components/Schedulers/MPIScheduler.h>
#include <CCA/Ports/DataWarehouse.h>
#include <CCA/Ports/LoadBalancerPort.h>
#include <CCA/Ports/Output.h>
#include <CCA/Ports/ProblemSpecInterface.h>
#include <CCA/Ports/Regridder.h>
#include <CCA/Ports/Scheduler.h>
#include <CCA/Ports/SimulationInterface.h>

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
#include <Core/Util/Time.h>


#ifdef HAVE_VISIT
#  include <VisIt/libsim/visit_libsim.h>
#endif


#ifdef HAVE_CUDA
//#  include <CCA/Components/Schedulers/GPUUtilities.h>
#  include <CCA/Components/Schedulers/GPUGridVariableInfo.h>
#endif


#include <sci_defs/malloc_defs.h>
#include <sci_defs/gperftools_defs.h>
#include <sci_defs/visit_defs.h>


#include <iostream>
#include <iomanip>

using namespace Uintah;


namespace {

DebugStream amrout("AMR", false);
DebugStream dbg("AMRSimulationController", false);
DebugStream dbg_barrier("MPIBarriers",false); DebugStream dbg_dwmem("LogDWMemory",false);
DebugStream gprofile("CPUProfiler",false);
DebugStream gheapprofile("HeapProfiler",false);
DebugStream gheapchecker("HeapChecker",false);

double barrier_times[5] = {0};

}


//______________________________________________________________________
//
AMRSimulationController::AMRSimulationController( const ProcessorGroup * myworld
                                                ,       bool                   doAMR
                                                ,       ProblemSpecP           pspec
                                                )
  : SimulationController( myworld, doAMR, pspec )
{
  scrubDataWarehouse = true;
}


//______________________________________________________________________
//
void
AMRSimulationController::run()
{
  bool   first = true;
  double time;

  // If VisIt has been included into the build, initialize the lib sim
  // so that a user can connect to the simulation via VisIt.
#ifdef HAVE_VISIT
  visit_simulation_data visitSimData;

  if( m_shared_state->getVisIt() )
  {
    visitSimData.simController = this;
    visitSimData.runMode = m_shared_state->getVisIt();

    // Running with VisIt so add in the variables that the user can
    // modify.
    // variable 1 - Must start with the component name and have NO
    // spaces in the var name.
    SimulationState::interactiveVar var;
    var.name     = "Scrub-Data-Warehouse";
    var.type     = Uintah::TypeDescription::bool_type;
    var.value    = (void *) &(scrubDataWarehouse);
    var.modifiable = true;
    var.recompile  = false;
    var.modified   = false;
    m_shared_state->d_stateVars.push_back( var );

    m_shared_state->d_debugStreams.push_back( &amrout );
    m_shared_state->d_debugStreams.push_back( &dbg );
    m_shared_state->d_debugStreams.push_back( &dbg_barrier );
    m_shared_state->d_debugStreams.push_back( &dbg_dwmem );
    m_shared_state->d_debugStreams.push_back( &gprofile );
    m_shared_state->d_debugStreams.push_back( &gheapprofile );
    m_shared_state->d_debugStreams.push_back( &gheapchecker );

    visit_InitLibSim( &visitSimData );
  }
#endif
    
  bool log_dw_mem = dbg_dwmem.active();

  MALLOC_TRACE_TAG_SCOPE("AMRSimulationController::run()");

#ifdef USE_GPERFTOOLS
  if (gprofile.active()){
    char gprofname[512];
    sprintf(gprofname, "cpuprof-rank%d", d_myworld->myrank());
    ProfilerStart(gprofname);
  }
  if (gheapprofile.active()){
    char gheapprofname[512];
    sprintf(gheapprofname, "heapprof-rank%d", d_myworld->myrank());
    HeapProfilerStart(gheapprofname);
  }

  HeapLeakChecker * heap_checker=nullptr;
  if (gheapchecker.active()){
    if (!gheapprofile.active()){
      char gheapchkname[512];
      sprintf(gheapchkname, "heapchk-rank%d", d_myworld->myrank());
      heap_checker= new HeapLeakChecker(gheapchkname);
    } else {
      cout<< "HEAPCHECKER: Cannot start with heapprofiler" <<endl;
    }
  }
#endif

  // Sets up sharedState, timeinfo, output, scheduler, lb.
  preGridSetup();
  
  // Create grid:
  GridP currentGrid = gridSetup();

#ifdef HAVE_CUDA
  GpuUtilities::assignPatchesToGpus(currentGrid);
#endif

  // Initialize the scheduler.
  m_scheduler->setInitTimestep( true );
  m_scheduler->setRestartInitTimestep( m_restarting );
  m_scheduler->initialize( 1, 1 );
  m_scheduler->advanceDataWarehouse( currentGrid, true );

  // Set up sim, regridder, and finalize sharedState also reload from
  // the DataArchive on restart, this call will set the time variable.
  postGridSetup( currentGrid, time );

#ifdef HAVE_CUDA
  GpuUtilities::assignPatchesToGpus(currentGrid);
#endif

  // Itiitalize the wall times. All clocking is relative. That is,
  // each call to calcuate the wall time for a particlar block of code
  // will be accumulated based on the call (total, exec, or in-situ).
  initWallTimes();

  // Set the simulaiton start time. It should be set at the same time
  // initWallTimes is called so the real time measures are sync'ed.
  setStartSimTime( time );

  //__________________________________
  //  reduceUda
  if( m_reduce_uda ) {
    Dir fromDir( m_from_dir );
    m_output->reduceUdaSetup( fromDir );
    m_time_info->delt_factor = 1;
    m_time_info->delt_min    = 0;
    m_time_info->delt_max    = 1e99;
    m_time_info->initTime    = static_cast<UdaReducer*>(m_sim)->getInitialTime();
    m_time_info->maxTime     = static_cast<UdaReducer*>(m_sim)->getMaxTime();
    m_time_info->max_delt_increase = 1e99;
    m_time_info->max_initial_delt  = 1e99;
  }

  // Setup, compile, and run the taskgraph for the initialization timestep
  doInitialTimestep( currentGrid, time );

  // 
#ifndef DISABLE_SCI_MALLOC
  AllocatorSetDefaultTagLineNumber( d_sharedState->getCurrentTopLevelTimeStep() );
#endif

  // Setup for PIDX
  static int  requested_nth_output_proc = -1;
  static bool need_to_recompile = false;
  static bool put_back          = false;

#ifdef HAVE_PIDX
  if( d_output && d_output->savingAsPIDX() ) {
    if( requested_nth_output_proc == -1 ) {
      requested_nth_output_proc = d_lb->getNthRank();

      if( requested_nth_output_proc > 1 ) {
        proc0cout << "Input file requests output to be saved by every " << requested_nth_output_proc << "th processor.\n"
                  << "  - However, setting output to every process until we hit a checkpoint\n";
        d_lb->setNthRank( 1 );
        d_lb->possiblyDynamicallyReallocate( currentGrid, LoadBalancerPort::regrid );
        d_output->setSaveAsPIDX();
      }
    }
  }
#endif

  // Get and reduce the performance run time stats for 0th time step
  // which is for either the initialization or the setup for a restart.
  getMemoryStats(m_shared_state->getCurrentTopLevelTimeStep());
  getPAPIStats();
  m_shared_state->d_runTimeStats.reduce(m_regridder && m_regridder->useDynamicDilation(), d_myworld);
  // Reduce the mpi run time stats.
  MPIScheduler *mpiScheduler = dynamic_cast<MPIScheduler*>(m_scheduler.get_rep());
  if (mpiScheduler) {
    mpiScheduler->mpi_info_.reduce(m_regridder && m_regridder->useDynamicDilation(), d_myworld);
  }

  // Print MPI statistics
  m_scheduler->printMPIStats();

  // Done with all the initialization.
  m_scheduler->setInitTimestep(false);

  m_lb->resetCostForecaster();

  // Retrieve the next delta T and adjust it based on timeinfo parameters.
  DataWarehouse* newDW = m_scheduler->getLastDW();
  delt_vartype delt_var;
  newDW->get( delt_var, m_shared_state->get_delt_label() );
  double delt = delt_var;
  adjustDelT( delt, m_prev_delt, time );

  newDW->override( delt_vartype(delt), m_shared_state->get_delt_label() );

  // Before printing stats calculate the execution time.
  calcExecWallTime();
  
  // Print the stats for the initalization or restart.
  printSimulationStats( m_shared_state->getCurrentTopLevelTimeStep(), delt, m_prev_delt, time, true );

  // If VisIt has been included into the build, check the lib sim
  // state to see if there is a connection and if so check to see if
  // anything needs to be done.
#ifdef HAVE_VISIT
  if( m_shared_state->getVisIt() )
  {
    // Before the in-situ update the total time.
    calcTotalWallTime();
  
    // Update all of the simulation grid and time dependent variables.
    visit_UpdateSimData( &visitSimData, currentGrid,
			 time, d_prev_delt, delt,
			 getTotalWallTime(), getTotalExecWallTime(),
			 getExecWallTime(), getExpMovingAverage(),
			 getInSituWallTime(), isLast( time ) );
    
    // Check the state - if the return value is true the user issued a
    // termination.
    if( visit_CheckState( &visitSimData ) )
      exit(0);
    
    // The user may have adjusted delt so get it from the data
    // warehouse. If not then this call is a no-op.
    newDW->get( delt_var, m_shared_state->get_delt_label() );
    delt = delt_var;

    // Report on the modiied variables. 
    for (std::map<std::string,std::string>::iterator
	   it = visitSimData.modifiedVars.begin();
	 it != visitSimData.modifiedVars.end();
	 ++it)
      proc0cout << "Visit libsim - At time step "
		<< m_shared_state->getCurrentTopLevelTimeStep() << " "
		<< "the user modified the variable " << it->first << " "
		<< "to be " << it->second << ". "
		<< std::endl;

    // TODO - Put this information into the NEXT time step xml.

    
    calcInSituWallTime();
  }
#endif

  // Reset the runtime performance stats
  m_shared_state->resetStats();
  // Reset memory use tracking variable
  m_scheduler->resetMaxMemValue();
  
  // Before the next step update the total time.
  calcTotalWallTime();


  // The main time loop. Here is where the specified problem is
  // actually getting solved.
  while( !isLast( time ) ) {

    // Put the current time into the shared state so other components
    // can access it.  Also increment (by one) the current time step
    // number so components can tell what timestep they are on.
    m_shared_state->setElapsedTime( time );
    m_shared_state->incrementCurrentTopLevelTimeStep();

#ifdef USE_GPERFTOOLS
    if (gheapprofile.active()){
      char heapename[512];
      sprintf(heapename, "Timestep %d", timestep);
      HeapProfilerDump(heapename);
    }
#endif
     
    MALLOC_TRACE_TAG_SCOPE( "AMRSimulationController::run()::control loop" );
    if( dbg_barrier.active() ) {
      for(int i = 0; i < 5; i++ ) {
        barrier_times[i] = 0;
      }
    }

#ifdef HAVE_PIDX
    bool checkpointing = false;
    if( m_output && m_output->savingAsPIDX() ) {

      int currentTimeStep = m_shared_state->getCurrentTopLevelTimeStep();

      // When using Wall Clock Time for checkpoints, we need to have
      // rank 0 determine this time and then send it to all other
      // ranks.
      int currsecs = -1;
      if( m_output->getCheckpointWalltimeInterval() > 0 ) {
        // If checkpointing based on wall clock time, then have
        // process 0 determine the current time and share it will
        // everyone else.
        if( Parallel::getMPIRank() == 0 ) {
          currsecs = (int)Time::currentSeconds();
        }
        Uintah::MPI::Bcast( &currsecs, 1, MPI_INT, 0, d_myworld->getComm() );
      }

      if( ( m_output->getCheckpointTimestepInterval() > 0 &&
              currentTimeStep == m_output->getNextCheckpointTimestep() ) ||
          ( m_output->getCheckpointInterval() > 0 &&
              ( time + delt ) >= m_output->getNextCheckpointTime() ) ||
          ( m_output->getCheckpointWalltimeInterval() > 0 &&
              ( currsecs >= m_output->getNextCheckpointWalltime() ) ) ) {

        checkpointing = true;

        if( requested_nth_output_proc > 1 ) {
          proc0cout << "This is a checkpoint timestep (" << currentTimeStep
          << ") - need to recompile with nth proc set to: "
          << requested_nth_output_proc << "\n";

          m_lb->setNthRank( requested_nth_output_proc );
          m_lb->possiblyDynamicallyReallocate( currentGrid,
              LoadBalancerPort::regrid );
          m_output->setSaveAsUDA();
          need_to_recompile = true;
        }
      }
      if( ( m_output->getOutputTimestepInterval() > 0 &&
              currentTimeStep == m_output->getNextOutputTimestep() ) ||
          ( m_output->getOutputInterval() > 0 &&
              ( time + delt ) >= m_output->getNextOutputTime() ) ) {

        proc0cout << "This is an output timestep: " << currentTimeStep << "\n";

        if( need_to_recompile ) {  // If this is also a checkpoint time step
          proc0cout << "   Postposing as this is also a checkpoint time step...\n";
          m_output->postponeNextOutputTimestep();
        }
      }
    }
#endif

    // Regridding
    if ( m_regridder ) {
      if( m_regridder->doRegridOnce() && m_regridder->isAdaptive() ) {
        proc0cout << "______________________________________________________________________\n";
        proc0cout << " Regridding once.\n";
        doRegridding( currentGrid, false );
        m_regridder->setAdaptivity( false );
        proc0cout << "______________________________________________________________________\n";
      }
      
      if( ( !first || !m_restarting ) &&  m_regridder->needsToReGrid( currentGrid ) ) {
        proc0cout << "______________________________________________________________________\n";
        proc0cout << " Need to regrid.\n";
        doRegridding( currentGrid, false );
        proc0cout << "______________________________________________________________________\n";
      }
    }

    // Compute number of dataWarehouses - multiplies by the time
    // refinement ratio for each level.
    int totalFine = 1;

    if (!m_shared_state->isLockstepAMR()) {
      for(int i=1; i<currentGrid->numLevels(); ++i) {
        totalFine *= currentGrid->getLevel(i)->getRefinementRatioMaxDim();
      }
    }
     
    if( log_dw_mem ) {
      // Remember, this isn't logged if DISABLE_SCI_MALLOC is set
      // (So usually in optimized mode this will not be run.)
      m_scheduler->logMemoryUse();
      std::ostringstream fn;
      fn << "alloc." << std::setw(5) << std::setfill('0') << d_myworld->myrank() << ".out";
      std::string filename(fn.str());
#ifndef DISABLE_SCI_MALLOC
      DumpAllocator(DefaultAllocator(), filename.c_str());
#endif
    }
     
    if( dbg_barrier.active() ) {
      double start_time = Time::currentSeconds();
      Uintah::MPI::Barrier( d_myworld->getComm() );
      barrier_times[2] += Time::currentSeconds() - start_time;
    }

    // This step is a hack but it is the only way to get a new grid
    // from UdaReducer and needs to be done before
    // advanceDataWarehouse is called.
    if ( m_reduce_uda ) {
      currentGrid = static_cast<UdaReducer*>( m_sim )->getGrid();
    }

    // After one step (either timestep or initialization) and
    // correction the delta finalize the old timestep, eg. finalize
    // and advance the Datawarehouse
    m_scheduler->advanceDataWarehouse( currentGrid );

#ifndef DISABLE_SCI_MALLOC
    AllocatorSetDefaultTagLineNumber( m_shared_state->getCurrentTopLevelTimeStep() );
#endif
    
    // Each component has their own init_delt specified.  On a switch
    // from one component to the next, delt needs to be adjusted to
    // the value specified in the input file.  To detect the switch of
    // components, compare the old_init_delt before the
    // needRecompile() to the new_init_delt after the needRecompile().
    double old_init_delt = m_time_info->max_initial_delt;
    double new_init_delt = 0.;

    bool nr = (needRecompile( time, delt, currentGrid ) ||
	       need_to_recompile || put_back);

    if( nr || first ) {

      // Recompile taskgraph, re-assign BCs, reset recompile flag.      
      if( nr ) {
        currentGrid->assignBCS( m_grid_ps, m_lb );
        currentGrid->performConsistencyCheck();
        m_shared_state->setRecompileTaskGraph( false );
      }

      if (put_back) {
        proc0cout << "This is the timestep following a checkpoint - " << "need to put the task graph back with a recompile - "
                  << "setting nth output to 1\n";
        m_lb->setNthRank(1);
        m_lb->possiblyDynamicallyReallocate(currentGrid, LoadBalancerPort::regrid);
        m_output->setSaveAsPIDX();
        put_back = false;
      }

      if( need_to_recompile ) {
        // Don't need to recompile on the next time step (as we are
        // about to do it on this one).  However, we will need to put
        // it back after this time step, so set put_back to true.
        need_to_recompile = false;
        put_back = true;
      }

      new_init_delt = m_time_info->max_initial_delt;
       
      if( new_init_delt != old_init_delt ) {
        // Writes to the DW in the next section below.
        delt = new_init_delt;
      }
      m_scheduler->setRestartInitTimestep( false );
      recompile( time, delt, currentGrid, totalFine );
    }
    else {
      if ( m_output ) {
        // This is not correct if we have switched to a different
        // component, since the delt will be wrong 
        m_output->finalizeTimestep( time, delt, currentGrid, m_scheduler, 0 );

        // ARS - THIS CALL DOES NOTHING BECAUSE THE LAST ARG IS 0
        // WHICH CAUSES THE METHOD TO IMMEIDATELY RETURN.
        m_output->sched_allOutputTasks( delt, currentGrid, m_scheduler, 0 );
      }
    }

    if( dbg_barrier.active() ) {
      double start_time = Time::currentSeconds();
      Uintah::MPI::Barrier( d_myworld->getComm() );
      barrier_times[3] += Time::currentSeconds() - start_time;
    }

    // Adjust the delt for each level and store it in all applicable dws.
    double delt_fine = delt;
    int    skip      = totalFine;

    for( int i = 0; i < currentGrid->numLevels(); ++i ) {
      const Level* level = currentGrid->getLevel(i).get_rep();
      
      if( m_do_amr && i != 0 && !m_shared_state->isLockstepAMR() ) {
        int rr = level->getRefinementRatioMaxDim();
        delt_fine /= rr;
        skip      /= rr;
      }
       
      for( int idw = 0; idw < totalFine; idw += skip ) {
        DataWarehouse* dw = m_scheduler->get_dw( idw );
        dw->override( delt_vartype( delt_fine ), m_shared_state->get_delt_label(), level );
      }
    }
     
    // Override for the global level as well (only matters on dw 0)
    DataWarehouse* oldDW = m_scheduler->get_dw(0);
    oldDW->override( delt_vartype(delt), m_shared_state->get_delt_label() );

    // A component may update the output interval or the checkpoint interval
    // during a simulation.  For example in deflagration -> detonation simulations
    if (!first && m_output) {
      if (m_shared_state->updateOutputInterval()) {
        min_vartype outputInv_var;
        oldDW->get(outputInv_var, m_shared_state->get_outputInterval_label());

        if (!outputInv_var.isBenignValue()) {
          m_output->updateOutputInterval(outputInv_var);
        }
      }
      if (m_shared_state->updateCheckpointInterval()) {
        min_vartype checkInv_var;
        oldDW->get(checkInv_var, m_shared_state->get_checkpointInterval_label());

        if (!checkInv_var.isBenignValue()) {
          m_output->updateCheckpointInterval(checkInv_var);
        }
      }
    }

    // Execute the current timestep, restarting if necessary
    executeTimestep( time, delt, currentGrid, totalFine );
      
    // Update the profiler weights
    m_lb->finalizeContributions(currentGrid);

    // If debugging, output the barrier times.
    if( dbg_barrier.active() ) {
      double start_time = Time::currentSeconds();
      Uintah::MPI::Barrier( d_myworld->getComm() );
      barrier_times[4]+=Time::currentSeconds() - start_time;
      double avg[5];
      Uintah::MPI::Reduce( barrier_times, avg, 5, MPI_DOUBLE, MPI_SUM, 0, d_myworld->getComm() );
       
      if(d_myworld->myrank()==0) {
        std::cout << "Barrier Times: ";
        for(int i=0;i<5;i++){
          avg[i]/=d_myworld->size();
          std::cout << avg[i] << " ";
        }
        std::cout << "\n";
      }
    }

    if( m_output ) {
#ifdef HAVE_PIDX
      if ( m_output->savingAsPIDX()) {
	// Only save timestep.xml if we are checkpointing.  Normal
	// time step dumps (using PIDX) do not need to write the xml
	// information.
        if( checkpointing ) {
          m_output->writeto_xml_files( delt, currentGrid );
        }
      }
      else
#endif    
      {
        // If PIDX is not being used write timestep.xml for both
        // checkpoints and time step dumps.
        m_output->writeto_xml_files( delt, currentGrid );
      }

      m_output->findNext_OutputCheckPoint_Timestep( delt, currentGrid );
    }

    // Get and reduce the performace run time stats
    getMemoryStats( m_shared_state->getCurrentTopLevelTimeStep() );
    getPAPIStats( );
    m_shared_state->d_runTimeStats.reduce(m_regridder && m_regridder->useDynamicDilation(), d_myworld);
    // Reduce the mpi run time stats.
    MPIScheduler * mpiScheduler = dynamic_cast<MPIScheduler*>( m_scheduler.get_rep() );
    
    if( mpiScheduler ) {
      mpiScheduler->mpi_info_.reduce( m_regridder && m_regridder->useDynamicDilation(),
                                      d_myworld );
    }

    // Print MPI statistics
    m_scheduler->printMPIStats();

    // Done with the time step.
    if( first ) {
      m_scheduler->setRestartInitTimestep( false );
      first = false;
    }

    // Update the time and save the delt used.
    time += delt;

    m_prev_delt = delt;

    // Retrieve the next delta T and adjust it based on timeinfo
    // parameters.
    DataWarehouse* newDW = m_scheduler->getLastDW();
    newDW->get( delt_var, m_shared_state->get_delt_label() );
    delt = delt_var;
    adjustDelT( delt, m_prev_delt, time );

    newDW->override( delt_vartype(delt), m_shared_state->get_delt_label() );

    // Before printing stats calculate the execution time.
    calcExecWallTime();
  
    // Print the stats for this time step
    printSimulationStats( m_shared_state->getCurrentTopLevelTimeStep(), delt, m_prev_delt, time, false );

    // If VisIt has been included into the build, check the lib sim
    // state to see if there is a connection and if so check to see if
    // anything needs to be done.
#ifdef HAVE_VISIT
    if( m_shared_state->getVisIt() )
    {
      // Before the in-situ update the total time.
      calcTotalWallTime();
  
      // Update all of the simulation grid and time dependent variables.
      visit_UpdateSimData( &visitSimData, currentGrid,
			   time, d_prev_delt, delt,
			   getTotalWallTime(), getTotalExecWallTime(),
			   getExecWallTime(), getExpMovingAverage(),
			   getInSituWallTime(), isLast( time ) );

      // Check the state - if the return value is true the user issued
      // a termination.
      if( visit_CheckState( &visitSimData ) )
      break;

      // This function is no longer used as last is now used in the
      // check state. 
      // Check to see if at the last iteration. If so stop so the
      // user can have once last chance see the data.
      // if( visitSimData.stopAtLastTimeStep && last )
      // visit_EndLibSim( &visitSimData );

      // The user may have adjusted delt so get it from the data
      // warehouse. If not then this call is a no-op.
      newDW->get( delt_var, m_shared_state->get_delt_label() );
      delt = delt_var;

      // Report on the modiied variables. 
      for (std::map<std::string,std::string>::iterator
          it = visitSimData.modifiedVars.begin();
          it != visitSimData.modifiedVars.end();
          ++it)
      proc0cout << "Visit libsim - At time step "
      << m_shared_state->getCurrentTopLevelTimeStep() << " "
      << "the user modified the variable " << it->first << " "
      << "to be " << it->second << ". "
      << std::endl;

      // TODO - Put this information into the NEXT time step xml.

      calcInSituWallTime();    
    }
#endif
    
    // Reset the runtime performance stats
    m_shared_state->resetStats();
    // Reset memory use tracking variable
    m_scheduler->resetMaxMemValue();
    
    // Before the next step update the total time.
    calcTotalWallTime();
  
  } // end while main time loop (time is not up, etc)
  
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
      std::cout << "HEAPCHECKER: MEMORY LEACK DETECTED!\n";
    }
    delete heap_checker;
  }
#endif
} // end run()

//______________________________________________________________________
void
AMRSimulationController::subCycleCompile( GridP & grid
                                        , int     startDW
                                        , int     dwStride
                                        , int     step
                                        , int     levelIndex
                                        )
{
  MALLOC_TRACE_TAG_SCOPE("AMRSimulationController::subCycleCompile()");

  if (amrout.active()) {
    amrout << "Start AMRSimulationController::subCycleCompile, finest level index = " << levelIndex << '\n';
  }

  // We are on the fine level with an index of levelIndex
  LevelP fineLevel = grid->getLevel(levelIndex);
  LevelP coarseLevel;
  int coarseStartDW;
  int coarseDWStride;
  int numCoarseSteps;  // how many steps between this level and the coarser
  int numFineSteps;   // how many steps between this level and the finer
  if (levelIndex > 0) {
    numCoarseSteps = m_shared_state->isLockstepAMR() ? 1 : fineLevel->getRefinementRatioMaxDim();
    coarseLevel = grid->getLevel(levelIndex - 1);
    coarseDWStride = dwStride * numCoarseSteps;
    coarseStartDW = (startDW / coarseDWStride) * coarseDWStride;
  }
  else {
    coarseDWStride = dwStride;
    coarseStartDW = startDW;
    numCoarseSteps = 0;
  }

  ASSERT(dwStride > 0 && levelIndex < grid->numLevels())

  m_scheduler->clearMappings();
  m_scheduler->mapDataWarehouse(Task::OldDW, startDW);
  m_scheduler->mapDataWarehouse(Task::NewDW, startDW + dwStride);
  m_scheduler->mapDataWarehouse(Task::CoarseOldDW, coarseStartDW);
  m_scheduler->mapDataWarehouse(Task::CoarseNewDW, coarseStartDW + coarseDWStride);

  m_sim->scheduleTimeAdvance(fineLevel, m_scheduler);

  if (m_do_amr) {
    if (levelIndex + 1 < grid->numLevels()) {
      numFineSteps = m_shared_state->isLockstepAMR() ? 1 : fineLevel->getFinerLevel()->getRefinementRatioMaxDim();
      int newStride = dwStride / numFineSteps;

      for (int substep = 0; substep < numFineSteps; substep++) {
        subCycleCompile(grid, startDW + substep * newStride, newStride, substep, levelIndex + 1);
      }

      // Coarsen and then refine_CFI at the end of the W-cycle
      m_scheduler->clearMappings();
      m_scheduler->mapDataWarehouse(Task::OldDW, 0);
      m_scheduler->mapDataWarehouse(Task::NewDW, startDW + dwStride);
      m_scheduler->mapDataWarehouse(Task::CoarseOldDW, startDW);
      m_scheduler->mapDataWarehouse(Task::CoarseNewDW, startDW + dwStride);
      m_sim->scheduleCoarsen(fineLevel, m_scheduler);
    }
  }

  m_scheduler->clearMappings();
  m_scheduler->mapDataWarehouse(Task::OldDW, startDW);
  m_scheduler->mapDataWarehouse(Task::NewDW, startDW + dwStride);
  m_scheduler->mapDataWarehouse(Task::CoarseOldDW, coarseStartDW);
  m_scheduler->mapDataWarehouse(Task::CoarseNewDW, coarseStartDW + coarseDWStride);
  m_sim->scheduleFinalizeTimestep(fineLevel, m_scheduler);

  // do refineInterface after the freshest data we can get; after the
  // finer level's coarsen completes do all the levels at this point
  // in time as well, so all the coarsens go in order, and then the
  // refineInterfaces
  if (m_do_amr && (step < numCoarseSteps - 1 || levelIndex == 0)) {

    for (int i = fineLevel->getIndex(); i < fineLevel->getGrid()->numLevels(); i++) {
      if (i == 0)
        continue;
      if (i == fineLevel->getIndex() && levelIndex != 0) {
        m_scheduler->mapDataWarehouse(Task::CoarseOldDW, coarseStartDW);
        m_scheduler->mapDataWarehouse(Task::CoarseNewDW, coarseStartDW + coarseDWStride);
        m_sim->scheduleRefineInterface(fineLevel, m_scheduler, true, true);
      }
      else {
        // look in the NewDW all the way down
        m_scheduler->mapDataWarehouse(Task::CoarseOldDW, 0);
        m_scheduler->mapDataWarehouse(Task::CoarseNewDW, startDW + dwStride);
        m_sim->scheduleRefineInterface(fineLevel->getGrid()->getLevel(i), m_scheduler, false, true);
      }
    }
  }
}

//______________________________________________________________________
//
void
AMRSimulationController::subCycleExecute( GridP & grid
                                        , int     startDW
                                        , int     dwStride
                                        , int     levelNum
                                        , bool    rootCycle
                                        )
{
  MALLOC_TRACE_TAG_SCOPE("AMRSimulationController::subCycleExecutue()");
  // there are 2n+1 taskgraphs, n for the basic timestep, n for intermediate 
  // timestep work, and 1 for the errorEstimate and stableTimestep, where n
  // is the number of levels.
  
  amrout << "Start AMRSimulationController::subCycleExecute, level=" << levelNum << '\n';

  // We are on (the fine) level numLevel
  int numSteps;
  if (levelNum == 0 || m_shared_state->isLockstepAMR()) {
    numSteps = 1;
  }
  else {
    numSteps = grid->getLevel(levelNum)->getRefinementRatioMaxDim();
  }
  
  int newDWStride = dwStride/numSteps;

//  DataWarehouse::ScrubMode oldScrubbing = (m_lb->isDynamic() || m_sim->restartableTimesteps()) ? DataWarehouse::ScrubNonPermanent : DataWarehouse::ScrubComplete;
  DataWarehouse::ScrubMode oldScrubbing = ( m_sim->restartableTimesteps()) ? DataWarehouse::ScrubNonPermanent : DataWarehouse::ScrubComplete;

  int curDW = startDW;
  for (int step = 0; step < numSteps; step++) {
  
    if (step > 0) {
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
    int iteration = curDW + (m_last_recompile_timestep == m_shared_state->getCurrentTopLevelTimeStep() ? 0 : 1);
    
    if (dbg.active()) {
      dbg << d_myworld->myrank() << "   Executing TG on level " << levelNum << " with old DW " 
          << curDW << "=" << m_scheduler->get_dw(curDW)->getID() << " and new "
          << curDW+newDWStride << "=" << m_scheduler->get_dw(curDW+newDWStride)->getID()
          << "CO-DW: " << startDW << " CNDW " << startDW+dwStride << " on iteration " << iteration << std::endl;
    }
    
    m_scheduler->execute(levelNum, iteration);
    
    if (levelNum + 1 < grid->numLevels()) {
      ASSERT(newDWStride > 0);
      subCycleExecute(grid, curDW, newDWStride, levelNum + 1, false);
    }
 
    if (m_do_amr && grid->numLevels() > 1 && (step < numSteps-1 || levelNum == 0)) {
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
        dbg << d_myworld->myrank() << "   Executing INT TG on level " << levelNum << " with old DW " 
            << curDW << "=" << m_scheduler->get_dw(curDW)->getID() << " and new "
            << curDW+newDWStride << "=" << m_scheduler->get_dw(curDW+newDWStride)->getID()
            << " CO-DW: " << startDW << " CNDW " << startDW+dwStride << " on iteration " << iteration << std::endl;
      }
      
      m_scheduler->get_dw(curDW+newDWStride)->unfinalize();
      m_scheduler->execute(levelNum+grid->numLevels(), iteration);
    }
    
    if (curDW % dwStride != 0) {
      //the currentDW(old datawarehouse) should no longer be needed - in the case of NonPermanent OldDW scrubbing
      m_scheduler->get_dw(curDW)->clear();
    }
    
  }
  if (levelNum == 0) {
    // execute the final TG
    if (dbg.active())
      dbg << d_myworld->myrank() << "   Executing Final TG on level " << levelNum << " with old DW " 
          << curDW << " = " << m_scheduler->get_dw(curDW)->getID() << " and new "
          << curDW+newDWStride << " = " << m_scheduler->get_dw(curDW+newDWStride)->getID() << std::endl;
    m_scheduler->get_dw(curDW+newDWStride)->unfinalize();
    m_scheduler->execute(m_scheduler->getNumTaskGraphs()-1, 1);
  }
} // end subCycleExecute()

//______________________________________________________________________
bool
AMRSimulationController::needRecompile(       double   time
                                      ,       double   delt
                                      , const GridP  & grid
                                      )
{
  MALLOC_TRACE_TAG_SCOPE("AMRSimulationController::needRecompile()");

  // Currently, m_output, m_sim, m_lb, m_regridder can request a recompile.
  bool recompile = false;
  
  // do it this way so everybody can have a chance to maintain their state
  recompile |= ( m_output && m_output->needRecompile(time, delt, grid));
  recompile |= ( m_sim    && m_sim->needRecompile(time, delt, grid));
  recompile |= ( m_lb     && m_lb->needRecompile(time, delt, grid));
  recompile |= ( m_shared_state->getRecompileTaskGraph() );
  
  if (m_do_amr){
    recompile |= ( m_regridder && m_regridder->needRecompile(time, delt, grid) );
  }

#ifdef HAVE_VISIT
  // Check all of the component variables that might require the task
  // graph to be recompiled.

  // ARS - Should this check be on the component level?
  for( unsigned int i=0; i<m_shared_state->d_UPSVars.size(); ++i )
  {
    SimulationState::interactiveVar &var = m_shared_state->d_UPSVars[i];

    if( var.modified && var.recompile )
    {
      recompile = true;
    }
  }
#endif
  
  return recompile;
}
//______________________________________________________________________
void
AMRSimulationController::doInitialTimestep( GridP & grid, double & time )
{
  MALLOC_TRACE_TAG_SCOPE("AMRSimulationController::doInitialTimestep()");

  double start = Time::currentSeconds();

  m_scheduler->mapDataWarehouse(Task::OldDW, 0);
  m_scheduler->mapDataWarehouse(Task::NewDW, 1);
  m_scheduler->mapDataWarehouse(Task::CoarseOldDW, 0);
  m_scheduler->mapDataWarehouse(Task::CoarseNewDW, 1);

  if (m_restarting) {

    m_lb->possiblyDynamicallyReallocate(grid, LoadBalancerPort::restart);
    // tsaad & bisaac: At this point, during a restart, a grid does
    // NOT have knowledge of the boundary conditions.  (See other
    // comments in SimulationController.cc for why that is the
    // case). Here, and given a legitimate load balancer, we can
    // assign the BCs to the grid in an efficient manner.
    grid->assignBCS(m_grid_ps, m_lb);

    grid->performConsistencyCheck();

    m_sim->restartInitialize();

    for (int i = grid->numLevels() - 1; i >= 0; i--) {
      m_sim->scheduleRestartInitialize(grid->getLevel(i), m_scheduler);
    }
    m_scheduler->compile();
    m_scheduler->get_dw(1)->setScrubbing(DataWarehouse::ScrubNone);
    m_scheduler->execute();

    // Now we know we're done with any additions to the new DW - finalize it
    m_scheduler->get_dw(1)->finalize();

    if (m_regridder && m_regridder->isAdaptive()) {
      // On restart:
      //   we must set up the tasks (but not compile) so we can have the
      //   initial OldDW Requirements in order to regrid straightaway
      for (int i = grid->numLevels() - 1; i >= 0; i--) {
        m_sim->scheduleTimeAdvance(grid->getLevel(i), m_scheduler);
      }
    }
  }
  else {
    m_shared_state->setCurrentTopLevelTimeStep(0);

    // for dynamic lb's, set up initial patch config
    m_lb->possiblyDynamicallyReallocate(grid, LoadBalancerPort::init);
    grid->assignBCS(m_grid_ps, m_lb);
    grid->performConsistencyCheck();
    time = m_time_info->initTime;

    bool needNewLevel = false;
    do {
      if (needNewLevel) {
        m_scheduler->initialize(1, 1);
        m_scheduler->advanceDataWarehouse(grid, true);
      }

      proc0cout << "Compiling initialization taskgraph...\n";

      // Initialize the CFD and/or MPM data
      for (int i = grid->numLevels() - 1; i >= 0; i--) {
        m_sim->scheduleInitialize(grid->getLevel(i), m_scheduler);

        if (m_regridder) {
          // so we can initially regrid
          m_regridder->scheduleInitializeErrorEstimate(grid->getLevel(i));
          m_sim->scheduleInitialErrorEstimate(grid->getLevel(i), m_scheduler);

          if (i < m_regridder->maxLevels() - 1) {  // we don't use error estimates if we don't make another level, so don't dilate
            m_regridder->scheduleDilation(grid->getLevel(i));
          }
        }
      }
      scheduleComputeStableTimestep(grid, m_scheduler);

      if (m_output) {
        double delT = 0;
        bool recompile = true;
        m_output->finalizeTimestep(time, delT, grid, m_scheduler, recompile);
        m_output->sched_allOutputTasks(delT, grid, m_scheduler, recompile);
      }

      m_scheduler->compile();
      double end = Time::currentSeconds() - start;

      proc0cout << "done initialization taskgraph compile (" << end << " seconds)\n";
      // No scrubbing for initial step
      m_scheduler->get_dw(1)->setScrubbing(DataWarehouse::ScrubNone);
      m_scheduler->execute();

      needNewLevel = m_regridder && m_regridder->isAdaptive() && grid->numLevels() < m_regridder->maxLevels()
                     && doRegridding(grid, true);
    }
    while (needNewLevel);

    if (m_output) {
      m_output->findNext_OutputCheckPoint_Timestep(0, grid);
      m_output->writeto_xml_files(0, grid);
    }
  }
}  // end doInitialTimestep()

//______________________________________________________________________

bool
AMRSimulationController::doRegridding( GridP & currentGrid, bool initialTimestep )
{
  MALLOC_TRACE_TAG_SCOPE("AMRSimulationController::doRegridding()");

  double start = Time::currentSeconds();

  GridP oldGrid = currentGrid;
  currentGrid = m_regridder->regrid(oldGrid.get_rep());

  if (dbg_barrier.active()) {
    double start;
    start = Time::currentSeconds();
    Uintah::MPI::Barrier(d_myworld->getComm());
    barrier_times[0] += Time::currentSeconds() - start;
  }

  double regridTime = Time::currentSeconds() - start;
  m_shared_state->d_runTimeStats[SimulationState::RegriddingTime] += regridTime;
  m_shared_state->setRegridTimestep(false);

  int lbstate = initialTimestep ? LoadBalancerPort::init : LoadBalancerPort::regrid;

  if (currentGrid != oldGrid) {
    m_shared_state->setRegridTimestep(true);

    m_lb->possiblyDynamicallyReallocate(currentGrid, lbstate);
    if (dbg_barrier.active()) {
      double start;
      start = Time::currentSeconds();
      Uintah::MPI::Barrier(d_myworld->getComm());
      barrier_times[1] += Time::currentSeconds() - start;
    }

    currentGrid->assignBCS(m_grid_ps, m_lb);
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

      if (amrout.active()) {
        amrout << "---------- NEW GRID ----------" << std::endl;
        amrout << "Grid has " << currentGrid->numLevels() << " level(s)" << std::endl;

        for (int levelIndex = 0; levelIndex < currentGrid->numLevels(); levelIndex++) {
          LevelP level = currentGrid->getLevel(levelIndex);

          amrout << "  Level " << level->getID() << ", indx: " << level->getIndex() << " has " << level->numPatches()
                 << " patch(es)" << std::endl;

          for (Level::patch_iterator patchIter = level->patchesBegin(); patchIter < level->patchesEnd(); patchIter++) {
            const Patch* patch = *patchIter;
            amrout << "(Patch " << patch->getID() << " proc " << m_lb->getPatchwiseProcessorAssignment(patch) << ": box="
                   << patch->getExtraBox() << ", lowIndex=" << patch->getExtraCellLowIndex() << ", highIndex="
                   << patch->getExtraCellHighIndex() << ")" << std::endl;
          }
        }
      }
    }  // rank 0

    double scheduleTime = Time::currentSeconds();

    if (!initialTimestep) {
      m_scheduler->scheduleAndDoDataCopy(currentGrid, m_sim);
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

//______________________________________________________________________

void
AMRSimulationController::recompile( double   time
                                  , double   delt
                                  , GridP  & currentGrid
                                  , int      totalFine
                                  )
{
  MALLOC_TRACE_TAG_SCOPE("AMRSimulationController::Recompile()");

  proc0cout << "Compiling taskgraph(s)...\n";

  m_last_recompile_timestep = m_shared_state->getCurrentTopLevelTimeStep();
  double start = Time::currentSeconds();

  m_scheduler->initialize(1, totalFine);
  m_scheduler->fillDataWarehouses(currentGrid);

  // Set up new DWs, DW mappings.
  m_scheduler->clearMappings();
  m_scheduler->mapDataWarehouse(Task::OldDW, 0);
  m_scheduler->mapDataWarehouse(Task::NewDW, totalFine);
  m_scheduler->mapDataWarehouse(Task::CoarseOldDW, 0);
  m_scheduler->mapDataWarehouse(Task::CoarseNewDW, totalFine);

  if (m_do_multi_taskgraphing) {
    for (int i = 0; i < currentGrid->numLevels(); i++) {
      // taskgraphs 0-numlevels-1
      if (i > 0) {
        // we have the first one already
        m_scheduler->addTaskGraph(Scheduler::NormalTaskGraph);
      }
      dbg << d_myworld->myrank() << "   Creating level " << i << " tg " << std::endl;
      m_sim->scheduleTimeAdvance(currentGrid->getLevel(i), m_scheduler);
    }

    for (int i = 0; i < currentGrid->numLevels(); i++) {
      if (m_do_amr && currentGrid->numLevels() > 1) {
        dbg << d_myworld->myrank() << "   Doing Intermediate TG level " << i << " tg " << std::endl;
        // taskgraphs numlevels-2*numlevels-1
        m_scheduler->addTaskGraph(Scheduler::IntermediateTaskGraph);
      }

      // schedule a coarsen from the finest level to this level
      for (int j = currentGrid->numLevels() - 2; j >= i; j--) {
        dbg << d_myworld->myrank() << "   schedule coarsen on level " << j << std::endl;
        m_sim->scheduleCoarsen(currentGrid->getLevel(j), m_scheduler);
      }

      m_sim->scheduleFinalizeTimestep(currentGrid->getLevel(i), m_scheduler);

      // schedule a refineInterface from this level to the finest level
      for (int j = i; j < currentGrid->numLevels(); j++) {
        if (j != 0) {
          dbg << d_myworld->myrank() << "   schedule RI on level " << j << " for tg " << i << " coarseold " << (j == i)
              << " coarsenew " << true << std::endl;
          m_sim->scheduleRefineInterface(currentGrid->getLevel(j), m_scheduler, j == i, true);
        }
      }
    }
    // for the final error estimate and stable timestep tasks
    m_scheduler->addTaskGraph(Scheduler::IntermediateTaskGraph);
  }
  else {
    subCycleCompile(currentGrid, 0, totalFine, 0, 0);
    m_scheduler->clearMappings();
    m_scheduler->mapDataWarehouse(Task::OldDW, 0);
    m_scheduler->mapDataWarehouse(Task::NewDW, totalFine);
  }

  for (int i = currentGrid->numLevels() - 1; i >= 0; i--) {
    dbg << d_myworld->myrank() << "   final TG " << i << "\n";

    if (m_regridder) {
      m_regridder->scheduleInitializeErrorEstimate(currentGrid->getLevel(i));
      m_sim->scheduleErrorEstimate(currentGrid->getLevel(i), m_scheduler);

      if (i < m_regridder->maxLevels() - 1) {  // we don't use error estimates if we don't make another level, so don't dilate
        m_regridder->scheduleDilation(currentGrid->getLevel(i));
      }
    }
  }

  scheduleComputeStableTimestep(currentGrid, m_scheduler);

  if (m_output) {
    m_output->finalizeTimestep(time, delt, currentGrid, m_scheduler, true);
    m_output->sched_allOutputTasks(delt, currentGrid, m_scheduler, true);
  }

  m_scheduler->compile();

  double dt = Time::currentSeconds() - start;

  proc0cout << "DONE TASKGRAPH RE-COMPILE (" << dt << " seconds)\n";

  m_shared_state->d_runTimeStats[SimulationState::CompilationTime] += dt;

}  // end recompile()

//______________________________________________________________________

void
AMRSimulationController::executeTimestep( double   time
                                        , double & delt
                                        , GridP  & currentGrid
                                        , int      totalFine
                                        )
{
  MALLOC_TRACE_TAG_SCOPE("AMRSimulationController::executeTimestep()");

  // If the timestep needs to be restarted, this loop will execute multiple times.
  bool success = true;
  double orig_delt = delt;

  do {
    bool restartable = m_sim->restartableTimesteps();
    m_scheduler->setRestartable(restartable);

    // Standard data warehouse scrubbing.
    if (scrubDataWarehouse && m_lb->getNthRank() == 1) {
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
    // If not scubbing or getNthRank requires the variables after they would have been scrubbed so turn off all scrubbing.
    else {  //if( !scrubDataWarehouse || m_lb->getNthRank() > 1 )
      for (int i = 0; i <= totalFine; ++i) {
        m_scheduler->get_dw(i)->setScrubbing(DataWarehouse::ScrubNone);
      }
    }

    // TODO: Let's document this well - APH, 10/06/16
    if (m_do_multi_taskgraphing) {
      subCycleExecute(currentGrid, 0, totalFine, 0, true);
    }
    else {
      int curr_timestep = m_shared_state->getCurrentTopLevelTimeStep();
      int tg_num = ( (curr_timestep % m_rad_calc_frequency != 0) && (curr_timestep != 1) ) ? 0 : 1;
      m_scheduler->execute( (m_scheduler->getNumTaskGraphs() == 1) ? 0 : tg_num, (m_last_recompile_timestep == curr_timestep) ? 0 : 1);
    }

    //__________________________________
    //  If timestep has been restarted
    if (m_scheduler->get_dw(totalFine)->timestepRestarted()) {
      ASSERT(restartable);

      // Figure out new delt
      double new_delt = m_sim->recomputeTimestep(delt);

      proc0cout << "Restarting timestep at " << time << ", changing delt from " << delt << " to " << new_delt << '\n';

      // bulletproofing
      if (new_delt < m_time_info->delt_min || new_delt <= 0) {
        std::ostringstream warn;
        warn << "The new delT (" << new_delt << ") is either less than delT_min (" << m_time_info->delt_min << ") or equal to 0";
        throw InternalError(warn.str(), __FILE__, __LINE__);
      }

      m_output->reEvaluateOutputTimestep(orig_delt, new_delt);
      delt = new_delt;

      m_scheduler->get_dw(0)->override(delt_vartype(new_delt), m_shared_state->get_delt_label());

      for (int i = 1; i <= totalFine; i++) {
        m_scheduler->replaceDataWarehouse(i, currentGrid);
      }

      double delt_fine = delt;
      int skip = totalFine;
      for (int i = 0; i < currentGrid->numLevels(); i++) {

        const Level* level = currentGrid->getLevel(i).get_rep();
        if (i != 0 && !m_shared_state->isLockstepAMR()) {
          int trr = level->getRefinementRatioMaxDim();
          delt_fine /= trr;
          skip /= trr;
        }

        for (int idw = 0; idw < totalFine; idw += skip) {
          DataWarehouse* dw = m_scheduler->get_dw(idw);
          dw->override(delt_vartype(delt_fine), m_shared_state->get_delt_label(), level);
        }
      }
      success = false;
    }
    else {
      if (m_scheduler->get_dw(1)->timestepAborted()) {
        throw InternalError("Execution aborted, cannot restart timestep\n", __FILE__, __LINE__);
      }
      success = true;
    }
  }
  while (!success);

}  // end executeTimestep()

//______________________________________________________________________
//

void
AMRSimulationController::scheduleComputeStableTimestep( const GridP & grid, SchedulerP & sched )
{
  MALLOC_TRACE_TAG_SCOPE("AMRSimulationController::scheduleComputeStableTimestep()");

  for (int i = 0; i < grid->numLevels(); i++) {
    m_sim->scheduleComputeStableTimestep(grid->getLevel(i), sched);
  }

  Task* task = scinew Task("reduceSysVar", this, &AMRSimulationController::reduceSysVar);

  //coarsenDelT task requires that delT is computed on every level, even if no tasks are 
  // run on that level.  I think this is a bug.  --Todd
  for (int i = 0; i < grid->numLevels(); i++) {
    task->requires(Task::NewDW, m_shared_state->get_delt_label(), grid->getLevel(i).get_rep());
  }

  if (m_shared_state->updateOutputInterval()) {
    task->requires(Task::NewDW, m_shared_state->get_outputInterval_label());
  }

  if (m_shared_state->updateCheckpointInterval()) {
    task->requires(Task::NewDW, m_shared_state->get_checkpointInterval_label());
  }

  //coarsen delt computes the global delt variable
  task->computes(m_shared_state->get_delt_label());
  task->setType(Task::OncePerProc);
  task->usesMPI(true);
  sched->addTask(task, m_lb->getPerProcessorPatchSet(grid), m_shared_state->allMaterials());
}

//______________________________________________________________________
//
void
AMRSimulationController::reduceSysVar( const ProcessorGroup *
                                     , const PatchSubset    * patches
                                     , const MaterialSubset * /*matls*/
                                     ,       DataWarehouse  * /*old_dw*/
                                     ,       DataWarehouse  *  new_dw
                                     )
{
  MALLOC_TRACE_TAG_SCOPE("AMRSimulationController::reduceSysVar()");

  // the goal of this task is to line up the delt across all levels.  If the coarse one
  // already exists (the one without an associated level), then we must not be doing AMR
  Patch* patch = nullptr;
  if (patches->size() != 0 && !new_dw->exists(m_shared_state->get_delt_label(), -1, patch)) {
    int multiplier = 1;
    const GridP grid = patches->get(0)->getLevel()->getGrid();

    for (int i = 0; i < grid->numLevels(); i++) {
      const LevelP level = grid->getLevel(i);

      if (i > 0 && !m_shared_state->isLockstepAMR()) {
        multiplier *= level->getRefinementRatioMaxDim();
      }

      if (new_dw->exists(m_shared_state->get_delt_label(), -1, *level->patchesBegin())) {
        delt_vartype deltvar;
        double delt;
        new_dw->get(deltvar, m_shared_state->get_delt_label(), level.get_rep());

        delt = deltvar;
        new_dw->put(delt_vartype(delt * multiplier), m_shared_state->get_delt_label());
      }
    }
  }

  if (d_myworld->size() > 1) {
    new_dw->reduceMPI(m_shared_state->get_delt_label(), 0, 0, -1);
  }

  // reduce output interval and checkpoint interval 
  // if no value computed on that MPI rank,  benign value will be set
  // when the reduction result is also benign value, this value will be ignored 
  // that means no MPI rank want to change the interval

  if (m_shared_state->updateOutputInterval()) {

    if (patches->size() != 0 && !new_dw->exists(m_shared_state->get_outputInterval_label(), -1, patch)) {
      min_vartype inv;
      inv.setBenignValue();
      new_dw->put(inv, m_shared_state->get_outputInterval_label());
    }
    if (d_myworld->size() > 1) {
      new_dw->reduceMPI(m_shared_state->get_outputInterval_label(), 0, 0, -1);
    }

  }

  if (m_shared_state->updateCheckpointInterval()) {

    if (patches->size() != 0 && !new_dw->exists(m_shared_state->get_checkpointInterval_label(), -1, patch)) {
      min_vartype inv;
      inv.setBenignValue();
      new_dw->put(inv, m_shared_state->get_checkpointInterval_label());
    }
    if (d_myworld->size() > 1) {
      new_dw->reduceMPI(m_shared_state->get_checkpointInterval_label(), 0, 0, -1);
    }

  }

}  // end reduceSysVar()
