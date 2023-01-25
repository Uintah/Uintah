/*
 * The MIT License
 *
 * Copyright (c) 1997-2020 The University of Utah
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

#include <Core/DataArchive/DataArchive.h>
#include <Core/Exceptions/InternalError.h>
#include <Core/Grid/Grid.h>
#include <Core/Grid/MaterialManager.h>
#include <Core/Grid/Variables/VarTypes.h>
#include <Core/OS/Dir.h>
#include <Core/OS/ProcessInfo.h>
#include <Core/Parallel/Parallel.h>
#include <Core/Parallel/ProcessorGroup.h>
#include <Core/ProblemSpec/ProblemSpec.h>
#include <Core/Util/DOUT.hpp>

#include <CCA/Components/Schedulers/MPIScheduler.h>
#include <CCA/Ports/LoadBalancer.h>
#include <CCA/Ports/Output.h>
#include <CCA/Ports/ProblemSpecInterface.h>
#include <CCA/Ports/Regridder.h>
#include <CCA/Ports/Scheduler.h>
#include <CCA/Ports/ApplicationInterface.h>


#include <sci_defs/malloc_defs.h>
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

namespace {

Uintah::Dout g_sim_stats(       "SimulationStats"           , "SimulationController", "Simulation general stats"         , true  );
Uintah::Dout g_sim_stats_mem(   "SimulationStatsMem"        , "SimulationController", "Simulation memory stats"          , true  );

Uintah::Dout g_comp_stats(      "ComponentStats"            , "SimulationController", "Aggregated component stats"       , false );
Uintah::Dout g_comp_node_stats( "ComponentNodeStats"        , "SimulationController", "Aggregated node component stats"  , false );
Uintah::Dout g_comp_indv_stats( "ComponentIndividualStats"  , "SimulationController", "Individual component stats"       , false );

Uintah::Dout g_app_stats(       "ApplicationStats"          , "SimulationController", "Aggregated application stats"     , false );
Uintah::Dout g_app_node_stats(  "ApplicationNodeStats"      , "SimulationController", "Aggregated node application stats", false );
Uintah::Dout g_app_indv_stats(  "ApplicationIndividualStats", "SimulationController", "Individual application stats"     , false );

}

namespace Uintah {

SimulationController::SimulationController( const ProcessorGroup * myworld
                                          ,       ProblemSpecP     prob_spec
                                          )
  : UintahParallelComponent( myworld )
  , m_ups( prob_spec )
{
  //initialize the overhead percentage
  for (int i = 0; i < OVERHEAD_WINDOW; ++i) {
    double x = (double) i / (double) (OVERHEAD_WINDOW/2);
    m_overhead_values[i]  = 0;
    m_overhead_weights[i] = 8.0 - x*x*x;
  }
  
  m_grid_ps = m_ups->findBlock( "Grid" );

  ProblemSpecP simController_ps = m_ups->findBlock( "SimulationController" );

  if (simController_ps) {
    ProblemSpecP runtimeStats_ps = simController_ps->findBlock("RuntimeStats");

    if (runtimeStats_ps) {
      runtimeStats_ps->get("frequency", m_reportStatsFrequency);
      runtimeStats_ps->get("onTimeStep", m_reportStatsOnTimeStep);

      if (m_reportStatsOnTimeStep >= m_reportStatsFrequency) {
        proc0cout << "Error: the frequency of reporting the run time stats " << m_reportStatsFrequency  << " "
                  << "is less than or equal to the time step ordinality "    << m_reportStatsOnTimeStep << " "
                  << ". Resetting the ordinality to ";

        if (m_reportStatsFrequency > 1) {
          m_reportStatsOnTimeStep = 1;
        }
        else {
          m_reportStatsOnTimeStep = 0;
        }
        proc0cout << m_reportStatsOnTimeStep << std::endl;
      }
    }
  }

  std::string timeStr("seconds");
  std::string bytesStr("MBytes");

  m_runtime_stats.insert( CompilationTime,           std::string("Compilation"),           timeStr );
  m_runtime_stats.insert( RegriddingTime,            std::string("Regridding"),            timeStr );
  m_runtime_stats.insert( RegriddingCompilationTime, std::string("RegriddingCompilation"), timeStr );
  m_runtime_stats.insert( RegriddingCopyDataTime,    std::string("RegriddingCopyData"),    timeStr );
  m_runtime_stats.insert( LoadBalancerTime,          std::string("LoadBalancer"),          timeStr );

  m_runtime_stats.insert( TaskExecTime,              std::string("TaskExec"),              timeStr );
  m_runtime_stats.insert( TaskLocalCommTime,         std::string("TaskLocalComm"),         timeStr );
  m_runtime_stats.insert( TaskWaitCommTime,          std::string("TaskWaitCommTime"),      timeStr );
  m_runtime_stats.insert( TaskReduceCommTime,        std::string("TaskReduceCommTime"),    timeStr );
  m_runtime_stats.insert( TaskWaitThreadTime,        std::string("TaskWaitThread"),        timeStr );

  m_runtime_stats.insert( XMLIOTime,                 std::string("XMLIO"),                 timeStr );
  m_runtime_stats.insert( OutputIOTime,              std::string("OutputIO"),              timeStr );
  m_runtime_stats.insert( OutputGlobalIOTime,        std::string("OutputGlobalIO"),        timeStr );
  m_runtime_stats.insert( CheckpointIOTime,          std::string("CheckpointIO"),          timeStr );
  m_runtime_stats.insert( CheckpointGlobalIOTime,    std::string("CheckpointGlobalIO"),    timeStr );
  m_runtime_stats.insert( TotalIOTime,               std::string("TotalIO"),               timeStr );

  m_runtime_stats.insert( OutputIORate,              std::string("OutputIORate"),           "MBytes/sec" );
  m_runtime_stats.insert( OutputGlobalIORate,        std::string("OutputGlobalIORate"),     "MBytes/sec" );
  m_runtime_stats.insert( CheckpointIORate,          std::string("CheckpointIORate"),       "MBytes/sec" );
  m_runtime_stats.insert( CheckpointGlobalIORate,    std::string("CheckpointGlobalIORate"), "MBytes/sec" );

  m_runtime_stats.insert( NumTasks,                  std::string("NumberOfTasks"),     "tasks" );
  m_runtime_stats.insert( NumPatches,                std::string("NumberOfPatches"),   "patches" );
  m_runtime_stats.insert( NumCells,                  std::string("NumberOfCells"),     "cells" );
  m_runtime_stats.insert( NumParticles,              std::string("NumberOfParticles"), "paticles" );

  m_runtime_stats.insert( SCIMemoryUsed,             std::string("SCIMemoryUsed"),         bytesStr );
  m_runtime_stats.insert( SCIMemoryMaxUsed,          std::string("SCIMemoryMaxUsed"),      bytesStr );
  m_runtime_stats.insert( SCIMemoryHighwater,        std::string("SCIMemoryHighwater"),    bytesStr );
  m_runtime_stats.insert( MemoryUsed,                std::string("MemoryUsed"),            bytesStr );
  m_runtime_stats.insert( MemoryResident,            std::string("MemoryResident"),        bytesStr );

  m_runtime_stats.calculateRankMinimum(true);
  m_runtime_stats.calculateRankStdDev (true);

  if( g_comp_node_stats ) {
    m_runtime_stats.calculateNodeSum    ( true );
    m_runtime_stats.calculateNodeMinimum( true );
    m_runtime_stats.calculateNodeAverage( true );
    m_runtime_stats.calculateNodeMaximum( true );
    m_runtime_stats.calculateNodeStdDev ( true );
  }

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

  if (!m_application) {
    throw InternalError("dynamic_cast of 'm_app' failed!", __FILE__, __LINE__);
  }

  m_loadBalancer = dynamic_cast<LoadBalancer*>( getPort("load balancer") );

  if (!m_loadBalancer) {
    throw InternalError("dynamic_cast of 'm_loadBalancer' failed!", __FILE__, __LINE__);
  }

  m_output = dynamic_cast<Output*>( getPort("output") );

  if (!m_output) {
    throw InternalError("dynamic_cast of 'm_output' failed!", __FILE__, __LINE__);
  }

  m_regridder = dynamic_cast<Regridder*>( getPort("regridder") );

  if (m_application->isDynamicRegridding() && !m_regridder) {
    throw InternalError("dynamic_cast of 'm_regridder' failed!", __FILE__, __LINE__);
  }

  m_scheduler = dynamic_cast<Scheduler*>( getPort("scheduler") );

  if (!m_scheduler) {
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
                               ,       int           index
                               ,       bool          fromScratch
                               ,       bool          removeOldDir
                               )
{
  m_restarting             = true;
  m_from_dir               = restartFromDir;
  m_restart_index          = index;
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

    // Find the right checkpoint timestep to query the grid
    if( indices.size() == 0) {
      std::ostringstream message;
      message << "No restart checkpoints found.";
      throw InternalError(message.str(), __FILE__, __LINE__);
    }
    else if (m_restart_index < 0 ) {
      m_restart_index = (unsigned int) (indices.size() - 1);
    }
    else if (m_restart_index >= indices.size() ) {
      std::ostringstream message;
      message << "Invalid restart checkpoint index " << m_restart_index << ". "
              << "Found " << indices.size() << " checkpoints";
      throw InternalError(message.str(), __FILE__, __LINE__);
    }

    m_restart_timestep = indices[m_restart_index];

    // Do this call before calling DataArchive::restartInitialize,
    // because problemSetup() creates VarLabels the DataArchive needs.
    m_restart_ps =
      m_restart_archive->getTimestepDocForComponent( m_restart_index );

    proc0cout << "Restart directory: \t'" << restartFromDir.getName() << "'\n"
              << "Restart time step: \t" << m_restart_timestep << "\n";
  }
}

//______________________________________________________________________
//
void
SimulationController::outputSetup( void )
{  
  // Set up the output - needs to be done before the application is setup.
  m_output->setRuntimeStats( &m_runtime_stats );

  m_output->problemSetup( m_ups, m_restart_ps, m_application->getMaterialManagerP() );

#ifdef HAVE_VISIT
  if( getVisIt() ) {
    m_output->setScrubSavedVariables( false );
  }
#endif
}

//______________________________________________________________________
//
void
SimulationController::gridSetup( void )
{
  // Set up the grid.
  if (m_restarting) {
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
    m_current_gridP = m_restart_archive->queryGrid( m_restart_index, m_ups, false );
  }
  else /* if( !m_restarting ) */ {
    m_current_gridP = scinew Grid();

    // The call to preGridProblemSetup() by the simulation interface allows
    // for a call to grid->setExtraCells() to be made before the grid
    // problemSetup() so that if there are no extra cells defined in the
    // ups file, all levels will use the grid extra cell value.

    // For instance, Wasatch does not allow users to specify extra
    // cells through the input file. Instead, Wasatch wants to specify
    // it internally. This call gives the option to do just that though
    // it does not follow normal paradigm of calling problemSetup
    // immediately after a component or other object is created.
    m_application->preGridProblemSetup( m_ups, m_current_gridP );

    // Now that the simulation interface has made its changes do the normal grid problemSetup()
    m_current_gridP->problemSetup( m_ups, d_myworld, m_application->isAMR() );
  }

  if (m_current_gridP->numLevels() == 0) {
    throw InternalError("No problem (no levels in grid) specified.", __FILE__, __LINE__);
  }

  // Print out metadata
  if (d_myworld->myRank() == 0) {
    m_current_gridP->printStatistics();
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
  if (m_regridder) {
    m_regridder->problemSetup(m_ups, m_current_gridP, m_application->getMaterialManagerP());
  }
}

//______________________________________________________________________
//
void
SimulationController::schedulerSetup( void )
{
  // Now that the grid is completely set up, set up the scheduler.
  m_scheduler->setRuntimeStats( &m_runtime_stats );

  m_scheduler->problemSetup( m_ups, m_application->getMaterialManagerP() );

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
  m_loadBalancer->problemSetup( m_ups, m_current_gridP, m_application->getMaterialManagerP() );
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

  // Finalize the materials
  m_application->getMaterialManagerP()->finalizeMaterials();

  m_application->setRestartTimeStep( m_restarting );
}


//______________________________________________________________________
//
void
SimulationController::timeStateSetup()
{
  // Restarting so initialize time state using the archive data.
  if( m_restarting ) {

    double simTime;
    
    m_restart_archive->restartInitialize( m_restart_index,
                                          m_current_gridP,
                                          m_scheduler->get_dw(1),
                                          m_loadBalancer,
                                          &simTime );

    // Set the time step to the restart time step which is immediately
    // written to the DW.
    m_application->setTimeStep( m_restart_timestep );

    // Set the simulation time to the restart simulation time which is
    // immediately written to the DW.
    m_application->setSimTime( simTime );

    // Set the next delta T which is immediately written to the DW.

    // Note the old delta t is the delta t used for that time step.
    m_application->setNextDelT( m_restart_archive->getOldDelt( m_restart_index ), m_restarting );

    // Tell the scheduler the generation of the re-started simulation.
    // (Add +1 because the scheduler will be starting on the next
    // time step.)
    m_scheduler->setGeneration( m_restart_timestep + 1 );

    // This delete is an enigma. If it is called then memory is not
    // leaked, but sometimes if is called, then everything segfaults.
    // delete m_restart_archive;
  }
  else {
    // Set the default time step to which is immediately written to the DW.
    m_application->setTimeStep( m_application->getTimeStep() );

    // Set the default simulation time which is immediately written to the DW.
    m_application->setSimTime( m_application->getSimTime() );

    // Note the above seems back asswards but the initial sim time
    // must be set in the UPS file, the time step will always default
    // to 0. However, it is read in the problem setup stage and the
    // data warehouse is not yet available. So the above gets the
    // values in the data warehouse.
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
  m_output->initializeOutput( m_ups, m_current_gridP );

  // This step is done after the output is initialized so that global
  // reduction output vars are copied to the new uda. Further, it must
  // be called after timeStateSetup() is call so that checkpoints are
  // copied to the new uda as well.
  if (m_restarting) {
    Dir dir( m_from_dir );
    m_output->restartSetup( dir, 0, m_restart_timestep,
                            m_application->getSimTime(),
                            m_restart_from_scratch, m_restart_remove_old_dir );
  }

  // Miscellaneous initializations.
  ProblemSpecP amr_ps = m_ups->findBlock("AMR");
  if (amr_ps) {
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
  Task* task = scinew Task("SimulationController::ReportStats", this, &SimulationController::ReportStats, header);
  
  task->setType(Task::OncePerProc);

  // Require delta T so that the task gets scheduled
  // correctly. Otherwise the scheduler/taskgraph will toss an error :
  // Caught std exception: map::at: key not found
  task->requires(Task::NewDW, m_application->getDelTLabel() );

  m_scheduler->addTask(task,
                       m_loadBalancer->getPerProcessorPatchSet(m_current_gridP),
                       m_application->getMaterialManagerP()->allMaterials() );
}

//______________________________________________________________________
//
void
SimulationController::ReportStats(const ProcessorGroup*,
                                  const PatchSubset*,
                                  const MaterialSubset*,
                                        DataWarehouse*,
                                        DataWarehouse*,
                                        bool header )
{
  bool reportStats = false;

  // If the reporting frequency is greater than 1 check to see if output is needed.
  if (m_reportStatsFrequency == 1) {
    reportStats = true;
  }

  // Note: this check is split up so to be assured that the call to
  // isLastTimeStep is last and called only when needed. Unfortunately,
  // the greater the frequency the more often it will be called.
  else {
    if (header) {
      reportStats = true;
    }
    else if (m_application->getTimeStep() % m_reportStatsFrequency == m_reportStatsOnTimeStep) {
      reportStats = true;
    }
    else {
      // Get the wall time if is needed, otherwise ignore it.
      double walltime;

      if (m_application->getWallTimeMax() > 0) {
        walltime = m_wall_timers.GetWallTime();
      }
      else {
        walltime = 0;
      }
      reportStats = m_application->isLastTimeStep(walltime);
    }
  }
  
  // Get and reduce the performance runtime stats
  getMemoryStats();

#ifdef HAVE_VISIT
  bool reduce = getVisIt();
#else
  bool reduce = false;
#endif
  
  // Reductions are only need if these are true.
  if ((m_regridder && m_regridder->useDynamicDilation()) || g_sim_stats_mem || g_comp_stats || g_comp_node_stats || reduce) {

    m_runtime_stats.reduce(m_regridder && m_regridder->useDynamicDilation(), d_myworld);

    // Reduce the MPI runtime stats.
    MPIScheduler * mpiScheduler = dynamic_cast<MPIScheduler*>(m_scheduler.get_rep());

    if (mpiScheduler) {
      mpiScheduler->m_mpi_info.reduce(m_regridder && m_regridder->useDynamicDilation(), d_myworld);
    }
  }

  if (g_app_stats || g_app_node_stats || reduce) {
    m_application->getApplicationStats().calculateNodeSum    ( true );
    m_application->getApplicationStats().calculateNodeMinimum( true );
    m_application->getApplicationStats().calculateNodeAverage( true );
    m_application->getApplicationStats().calculateNodeMaximum( true );
    m_application->getApplicationStats().calculateNodeStdDev ( true );

    m_application->reduceApplicationStats(m_regridder && m_regridder->useDynamicDilation(), d_myworld);
  }
  
  // Update the moving average and get the wall time for this time step.
  // Timers::nanoseconds timeStepTime =
    m_wall_timers.updateExpMovingAverage();

  // Print the stats for this time step
  if (d_myworld->myRank() == 0 && g_sim_stats) {
    std::ostringstream message;

    if( header ) {
      message << std::endl
              << "Simulation and run time stats are reported "
              << "at the end of each time step" << std::endl
              << "EMA == Wall time as an exponential moving average "
              << "using a window of the last " << m_wall_timers.getWindow()
              << " time steps" << std::endl;
    }

    message << std::left
            << "Timestep "      << std::setw(8)  << m_application->getTimeStep()
            << "Time="          << std::setw(12) << m_application->getSimTime()
            << "Next delT="     << std::setw(12) << m_application->getNextDelT()
            << "Wall Time="     << std::setw(10) << m_wall_timers.GetWallTime()
//          << "Net Wall Time=" << std::setw(10) << timeStepTime.seconds()
            << "EMA="           << std::setw(12) << m_wall_timers.ExpMovingAverage().seconds();

    // Report on the memory used.
    if (g_sim_stats_mem) {
      // With the sum reduces, use double, since with memory it is possible that it will overflow
      double        avg_memused      = m_runtime_stats.getRankAverage(    SCIMemoryUsed );
      unsigned long max_memused      = m_runtime_stats.getRankMaximum(    SCIMemoryUsed );
      int           max_memused_rank = m_runtime_stats.getRankForMaximum( SCIMemoryUsed );
      
      double        avg_highwater      = m_runtime_stats.getRankAverage(    SCIMemoryHighwater );
      unsigned long max_highwater      = m_runtime_stats.getRankMaximum(    SCIMemoryHighwater );
      int           max_highwater_rank = m_runtime_stats.getRankForMaximum( SCIMemoryHighwater );
      
      if (avg_memused == max_memused && avg_highwater == max_highwater) {
        message << "Memory Use=" << std::setw(8)
                << ProcessInfo::toHumanUnits((unsigned long) avg_memused);

        if (avg_highwater) {
          message << "    Highwater Memory Use=" << std::setw(8)
                  << ProcessInfo::toHumanUnits((unsigned long) avg_highwater);
        }
      }
      else {
        message << "Memory Used=" << std::setw(10)
                << ProcessInfo::toHumanUnits((unsigned long) avg_memused)
                << " (avg) " << std::setw(10)
                << ProcessInfo::toHumanUnits(max_memused)
                << " (max on rank: " << std::setw(6) << max_memused_rank << ")";

        if (avg_highwater) {
          message << "    Highwater Memory Used=" << std::setw(10)
                  << ProcessInfo::toHumanUnits((unsigned long)avg_highwater)
                  << " (avg) " << std::setw(10)
                  << ProcessInfo::toHumanUnits(max_highwater)
                  << " (max on rank: " << std::setw(6) << max_highwater_rank << ")";
        }
      }
    }
    else {
      double  memused   = m_runtime_stats[SCIMemoryUsed];
      double  highwater = m_runtime_stats[SCIMemoryHighwater];
      
      message << "Memory Use=" << std::setw(8)
              << ProcessInfo::toHumanUnits((unsigned long) memused );

      if (highwater) {
        message << "    Highwater Memory Use=" << std::setw(8)
                << ProcessInfo::toHumanUnits((unsigned long) highwater);
      }

      message << " (on rank 0 only)";
    }

    DOUT(reportStats, message.str());
    std::cout.flush();
  }

  // Variable for calculating the percentage of time spent in overhead.
  double percent_overhead = 0;
  
  if ((m_regridder && m_regridder->useDynamicDilation()) || g_comp_stats) {

    // Sum up the average time for overhead related components.
    double overhead_time =
      (m_runtime_stats.getRankAverage(CompilationTime)           +
       m_runtime_stats.getRankAverage(RegriddingTime)            +
       m_runtime_stats.getRankAverage(RegriddingCompilationTime) +
       m_runtime_stats.getRankAverage(RegriddingCopyDataTime)    +
       m_runtime_stats.getRankAverage(LoadBalancerTime));
    
    // Sum up the average times for simulation components.
    double total_time =
      (overhead_time +
       m_runtime_stats.getRankAverage(TaskExecTime)       +
       m_runtime_stats.getRankAverage(TaskLocalCommTime)  +
       m_runtime_stats.getRankAverage(TaskWaitCommTime)   +
       m_runtime_stats.getRankAverage(TaskReduceCommTime) +
       m_runtime_stats.getRankAverage(TaskWaitThreadTime));
    
    // Calculate percentage of time spent in overhead.
    percent_overhead = overhead_time / total_time;
  }
  
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

    if (m_regridder) {
      m_regridder->setOverheadAverage(overheadAverage);
    }
  }

  // Ignore the first sample as that is for initialization.
  if (reportStats && m_num_samples) {

    // Infrastructure proc runtime performance stats.
    if (g_comp_stats && d_myworld->myRank() == 0 ) {
      m_runtime_stats.reportRankSummaryStats( "Runtime Summary ", "",
                                              d_myworld->myRank(),
                                              d_myworld->nRanks(),
                                              m_application->getTimeStep(),
                                              m_application->getSimTime(),
                                              BaseInfoMapper::Dout,
                                              true );

      // Report the overhead percentage.
      if (!std::isnan(overheadAverage)) {
        std::ostringstream message;
        message << "  Percentage of time spent in overhead : "
                << overheadAverage * 100.0;

        // This code is here in case one wants to write to disk the
        // stats. Currently theses are written via Dout.
        if( 1 ) {
          DOUT(true, message.str());
        }
        // else if( 1 ) {
        //   std::ofstream fout;
        //   std::string filename = "Runtime Summary " +
        //     (nRanks != -1 ? "." + std::to_string(nRanks)   : "") +
        //     (rank   != -1 ? "." + std::to_string(rank)     : "") +
        //     (oType == Write_Separate ? "." + std::to_string(timeStep) : "");

        //   if( oType == Write_Append )
        //     fout.open(filename, std::ofstream::out | std::ofstream::app);
        //   else 
        //     fout.open(filename, std::ofstream::out);
            
        //   fout << message.str() << std::endl;
        //   fout.close();
        // }
      }
    }

    // Infrastructure per node runtime performance stats.
    if (g_comp_node_stats && d_myworld->myNode_myRank() == 0 ) {
      m_runtime_stats.reportNodeSummaryStats( ("Runtime Node " + d_myworld->myNodeName()).c_str(), "",
                                              d_myworld->myNode_myRank(),
                                              d_myworld->myNode_nRanks(),
                                              d_myworld->myNode(),
                                              d_myworld->nNodes(),
                                              m_application->getTimeStep(),
                                              m_application->getSimTime(),
                                              BaseInfoMapper::Dout,
                                              true );
    }
    
    // Infrastructure per proc runtime performance stats
    if (g_comp_indv_stats) {
      m_runtime_stats.reportIndividualStats( "Runtime", "",
                                             d_myworld->myRank(),
                                             d_myworld->nRanks(),
                                             m_application->getTimeStep(),
                                             m_application->getSimTime(),
                                             BaseInfoMapper::Dout );
    }

    // Application proc runtime performance stats.
    if (g_app_stats && d_myworld->myRank() == 0) {      
      m_application->getApplicationStats().
        reportRankSummaryStats( "Application Summary", "",
                                d_myworld->myRank(),
                                d_myworld->nRanks(),
                                m_application->getTimeStep(),
                                m_application->getSimTime(),
                                BaseInfoMapper::Dout,
                                false );
    }

    // Application per node runtime performance stats.
    if (g_app_node_stats && d_myworld->myNode_myRank() == 0 ) {
      m_application->getApplicationStats().
        reportNodeSummaryStats( ("Application Node " + d_myworld->myNodeName()).c_str(), "",
                                d_myworld->myNode_myRank(),
                                d_myworld->myNode_nRanks(),
                                d_myworld->myNode(),
                                d_myworld->nNodes(),
                                m_application->getTimeStep(),
                                m_application->getSimTime(),
                                BaseInfoMapper::Dout,
                                false );
    }

    // Application per proc runtime performance stats
    if (g_app_indv_stats) {
      m_application->getApplicationStats().
        reportIndividualStats( "Application", "",
                               d_myworld->myRank(),
                               d_myworld->nRanks(),
                               m_application->getTimeStep(),
                               m_application->getSimTime(),
                               BaseInfoMapper::Dout );
    }
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
                         m_application->getMaterialManagerP()->allMaterials() );
  }
#endif      
}

//______________________________________________________________________
//
void
SimulationController::CheckInSitu( const ProcessorGroup *
                                 , const PatchSubset    *
                                 , const MaterialSubset *
                                 ,       DataWarehouse  *
                                 ,       DataWarehouse  * new_dw 
                                 ,       bool first
                                 )
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

    // Get the wall time if it is needed, otherwise ignore it.
    double walltime;
    
    if( m_application->getWallTimeMax() > 0 )
      walltime = m_wall_timers.GetWallTime();
    else
      walltime = 0;
    
    // Update all of the simulation grid and time dependent variables.
    visit_UpdateSimData( m_visitSimData, m_current_gridP,
                         first, m_application->isLastTimeStep(walltime) );

    // Check the state - if the return value is true the user issued
    // a termination.
    if( visit_CheckState( m_visitSimData ) ) {

      if( new_dw ) {
        m_application->activateReductionVariable( endSimulation_name, true );
        m_application->setReductionVariable( new_dw, endSimulation_name, true );
      }
      
      // Set the max wall time to the current wall time which will
      // cause the simulation to terminate because the next wall time
      // check will be greater.
      else
        m_application->setWallTimeMax( m_wall_timers.GetWallTime() );

      // Disconnect from VisIt.
      visit_EndLibSim( m_visitSimData );
    }

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
