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

#ifndef CCA_COMPONENTS_SIMULATIONCONTROLLER_SIMULATIONCONTROLLER_H
#define CCA_COMPONENTS_SIMULATIONCONTROLLER_SIMULATIONCONTROLLER_H

#include <CCA/Components/Schedulers/RuntimeStatsEnum.h>

#include <Core/Grid/GridP.h>
#include <Core/Grid/LevelP.h>
#include <Core/Parallel/UintahParallelComponent.h>
#include <Core/ProblemSpec/ProblemSpec.h>
#include <Core/ProblemSpec/ProblemSpecP.h>
#include <Core/Util/InfoMapper.h>
#include <Core/Util/Timers/Timers.hpp>

#include <CCA/Ports/DataWarehouseP.h>
#include <CCA/Ports/Scheduler.h>
#include <CCA/Ports/SchedulerP.h>

#include <sci_defs/visit_defs.h>

#ifdef HAVE_VISIT
#  include <VisIt/libsim/visit_libsim.h>
#endif

// Window size for the overhead calculation
#define OVERHEAD_WINDOW 40

// Window size for the exponential moving average
#define AVERAGE_WINDOW 10


namespace Uintah {

class  DataArchive;
class  LoadBalancer;
class  Output;
class  Regridder;
class  ApplicationInterface;
class  SimulationTime;

/**************************************

 CLASS
   WallTimer

 KEYWORDS
   Util, Wall Timers

 DESCRIPTION
   Utility class to manage the Wall Time.

 ****************************************/

class WallTimers {

public:

  WallTimers() { m_num_samples = 0; m_wall_timer.start(); };

public:

  Timers::Simple TimeStep;           // Total time for all time steps
  Timers::Simple ExpMovingAverage;   // Execution exponential moving average
                                     // for N time steps.
  Timers::Simple InSitu;             // In-situ time for previous time step

  int    getWindow( void ) { return AVERAGE_WINDOW; };
  void resetWindow( void ) { m_num_samples = 0; };
  
  Timers::nanoseconds updateExpMovingAverage( void ) {

    Timers::nanoseconds laptime = TimeStep.lap();
    
    // Ignore the first sample as that is for initialization.
    if( m_num_samples ) {
      // Calculate the exponential moving average for this time step.
      // Multiplier: (2 / (Time periods + 1) )
      // EMA: {current - EMA(previous)} x multiplier + EMA(previous).
      
      double mult = 2.0 / ((double) std::min(m_num_samples, AVERAGE_WINDOW) + 1.0);
      
      ExpMovingAverage = mult * laptime + (1.0-mult) * ExpMovingAverage();
    }
    else {
      ExpMovingAverage = laptime;
    }
      
    ++m_num_samples;

    return laptime;

  } // end Timers::nanoseconds

  double GetWallTime() { return m_wall_timer().seconds(); };

private:

  int              m_num_samples{0};  // Number of samples for the moving average
  Timers::Simple   m_wall_timer{};
};


/**************************************
      
  CLASS
       SimulationController
      
       Short description...
      
  GENERAL INFORMATION
      
       SimulationController.h
      
       Steven G. Parker
       Department of Computer Science
       University of Utah
      
       Center for the Simulation of Accidental Fires and Explosions (C-SAFE)
       
             
  KEYWORDS
       Simulation_Controller
      
  DESCRIPTION
       Abstract base class for the SimulationControllers.
       Introduced to make the "old" SimulationController
       and the new AMRSimulationController interchangeable.
     
  WARNING
      
****************************************/

//! The main component that controls the execution of the 
//! entire simulation. 
class SimulationController : public UintahParallelComponent {

public:

  SimulationController( const ProcessorGroup * myworld, ProblemSpecP pspec );

  virtual ~SimulationController();

  // Methods for managing the components attached via the ports.
  virtual void setComponents( UintahParallelComponent *comp ) {};
  virtual void getComponents();
  virtual void releaseComponents();

  //! Notifies (before calling run) the SimulationController
  //! that this is simulation is a restart.
  void doRestart( const std::string & restartFromDir
                ,       int           index
                ,       bool          fromScratch
                ,       bool          removeOldDir
                );

  //! Execute the simulation
  virtual void run() = 0;

  //  sets simulationController flags
  void setPostProcessFlags();
     
  ProblemSpecP          getProblemSpecP() { return m_ups; }
  ProblemSpecP          getGridProblemSpecP() { return m_grid_ps; }
  SchedulerP            getSchedulerP() { return m_scheduler; }
  LoadBalancer*         getLoadBalancer() { return m_loadBalancer; }
  Output*               getOutput() { return m_output; }
  ApplicationInterface* getApplicationInterface() { return m_application; }
  Regridder*            getRegridder() { return m_regridder; }
  WallTimers*           getWallTimers() { return &m_wall_timers; }

  bool getRecompileTaskGraph() const { return m_recompile_taskgraph; }
  void setRecompileTaskGraph(bool val) { m_recompile_taskgraph = val; }

  void ScheduleReportStats( bool header );
  void ReportStats( const ProcessorGroup *
                  , const PatchSubset    *
                  , const MaterialSubset *
                  ,       DataWarehouse  *
                  ,       DataWarehouse  *
                  ,       bool header
                  );

  ReductionInfoMapper< RuntimeStatsEnum, double > & getRuntimeStats()
  { return m_runtime_stats; };

protected:

  void restartArchiveSetup();
  void outputSetup();
  void gridSetup();
  void regridderSetup();
  void loadBalancerSetup();
  void applicationSetup();
  void schedulerSetup();
  void timeStateSetup();
  void finalSetup();
  void ResetStats( void );

  void getMemoryStats( bool create = false );
  
  ProblemSpecP           m_ups           {nullptr};
  ProblemSpecP           m_grid_ps       {nullptr};
  ProblemSpecP           m_restart_ps    {nullptr};
  GridP                  m_current_gridP {nullptr};

  ApplicationInterface * m_application   {nullptr};
  LoadBalancer         * m_loadBalancer  {nullptr};
  Output               * m_output        {nullptr};
  Regridder            * m_regridder     {nullptr};
  SchedulerP             m_scheduler     {nullptr};

  // Only used when restarting: Data from checkpoint UDA.
  DataArchive          * m_restart_archive {nullptr};

  bool m_do_multi_taskgraphing{false};
    
  WallTimers m_wall_timers;

  // Used when restarting.
  bool        m_restarting{false};
  std::string m_from_dir;
  int         m_restart_timestep{0};
  int         m_restart_index{0};
  int         m_last_recompile_timeStep{0};
  bool        m_post_process_uda{false};
      
  // If m_restart_from_scratch is true then don't copy or move any of
  // the old time steps or dat files from the old directory.  Run as
  // as if it were running from scratch but with initial conditions
  // given by the restart checkpoint.
  bool m_restart_from_scratch{false};

  // If not m_restart_from_scratch, then this indicates whether to move
  // or copy the old time steps.
  bool m_restart_remove_old_dir{false};

  bool m_recompile_taskgraph{false};
  
  // Runtime stat mappers.
  ReductionInfoMapper< RuntimeStatsEnum, double > m_runtime_stats;


public:

  void ScheduleCheckInSitu( bool header );
  void CheckInSitu( const ProcessorGroup *
                  , const PatchSubset    *
                  , const MaterialSubset *
                  ,       DataWarehouse  *
                  ,       DataWarehouse  *
                  ,       bool first
                  );

#ifdef HAVE_VISIT
  void setVisIt( unsigned int val ) { m_do_visit = val; }
  unsigned int  getVisIt() { return m_do_visit; }

protected:
  unsigned int m_do_visit;
  visit_simulation_data *m_visitSimData;
#endif


private:

  // For reporting stats Frequency > OnTimeStep
  unsigned int m_reportStatsFrequency {1};
  unsigned int m_reportStatsOnTimeStep {0};
  
  // Percent time in overhead samples
  double m_overhead_values[OVERHEAD_WINDOW];
  double m_overhead_weights[OVERHEAD_WINDOW];
  int    m_overhead_index{0}; // Next sample for writing
  int    m_num_samples{0};

  // eliminate copy, assignment and move
  SimulationController( const SimulationController & )            = delete;
  SimulationController& operator=( const SimulationController & ) = delete;
  SimulationController( SimulationController && )                 = delete;
  SimulationController& operator=( SimulationController && )      = delete;
};

} // end namespace Uintah

#endif // CCA_COMPONENTS_SIMULATIONCONTROLLER_SIMULATIONCONTROLLER_H
