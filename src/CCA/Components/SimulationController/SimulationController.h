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

#ifndef UINTAH_HOMEBREW_SIMULATIONCONTROLLER_H
#define UINTAH_HOMEBREW_SIMULATIONCONTROLLER_H

#include <Core/Grid/GridP.h>
#include <Core/Grid/LevelP.h>
#include <Core/Grid/SimulationState.h>
#include <Core/Grid/SimulationStateP.h>
#include <Core/Parallel/UintahParallelComponent.h>
#include <Core/ProblemSpec/ProblemSpec.h>
#include <Core/ProblemSpec/ProblemSpecP.h>
#include <Core/Util/Timers/Timers.hpp>

#include <CCA/Ports/DataWarehouseP.h>
#include <CCA/Ports/Scheduler.h>
#include <CCA/Ports/SchedulerP.h>

#include <sci_defs/visit_defs.h>
#include <sci_defs/papi_defs.h> // for PAPI performance counters

#ifdef HAVE_VISIT
#  include <VisIt/libsim/visit_libsim.h>
#endif

// Window size for the overhead calculation
#define OVERHEAD_WINDOW 40

// Window size for the exponential moving average
#define AVERAGE_WINDOW 10

namespace Uintah {

class  DataArchive;
class  LoadBalancerPort;
class  Output;
class  Regridder;
class  SimulationInterface;
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
  WallTimers() { d_nSamples = 0; d_wallTimer.start(); };

public:

  Timers::Simple TimeStep;           // Total time for all time steps
  Timers::Simple ExpMovingAverage;   // Execution exponential moving average
                                     // for N time steps.
  Timers::Simple InSitu;             // In-situ time for previous time step

  int    getWindow( void ) { return AVERAGE_WINDOW; };
  void resetWindow( void ) { d_nSamples = 0; };
  
  Timers::nanoseconds updateExpMovingAverage( void )
  {
    Timers::nanoseconds laptime = TimeStep.lap();
    
    // Ignore the first sample as that is for initalization.
    if( d_nSamples )
    {
      // Calulate the exponential moving average for this time step.
      // Multiplier: (2 / (Time periods + 1) )
      // EMA: {current - EMA(previous)} x multiplier + EMA(previous).
      
      double mult = 2.0 / ((double) std::min(d_nSamples, AVERAGE_WINDOW) + 1.0);
      
      ExpMovingAverage = mult * laptime + (1.0-mult) * ExpMovingAverage();
    }
    else
      ExpMovingAverage = laptime;
      
    ++d_nSamples;

    return laptime;
  }

  double GetWallTime() { return d_wallTimer().seconds(); };

private:
  int d_nSamples;        // Number of samples for the moving average

  Timers::Simple d_wallTimer;
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
       Abstract baseclass for the SimulationControllers.
       Introduced to make the "old" SimulationController
       and the new AMRSimulationController interchangeable.
     
  WARNING
      
****************************************/

//! The main component that controls the execution of the 
//! entire simulation. 
class SimulationController : public UintahParallelComponent {

public:
  SimulationController( const ProcessorGroup* myworld, bool doAMR, ProblemSpecP pspec );
  virtual ~SimulationController();

  //! Notifies (before calling run) the SimulationController
  //! that this is simulation is a restart.
  void doRestart( const std::string& restartFromDir,
                  int           timestep,
                  bool          fromScratch,
                  bool          removeOldDir );

  //! Execute the simulation
  virtual void run() = 0;

  //  sets simulationController flags
  void setReduceUdaFlags( const std::string& fromDir );
     
  ProblemSpecP         getProblemSpecP() { return d_ups; }
  ProblemSpecP         getGridProblemSpecP() { return d_grid_ps; }
  SimulationStateP     getSimulationStateP() { return d_sharedState; }
  SchedulerP           getSchedulerP() { return d_scheduler; }
  LoadBalancerPort*    getLoadBalancer() { return d_lb; }
  Output*              getOutput() { return d_output; }
  SimulationTime*      getSimulationTime() { return d_timeinfo; }
  SimulationInterface* getSimulationInterface() { return d_sim; }
  Regridder*           getRegridder() { return d_regridder; }

  bool                 doAMR() { return d_doAMR; }

  WallTimers*          getWallTimers() { return &walltimers; }

protected:

  bool isLast( void );
  bool maybeLast( void );
    
  void restartArchiveSetup();
  void outputSetup();
  void schedulerSetup();
  void simulationInterfaceSetup();
  void gridSetup();
  void regridderSetup();
  void loadBalancerSetup();
  void outOfSyncSetup();
  void timeStateSetup();
  void finalSetup();

  // Get the next delta T
  void getNextDeltaT( void );

  void ReportStats( bool first );     
  void getMemoryStats( bool create = false );
  void getPAPIStats  ( );
  
  ProblemSpecP         d_ups{nullptr};
  ProblemSpecP         d_grid_ps{nullptr};       // Problem Spec for the Grid
  ProblemSpecP         d_restart_ps{nullptr};    // Problem Spec for restarting
  SimulationStateP     d_sharedState{nullptr};
  SchedulerP           d_scheduler{nullptr};
  LoadBalancerPort*    d_lb{nullptr};
  Output*              d_output{nullptr};
  SimulationTime*      d_timeinfo{nullptr};
  SimulationInterface* d_sim{nullptr};
  Regridder*           d_regridder{nullptr};
  DataArchive*         d_restart_archive{nullptr};     // Only used when restarting: Data from UDA we are restarting from.

  GridP                d_currentGridP;

  bool d_doAMR{false};
  bool d_do_multi_taskgraphing{false};
  int  d_rad_calc_frequency{1};

  double d_delt{0.0};
  double d_prev_delt{0.0};
  
  double d_simTime{0.0};               // current sim time
  double d_startSimTime{0.0};          // starting sim time
  
  WallTimers walltimers;

  /* For restarting */
  bool        d_restarting{false};
  std::string d_fromDir;
  int         d_restartTimestep{0};
  int         d_restartIndex{0};
  int         d_last_recompile_timestep{0};
  bool        d_reduceUda{false};
      
  // If d_restartFromScratch is true then don't copy or move any of
  // the old timesteps or dat files from the old directory.  Run as
  // as if it were running from scratch but with initial conditions
  // given by the restart checkpoint.
  bool d_restartFromScratch{false};

  // If !d_restartFromScratch, then this indicates whether to move
  // or copy the old timesteps.
  bool d_restartRemoveOldDir{false};

#ifdef USE_PAPI_COUNTERS
  int         m_papi_event_set;            // PAPI event set
  long long * m_papi_event_values;         // PAPI event set values

  struct PapiEvent {
    bool                           m_is_supported{false};
    int                            m_event_value_idx{0};
    std::string                    m_name{""};
    SimulationState::RunTimeStat   m_sim_stat_name{};

    PapiEvent( const std::string                  & name
             , const SimulationState::RunTimeStat & sim_stat_name )
      : m_name(name)
      , m_sim_stat_name(sim_stat_name)
    { }
  };

  std::map<int, PapiEvent>   m_papi_events;
#endif

#ifdef HAVE_VISIT
  bool CheckInSitu( visit_simulation_data *visitSimData, bool first );
#endif     
private:

  // eliminate copy, assignment and move
  SimulationController( const SimulationController & )            = delete;
  SimulationController& operator=( const SimulationController & ) = delete;
  SimulationController( SimulationController && )                 = delete;
  SimulationController& operator=( SimulationController && )      = delete;

  // Percent time in overhead samples
  double overheadValues[OVERHEAD_WINDOW];
  double overheadWeights[OVERHEAD_WINDOW];
  int    overheadIndex{0}; // Next sample for writing

  int    d_nSamples{0};

};

} // End namespace Uintah

#endif // CCA_COMPONENTS_SIMULATIONCONTROLLER_SIMULATIONCONTROLLER_H
