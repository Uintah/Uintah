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

#ifndef CCA_COMPONENTS_SIMULATIONCONTROLLER_SIMULATIONCONTROLLER_H
#define CCA_COMPONENTS_SIMULATIONCONTROLLER_SIMULATIONCONTROLLER_H

#include <Core/Grid/GridP.h>
#include <Core/Grid/LevelP.h>
#include <Core/Grid/SimulationState.h>
#include <Core/Grid/SimulationStateP.h>
#include <Core/Parallel/UintahParallelComponent.h>
#include <Core/ProblemSpec/ProblemSpec.h>
#include <Core/ProblemSpec/ProblemSpecP.h>

#include <CCA/Ports/DataWarehouseP.h>
#include <CCA/Ports/Scheduler.h>
#include <CCA/Ports/SchedulerP.h>

#include <sci_defs/papi_defs.h> // for PAPI performance counters

// Increase overhead size if needed.
#define OVERHEAD_WINDOW 40

namespace Uintah {

class  DataArchive;
class  LoadBalancerPort;
class  Output;
class  Regridder;
class  SimulationInterface;
class  SimulationTime;

/**************************************
      
  CLASS
       SimulationController
      
      
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
     
      
****************************************/

//! The main component that controls the execution of the entire simulation.
class SimulationController : public UintahParallelComponent {

public:

  SimulationController( const ProcessorGroup* myworld, bool doAMR, ProblemSpecP pspec );

  virtual ~SimulationController();

  //! Notifies (before calling run) the SimulationController
  //! that this is simulation is a restart.
  void doRestart( const std::string & restartFromDir
                ,       int           timestep
                ,       bool          fromScratch
                ,       bool          removeOldDir
                );

  bool                 doAMR() { return m_do_amr; }

  //! Execute the simulation
  virtual void run() = 0;

  //  sets simulationController flags
  void setReduceUdaFlags( const std::string& fromDir );
     
  ProblemSpecP         getProblemSpecP() { return m_ups; }

  ProblemSpecP         getGridProblemSpecP() { return m_grid_ps; }

  SimulationStateP     getSimulationStateP() { return m_shared_state; }

  SchedulerP           getSchedulerP() { return m_scheduler; }

  LoadBalancerPort*    getLoadBalancer() { return m_lb; }

  Output*              getOutput() { return m_output; }

  SimulationTime*      getSimulationTime() { return m_time_info; }

  SimulationInterface* getSimulationInterface() { return m_sim; }

  Regridder*           getRegridder() { return m_regridder; }

  DataArchive*         getDataArchive() { return m_archive; }


  bool                 isLast( double time );
    
  void   initWallTimes      ( void );
  void   calcTotalWallTime ( void );
  void   calcExecWallTime  ( void );
  void   calcInSituWallTime( void );

  double getTotalWallTime    ( void );
  double getTotalExecWallTime( void );
  double getExecWallTime     ( void );
  double getExpMovingAverage ( void );
  double getInSituWallTime   ( void );

protected:

  void   setStartSimTime ( double time );

  void preGridSetup();
  GridP gridSetup();
  void postGridSetup( GridP & grid, double & time );

  //! adjust delt based on timeinfo and other parameters
  //    'first' is whether this is the first time adjustDelT is called.
  void adjustDelT( double& delt, double prev_delt, double time );
  void printSimulationStats( int timestep, double next_delt, double prev_delt, double time, bool header = false  );

  void getMemoryStats( int timestep, bool create = false );
  void getPAPIStats  ( );
  
  ProblemSpecP          m_ups{nullptr};
  ProblemSpecP          m_grid_ps{nullptr};         // Problem Spec for the Grid
  SimulationStateP      m_shared_state{nullptr};
  SchedulerP            m_scheduler{nullptr};
  LoadBalancerPort    * m_lb{nullptr};
  Output              * m_output{nullptr};
  SimulationTime      * m_time_info{nullptr};
  SimulationInterface * m_sim{nullptr};
  Regridder           * m_regridder{nullptr};
  DataArchive         * m_archive{nullptr};

  bool m_do_amr{false};
  bool m_do_multi_taskgraphing{false};
  int  m_rad_calc_frequency{1};

  double m_prev_delt{0.0};
  
  /* For restarting */
  bool          m_restarting{false};
  bool          m_reduce_uda{false};
  int           m_restart_timestep{0};
  int           m_restart_index{0};
  int           m_last_recompile_timestep{0};
  std::string   m_from_dir;
      
  // If d_restartFromScratch is true then don't copy or move any of
  // the old timesteps or dat files from the old directory.  Run as
  // as if it were running from scratch but with initial conditions
  // given by the restart checkpoint.
  bool m_restart_from_scratch{false};

  // If !d_restartFromScratch, then this indicates whether to move
  // or copy the old timesteps.
  bool m_restart_remove_old_dir{false};


#ifdef USE_PAPI_COUNTERS

  struct PapiEvent {
    bool         isSupported;
    int          eventValueIndex;
    std::string  name;
    std::string  simStatName;

    PapiEvent( const std::string& _name, const std::string& _simStatName )
      : name(_name), simStatName(_simStatName)
    {
      eventValueIndex = 0;
      isSupported     = false;
    }
  };

  int         m_event_set{0};            // PAPI event set
  long long * m_event_values{0};         // PAPI event set values

  std::map<int, PapiEvent>   m_papi_events;
  std::map<int, std::string> m_papi_error_codes;
#endif


private:

  // eliminate copy, assignment and move
  SimulationController( const SimulationController & )            = delete;
  SimulationController& operator=( const SimulationController & ) = delete;
  SimulationController( SimulationController && )                 = delete;
  SimulationController& operator=( SimulationController && )      = delete;


  // Percent time in overhead samples
  double m_overhead_values[OVERHEAD_WINDOW];
  double m_overhead_weights[OVERHEAD_WINDOW];
  int    m_overhead_index{0};  // Next sample for writing

  int    m_num_samples{0};
  
  double m_start_sim_time{0.0};         // starting sim time
  double m_start_wall_time{0.0};        // starting wall time
  double m_total_wall_time{0.0};        // total wall time
  double m_total_exec_wall_time{0.0};   // total execution wall time for all time steps
  double m_exec_wall_time{0.0};         // execution wall time for last time step
  double m_insitu_wall_time{0.0};       // in-situ wall time
  
  // For calculating an exponential moving average of the execution wall time
  double m_exp_moving_average{0.0};
  
};

} // namespace Uintah

#endif // CCA_COMPONENTS_SIMULATIONCONTROLLER_SIMULATIONCONTROLLER_H
