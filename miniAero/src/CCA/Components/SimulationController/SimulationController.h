/*
 * The MIT License
 *
 * Copyright (c) 1997-2015 The University of Utah
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

#include <sci_defs/papi_defs.h> // for PAPI performance counters

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

namespace Uintah {

class  DataArchive;
class  LoadBalancer;
class  Output;
class  Regridder;
class  SimulationInterface;
struct SimulationTime;

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
     
protected:

  double getWallTime     ( void );
  void   calcWallTime    ( void );

  double getStartTime    ( void );
  void   calcStartTime   ( void );
  void   setStartSimTime ( double t );

  void preGridSetup();
  GridP gridSetup();
  void postGridSetup( GridP& grid, double& t);

  //! adjust delt based on timeinfo and other parameters
  //    'first' is whether this is the first time adjustDelT is called.
  void adjustDelT( double& delt, double prev_delt, bool first, double t );
  void initSimulationStatsVars ( void );
  void printSimulationStats    ( int timestep, double delt, double time );

  ProblemSpecP         d_ups;
  ProblemSpecP         d_grid_ps;         // Problem Spec for the Grid
  SimulationStateP     d_sharedState;
  SchedulerP           d_scheduler;
  LoadBalancer*        d_lb;
  Output*              d_output;
  SimulationTime*      d_timeinfo;
  SimulationInterface* d_sim;
  Regridder*           d_regridder;
  DataArchive*         d_archive;

  bool d_doAMR;
  bool d_doMultiTaskgraphing;

  /* For restarting */
  bool        d_restarting;
  std::string d_fromDir;
  int         d_restartTimestep;
  int         d_restartIndex;
  int         d_lastRecompileTimestep;
  bool        d_reduceUda;
      
  // If d_restartFromScratch is true then don't copy or move any of
  // the old timesteps or dat files from the old directory.  Run as
  // as if it were running from scratch but with initial conditions
  // given by the restart checkpoint.
  bool d_restartFromScratch;

  // If !d_restartFromScratch, then this indicates whether to move
  // or copy the old timesteps.
  bool d_restartRemoveOldDir;

#ifdef USE_PAPI_COUNTERS
  int         d_eventSet;            // PAPI event set
  long long * d_eventValues;         // PAPI event set values
  struct PapiEvent {
    int         eventValueIndex;
    std::string name;
    std::string simStatName;
    bool        isSupported;
    PapiEvent( const std::string& _name, const std::string& _simStatName )
      : name(_name), simStatName(_simStatName)
    {
      eventValueIndex = 0;
      isSupported = false;
    }
  };
  std::map<int, PapiEvent>   d_papiEvents;
  std::map<int, std::string> d_papiErrorCodes;
#endif

private:

  int    d_n;
  double d_wallTime;              // current wall time
  double d_startTime;             // starting wall time
  double d_startSimTime;          // starting sim time
  double d_prevWallTime;
  //double d_sumOfWallTimes;
  //double d_sumOfWallTimeSquares;
     
  // this is for calculating an exponential moving average
  double d_movingAverage;

  // void problemSetup( const ProblemSpecP&, GridP& ) = 0;
  // bool needRecompile( double t, double delt, const LevelP& level,
  //                     SimulationInterface* cfd, Output* output,
  //                     LoadBalancer* lb ) = 0;
  // SimulationController(const SimulationController&) = 0;
  // SimulationController& operator=(const SimulationController&) = 0;

};

} // End namespace Uintah

#endif
