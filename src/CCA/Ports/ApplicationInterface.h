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

#ifndef UINTAH_HOMEBREW_APPLICATIONINTERFACE_H
#define UINTAH_HOMEBREW_APPLICATIONINTERFACE_H

#include <CCA/Ports/Output.h>
#include <CCA/Ports/SchedulerP.h>

#include <Core/Parallel/UintahParallelPort.h>
#include <Core/Grid/GridP.h>
#include <Core/Grid/LevelP.h>
#include <Core/Grid/SimulationStateP.h>
#include <Core/Grid/Variables/ComputeSet.h>
#include <Core/Grid/Variables/VarTypes.h>
#include <Core/ProblemSpec/ProblemSpecP.h>

#include <sci_defs/visit_defs.h>

namespace Uintah {

/**************************************

CLASS
   ApplicationInterface
   
   Short description...

GENERAL INFORMATION

   ApplicationInterface.h

   Steven G. Parker
   Department of Computer Science
   University of Utah

   Center for the Simulation of Accidental Fires and Explosions (C-SAFE)
  
   
KEYWORDS
   Simulation_Interface

DESCRIPTION
   Long description...
  
WARNING
  
****************************************/

  class UintahParallelComponent;
  
  class DataWarehouse;

  class SimulationTime;
  class VarLabel;

  class DebugStream;
  class Dout;
  

  class ApplicationInterface : public UintahParallelPort {

    // NOTE: ONLY CS infrasture should be a friend class - the actual
    // applictions should NOT have access to private member methods as
    // they should get those values via the Data Warehouse.
    friend class SimulationController;
    friend class AMRSimulationController;
    friend class DataArchiver;

    friend class LoadBalancersCommon;
    friend class DynamicLoadBalancer;
    friend class ParticleLoadBalancer;

    friend class Switcher;
    
  public:
    ApplicationInterface();
    virtual ~ApplicationInterface();
    
    // Methods for managing the components attached via the ports.
    virtual void setComponents( UintahParallelComponent *comp ) = 0;
    virtual void getComponents() = 0;
    virtual void releaseComponents() = 0;

    virtual Scheduler *getScheduler() = 0;
    virtual Output    *getOutput()    = 0;
    
    // Top level problem set up called by sus.
    virtual void problemSetup( const ProblemSpecP &prob_spec ) = 0;
    
    // Top level problem set up called by simulation controller.
    virtual void problemSetup( const ProblemSpecP     & params,
                               const ProblemSpecP     & restart_prob_spec,
                                     GridP            & grid ) = 0;

    // Called to add missing grid based UPS specs.
    virtual void preGridProblemSetup( const ProblemSpecP & params, 
                                            GridP        & grid ) = 0;

    // Used to write parts of the problem spec.
    virtual void outputProblemSpec( ProblemSpecP & ps ) = 0;

    // Schedule the inital setup of the problem.
    virtual void scheduleInitialize( const LevelP     & level,
                                           SchedulerP & scheduler ) = 0;
                                 
    // On a restart schedule an initialization task.
    virtual void scheduleRestartInitialize( const LevelP     & level,
                                                  SchedulerP & scheduler ) = 0;

    // restartInitialize() is called once and only once if and when a
    // simulation is restarted.  This allows the simulation component
    // to handle initializations that are necessary when a simulation
    // is restarted.
    virtual void restartInitialize() = 0;

    // Ask the application which primary task graph it wishes to
    // execute this time step, this will be an index into the
    // scheduler's vector of task-graphs.
    virtual int computeTaskGraphIndex() = 0;

    // Schedule the inital switching.
    virtual void scheduleSwitchInitialization( const LevelP     & level,
					             SchedulerP & sched ) = 0;
      
    virtual void scheduleSwitchTest( const LevelP &     level,
                                           SchedulerP & scheduler ) = 0;

    // Schedule the actual time step advencement tasks.
    virtual void scheduleTimeAdvance( const LevelP     & level,
				            SchedulerP & scheduler ) = 0;

    // Optionally schedule tasks that can not be done in scheduleTimeAdvance.
    virtual void scheduleFinalizeTimestep( const LevelP     & level,
					         SchedulerP & scheduler ) = 0;

    // Optionally schedule analysis tasks.
    virtual void scheduleAnalysis( const LevelP     & level,
				         SchedulerP & scheduler) = 0;

    // Optionally schedule a task that determines the next delt T value.
    virtual void scheduleComputeStableTimeStep( const LevelP & level,
                                                SchedulerP   & scheduler ) = 0;
      
    // Reduce the system wide values such as the next delta T.
    virtual void scheduleReduceSystemVars(const GridP      & grid,
					  const PatchSet   * perProcPatchSet,
					        SchedulerP & scheduler) = 0;

    // Schedule the initialization of system values such at the time step.
    virtual void scheduleInitializeSystemVars(const GridP      & grid,
					      const PatchSet   * perProcPatchSet,
					            SchedulerP & scheduler) = 0;
    
    // Schedule the updating of system values such at the time step.
    virtual void scheduleUpdateSystemVars(const GridP      & grid,
					  const PatchSet   * perProcPatchSet,
					        SchedulerP & scheduler) = 0;
    
    // Methods used for scheduling AMR regridding.
    virtual void scheduleRefine( const PatchSet   * patches,
				       SchedulerP & scheduler ) = 0;
    
    virtual void scheduleRefineInterface( const LevelP     & fineLevel, 
                                                SchedulerP & scheduler,
                                                bool         needCoarseOld,
                                                bool         needCoarseNew ) = 0;

    virtual void scheduleCoarsen( const LevelP     & coarseLevel, 
                                        SchedulerP & scheduler ) = 0;

    // Schedule to mark flags for AMR regridding
    virtual void scheduleErrorEstimate( const LevelP     & coarseLevel,
					      SchedulerP & scheduler) = 0;

    // Schedule to mark initial flags for AMR regridding
    virtual void scheduleInitialErrorEstimate(const LevelP     & coarseLevel,
                                                    SchedulerP & sched) = 0;

    // Used to get the progress ratio of an AMR regridding subcycle.
    virtual double getSubCycleProgress(DataWarehouse* fineNewDW) = 0;


    // Recompute a time step if current time advance is not
    // converging.  The returned time is the new delta T.
    virtual void   recomputeTimeStep() = 0;
    virtual double recomputeTimeStep( double delt ) = 0;
    virtual bool restartableTimeSteps() = 0;

    // Updates the tiem step and the delta T.
    virtual void prepareForNextTimeStep( const GridP & grid ) = 0;

    // Asks the application if it needs to be recompiled.
    virtual bool needRecompile( const GridP & grid ) = 0;

    // Labels for access value in the data warehouse.
    virtual const VarLabel* getTimeStepLabel() const = 0;
    virtual const VarLabel* getSimTimeLabel() const = 0;
    virtual const VarLabel* getDelTLabel() const = 0;

    //////////
    virtual void setModelMaker(bool val) = 0;
    virtual bool needModelMaker() const = 0;
    
    virtual void setAMR(bool val) = 0;
    virtual bool isAMR() const = 0;
  
    virtual void setLockstepAMR(bool val) = 0;
    virtual bool isLockstepAMR() const = 0;

    virtual void setDynamicRegridding(bool val) = 0;
    virtual bool isDynamicRegridding() const = 0;
  
    // Boolean for vars chanegd by the in-situ.
    virtual void haveModifiedVars( bool val ) = 0;
    virtual bool haveModifiedVars() const = 0;
     
    //////////
    virtual bool isRegridTimeStep() const = 0;
    virtual void setRegridTimeStep(bool ans) = 0;
    
    // Some applications can adjust the output interval.
    virtual void adjustOutputInterval(bool ans) = 0;
    virtual bool adjustOutputInterval() const = 0;
     
    // Some applications can adjust the checkpoint interval.
    virtual void adjustCheckpointInterval(bool ans) = 0;
    virtual bool adjustCheckpointInterval() const = 0;

    // Some applications can end the simulation early.
    virtual void mayEndSimulation(bool ans) = 0;
    virtual bool mayEndSimulation() const = 0;

    // Access methods for member classes.
    virtual SimulationTime * getSimulationTime() const = 0;
    virtual SimulationStateP getSimulationStateP() const = 0;

    // The member methods are private as the child application should
    // ONLY get/set these values via the data warehouse.
  private:
    virtual   void setDelT( double val ) = 0;
    virtual double getDelT() const = 0;
    virtual   void setDelTForAllLevels( SchedulerP& scheduler,
					const GridP & grid,
					const int totalFine ) = 0;

    virtual   void setNextDelT( double val ) = 0;
    virtual double getNextDelT() const = 0;
    
    virtual   void setSimTime( double val ) = 0;
    virtual double getSimTime() const = 0;
    
    virtual   void setSimTimeStart( double val ) = 0;
    virtual double getSimTimeStart() const = 0;
    
    // Returns the integer time step index of the simulation.  All
    // simulations start with a time step number of 0.  This value is
    // incremented by one for each simulation time step processed.

    // The 'set' function should only be called by the
    // SimulationController at the beginning of a simulation.  The
    // 'increment' function is called by the SimulationController at
    // the begining of each time step.
    virtual void setTimeStep( int ts ) = 0;
    virtual void incrementTimeStep( const GridP & grid ) = 0;
    virtual int  getTimeStep() const = 0;

    virtual bool isLastTimeStep( double walltime ) const = 0;
    virtual bool maybeLastTimeStep( double walltime ) const = 0;

  private:
    ApplicationInterface(const ApplicationInterface&);
    ApplicationInterface& operator=(const ApplicationInterface&);
  
#ifdef HAVE_VISIT
  public:
    // Reduction analysis variables for on the fly analysis
    struct analysisVar {
      std::string name;
      int matl;
      int level;
      std::vector< const VarLabel* > labels;
    };
  
    virtual std::vector< analysisVar > & getAnalysisVars() = 0;

    // Interactive variables from the UPS problem spec or other state
    // variables.
    struct interactiveVar {
      std::string name;
      TypeDescription::Type type;
      void * value;
      bool   modifiable {false}; // If true the variable maybe modified.
      bool   modified   {false}; // If true the variable was modified.
      bool   recompile  {false}; // If true and the variable was modified
      // recompile the task graph.
      double range[2];   // If modifiable min/max range of acceptable values.
    };
  
    // Interactive variables from the UPS problem spec.
    virtual std::vector< interactiveVar > & getUPSVars() = 0;

    // Interactive state variables from components.
    virtual std::vector< interactiveVar > & getStateVars() = 0;
     
    // Debug streams that can be turned on or off.
    virtual std::vector< DebugStream * > & getDebugStreams() = 0;
    virtual std::vector< Dout * > & getDouts() = 0;
  
    virtual void setVisIt( unsigned int val ) = 0;
    virtual unsigned int  getVisIt() = 0;
#endif
  };
} // End namespace Uintah

#endif
