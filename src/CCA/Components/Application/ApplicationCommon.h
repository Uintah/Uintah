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

#ifndef UINTAH_HOMEBREW_APPLICATIONCOMMON_H
#define UINTAH_HOMEBREW_APPLICATIONCOMMON_H

#include <Core/Parallel/UintahParallelComponent.h>
#include <CCA/Ports/ApplicationInterface.h>
#include <CCA/Ports/LoadBalancer.h>
#include <CCA/Ports/SchedulerP.h>
#include <CCA/Ports/SolverInterface.h>

#include <Core/Grid/Variables/ComputeSet.h>
#include <Core/Grid/GridP.h>
#include <Core/Grid/LevelP.h>
#include <Core/Grid/SimulationStateP.h>
#include <Core/Grid/SimulationState.h>
#include <Core/Grid/SimulationTime.h>
#include <Core/OS/Dir.h>
#include <Core/Util/Handle.h>
#include <Core/ProblemSpec/ProblemSpecP.h>

#include <sci_defs/visit_defs.h>

namespace Uintah {

  class DataWarehouse;  
  class Output;
  class Regridder;

/**************************************

CLASS
   ApplicationCommon
   
   Short description...

GENERAL INFORMATION

   ApplicationCommon.h

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

  class ApplicationCommon : public UintahParallelComponent,
                            public ApplicationInterface {

    friend class Switcher;

  public:
    ApplicationCommon(const ProcessorGroup* myworld,
                      const SimulationStateP sharedState);

    virtual ~ApplicationCommon();
  
    // Methods for managing the components attached via the ports.
    virtual void setComponents( UintahParallelComponent *comp );
    virtual void getComponents();
    virtual void releaseComponents();

    virtual Scheduler *getScheduler() { return m_scheduler; }
    virtual Regridder *getRegridder() { return m_regridder; }
    virtual Output    *getOutput()    { return m_output; }
    
    // Top level problem set up called by sus.
    virtual void problemSetup( const ProblemSpecP &prob_spec );
    
    // Top level problem set up called by simulation controller.
    virtual void problemSetup( const ProblemSpecP     & params,
                               const ProblemSpecP     & restart_prob_spec,
                                     GridP            & grid ) = 0;

    // Called to add missing grid based UPS specs.
    virtual void preGridProblemSetup( const ProblemSpecP & params, 
                                            GridP        & grid ) {};

    // Used to write parts of the problem spec.
    virtual void outputProblemSpec( ProblemSpecP & ps ) {};
      
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
    virtual void restartInitialize() {}

    // Ask the application which primary task graph it wishes to
    // execute this time step, this will be an index into the
    // scheduler's vector of task-graphs.
    virtual int computeTaskGraphIndex();
    virtual int computeTaskGraphIndex( const int timeStep ) { return 0; }

    // Schedule the inital switching.
    virtual void scheduleSwitchInitialization( const LevelP     & level,
                                                     SchedulerP & sched ) {}
      
    virtual void scheduleSwitchTest( const LevelP &     level,
                                           SchedulerP & scheduler ) {};

    // Schedule the actual time step advencement tasks.
    virtual void scheduleTimeAdvance( const LevelP     & level,
                                            SchedulerP & scheduler );

    // Optionally schedule tasks that can not be done in scheduleTimeAdvance.
    virtual void scheduleFinalizeTimestep( const LevelP     & level,
                                                 SchedulerP & scheduler ) {}

    // Optionally schedule analysis tasks.
    virtual void scheduleAnalysis( const LevelP     & level,
                                         SchedulerP & scheduler) {}

    // Optionally schedule a task that determines the next delt T value.
    virtual void scheduleComputeStableTimeStep( const LevelP & level,
                                                SchedulerP   & scheduler ) = 0;
      
    // Reduce the system wide values such as the next delta T.
    virtual void scheduleReduceSystemVars(const GridP      & grid,
                                          const PatchSet   * perProcPatchSet,
                                                SchedulerP & scheduler);

    void reduceSystemVars( const ProcessorGroup *,
                           const PatchSubset    * patches,
                           const MaterialSubset * /*matls*/,
                                 DataWarehouse  * /*old_dw*/,
                                 DataWarehouse  * new_dw );
      
    // Schedule the initialization of system values such at the time step.
    virtual void scheduleInitializeSystemVars(const GridP      & grid,
                                              const PatchSet   * perProcPatchSet,
                                                    SchedulerP & scheduler);
    
    void initializeSystemVars( const ProcessorGroup *,
                               const PatchSubset    * patches,
                               const MaterialSubset * /*matls*/,
                                     DataWarehouse  * /*old_dw*/,
                                     DataWarehouse  * new_dw );
    
    // Schedule the updating of system values such at the time step.
    virtual void scheduleUpdateSystemVars(const GridP      & grid,
                                          const PatchSet   * perProcPatchSet,
                                                SchedulerP & scheduler);
    
    virtual void updateSystemVars( const ProcessorGroup *,
                                   const PatchSubset    * patches,
                                   const MaterialSubset * /*matls*/,
                                         DataWarehouse  * /*old_dw*/,
                                         DataWarehouse  * new_dw );
    
    // Methods used for scheduling AMR regridding.
    virtual void scheduleRefine( const PatchSet   * patches,
                                       SchedulerP & scheduler );
    
    virtual void scheduleRefineInterface( const LevelP     & fineLevel, 
                                                SchedulerP & scheduler,
                                                bool         needCoarseOld,
                                                bool         needCoarseNew );

    virtual void scheduleCoarsen( const LevelP     & coarseLevel, 
                                        SchedulerP & scheduler );

    // Schedule to mark flags for AMR regridding
    virtual void scheduleErrorEstimate( const LevelP     & coarseLevel,
                                              SchedulerP & scheduler);

    // Schedule to mark initial flags for AMR regridding
    virtual void scheduleInitialErrorEstimate(const LevelP     & coarseLevel,
                                                    SchedulerP & sched);

    // Used to get the progress ratio of an AMR regridding subcycle.
    virtual double getSubCycleProgress(DataWarehouse* fineNewDW);


    // Recompute a time step if current time advance is not
    // converging.  The returned time is the new delta T.
    virtual void   recomputeDelT();
    virtual double recomputeDelT( const double delT );

    virtual bool restartableTimeSteps() { return false; }

    // Updates the time step and the delta T.
    virtual void prepareForNextTimeStep();

    // Asks the application if it needs to be recompiled.
    virtual bool needRecompile( const GridP & grid );

    // Labels for access value in the data warehouse.
    virtual const VarLabel* getTimeStepLabel() const { return m_timeStepLabel; }
    virtual const VarLabel* getSimTimeLabel() const { return m_simulationTimeLabel; }
    virtual const VarLabel* getDelTLabel() const { return m_delTLabel; }

    //////////
    virtual void setAMR(bool val) { m_AMR = val; }
    virtual bool isAMR() const { return m_AMR; }
  
    virtual void setLockstepAMR(bool val) { m_lockstepAMR = val; }
    virtual bool isLockstepAMR() const { return m_lockstepAMR; }

    virtual void setDynamicRegridding(bool val) {m_dynamicRegridding = val; }
    virtual bool isDynamicRegridding() const { return m_dynamicRegridding; }
  
    // Boolean for vars chanegd by the in-situ.
    virtual void haveModifiedVars( bool val ) { m_haveModifiedVars = val; }
    virtual bool haveModifiedVars() const { return m_haveModifiedVars; }
     
    // For restarting.
    virtual bool isRestartTimeStep() const { return m_isRestartTimestep; } 
    virtual void setRestartTimeStep( bool val ){ m_isRestartTimestep = val; }

    // For regridding.
    virtual bool isRegridTimeStep() const { return m_isRegridTimeStep; }
    virtual void setRegridTimeStep( bool val ) {
      m_isRegridTimeStep = val;
      
      if( m_isRegridTimeStep )
	m_lastRegridTimestep = m_timeStep;
    }
    virtual int  getLastRegridTimeStep() { return m_lastRegridTimestep; }

    // Some applications can adjust the output interval.
    virtual void adjustOutputInterval(bool val) { m_adjustOutputInterval = val; }
    virtual bool adjustOutputInterval() const { return m_adjustOutputInterval; }
     
    // Some applications can adjust the checkpoint interval.
    virtual void adjustCheckpointInterval(bool val) { m_adjustCheckpointInterval = val; }
    virtual bool adjustCheckpointInterval() const { return m_adjustCheckpointInterval; }

    // Some applications can end the simulation early.
    virtual void mayEndSimulation(bool val) { m_mayEndSimulation = val; }
    virtual bool mayEndSimulation() const { return m_mayEndSimulation; }

    // Access methods for member classes.
    virtual SimulationTime * getSimulationTime() const { return m_simulationTime; }
    virtual SimulationStateP getSimulationStateP() const { return m_sharedState; }
  
  private:
    // The classes are private because only the top level application
    // should be changing them. This only really matter when there are
    // application built upon multiple application. The children
    // applications will not have valid values. They should ALWAYS get
    // the values via the data warehouse.
    
    //////////
    virtual   void setDelT( double delT ) { m_delT = delT; }
    virtual double getDelT() const { return m_delT; }
    virtual   void setDelTForAllLevels( SchedulerP& scheduler,
                                        const GridP & grid,
                                        const int totalFine );

    virtual   void setNextDelT( double delT );
    virtual double getNextDelT() const { return m_nextDelT; }
    virtual   void validateNextDelT( DataWarehouse  * new_dw );

    //////////
    virtual   void setSimTime( double simTime );
    virtual double getSimTime() const { return m_simTime; };

    virtual   void setSimTimeStart( double simTime )
    {
      m_simTimeStart = simTime;
      setSimTime(simTime);
    }
    
    virtual double getSimTimeStart() const { return m_simTimeStart; }
    
    // Returns the integer time step index of the simulation.  All
    // simulations start with a time step number of 0.  This value is
    // incremented by one before a time step is processed.  The 'set'
    // function should only be called by the SimulationController at the
    // beginning of a simulation.  The 'increment' function is called by
    // the SimulationController at the beginning of each time step.
    virtual void setTimeStep( int timeStep );
    virtual void incrementTimeStep();
    virtual int  getTimeStep() const { return m_timeStep; }

    virtual bool isLastTimeStep( double walltime ) const;
    virtual bool maybeLastTimeStep( double walltime ) const;

    virtual ReductionInfoMapper< ApplicationStatsEnum,
                                 double > & getApplicationStats()
    { return m_application_stats; };

    virtual void resetApplicationStats( double val )
    { m_application_stats.reset( val ); };
      
    virtual void reduceApplicationStats( bool allReduce,
                                         const ProcessorGroup* myWorld )
    { m_application_stats.reduce( allReduce, myWorld ); };      
    
  protected:
    Scheduler       * m_scheduler    {nullptr};
    LoadBalancer    * m_loadBalancer {nullptr};
    SolverInterface * m_solver       {nullptr};
    Regridder       * m_regridder    {nullptr};
    Output          * m_output       {nullptr};

    bool m_recompile {false};
    
  private:
    bool m_AMR {false};
    bool m_lockstepAMR {false};

    bool m_dynamicRegridding {false};
  
    bool m_isRestartTimestep {false};
    
    bool m_isRegridTimeStep {false};
    int  m_lastRegridTimestep { 0 }; // While it may not have been a "re"-grid, the original grid is created on TS 0.

    bool m_haveModifiedVars {false};

    bool m_adjustCheckpointInterval {false};
    bool m_adjustOutputInterval {false};

    bool m_mayEndSimulation {false};
  
    const VarLabel* m_timeStepLabel;
    const VarLabel* m_simulationTimeLabel;
    const VarLabel* m_delTLabel;
  
    const VarLabel* m_outputIntervalLabel;
    const VarLabel* m_outputTimeStepIntervalLabel;
    const VarLabel* m_checkpointIntervalLabel;
    const VarLabel* m_checkpointTimeStepIntervalLabel;

    const VarLabel* m_endSimulationLabel;

    SimulationTime* m_simulationTime {nullptr};
  
    double m_delT{0.0};
    double m_nextDelT{0.0};

    double m_simTime{0.0};             // current sim time
    double m_simTimeStart{0.0};        // starting sim time

    // The time step that the simulation is at.
    int    m_timeStep{0};

    bool   m_endSimulation{false};

  protected:    
    SimulationStateP m_sharedState{nullptr};

    ReductionInfoMapper< ApplicationStatsEnum,
                         double > m_application_stats;    
    
  private:
    ApplicationCommon(const ApplicationCommon&);
    ApplicationCommon& operator=(const ApplicationCommon&);

#ifdef HAVE_VISIT
  public:
    // Reduction analysis variables for on the fly analysis
    virtual std::vector< analysisVar > & getAnalysisVars() { return m_analysisVars; }
  
    // Interactive variables from the UPS problem spec.
    virtual std::vector< interactiveVar > & getUPSVars() { return m_UPSVars; }
  
    // Interactive state variables from the application.
    virtual std::vector< interactiveVar > & getStateVars() { return m_stateVars; }
  
    virtual void setVisIt( unsigned int val ) { m_doVisIt = val; }
    virtual unsigned int  getVisIt() { return m_doVisIt; }

  protected:
    // Reduction analysis variables for on the fly analysis
    std::vector< analysisVar > m_analysisVars;
  
    // Interactive variables from the UPS problem spec.
    std::vector< interactiveVar > m_UPSVars;

    // Interactive state variables from the application.
    std::vector< interactiveVar > m_stateVars;

    unsigned int m_doVisIt{false};
#endif
  };

} // End namespace Uintah
   
#endif
