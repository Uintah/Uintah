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

#ifndef UINTAH_HOMEBREW_APPLICATIONCOMMON_H
#define UINTAH_HOMEBREW_APPLICATIONCOMMON_H

#include <Core/Parallel/UintahParallelComponent.h>
#include <CCA/Ports/ApplicationInterface.h>
#include <CCA/Ports/SchedulerP.h>
#include <CCA/Ports/SolverInterface.h>

#include <Core/Parallel/UintahParallelPort.h>
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

class ModelMaker;
class Regridder;
class Output;

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

class DataWarehouse;
  
  class ApplicationCommon : public UintahParallelComponent,
                            public ApplicationInterface {

    friend class SimulationController;
    friend class AMRSimulationController;
    friend class Switcher;

public:
  ApplicationCommon(const ProcessorGroup* myworld,
                    const SimulationStateP sharedState);

  virtual ~ApplicationCommon();
  
  //////////
  // Insert Documentation Here:
  virtual void getComponents();
  virtual void setComponents( const ApplicationCommon *parent );
  virtual void releaseComponents();

  virtual void problemSetup( const ProblemSpecP &prob_spec );
  
  virtual void problemSetup( const ProblemSpecP & prob_spec,
                             const ProblemSpecP & restart_prob_spec,
                                          GridP & grid ) = 0;
  
  virtual void preGridProblemSetup( const ProblemSpecP & params, 
                                    GridP              & grid ) {};
  
  virtual void outputProblemSpec( ProblemSpecP & prob_spec ) {};
  
  //////////
  // Insert Documentation Here:
  virtual void scheduleInitialize( const LevelP & level,
                                   SchedulerP & scheduler ) = 0;
  
  // on a restart schedule an initialization task
  virtual void scheduleRestartInitialize( const LevelP & level,
                                          SchedulerP & scheduler )  = 0;
  
  //////////
  // restartInitialize() is called once and only once if and when a
  // simulation is restarted.  This allows the simulation component
  // to handle initializations that are necessary when a simulation
  // is restarted.
     // 
  virtual void restartInitialize() {}

  virtual void switchInitialize( const LevelP & level, SchedulerP & scheduler ) {}
  
  //////////
  // Insert Documentation Here:
  virtual void scheduleComputeStableTimeStep( const LevelP & level,
                                              SchedulerP & scheduler ) = 0;
  
  //////////
  // Insert Documentation Here:
  virtual void scheduleTimeAdvance(const LevelP& level, SchedulerP& scheduler);
  
  //////////
  // Insert Documentation Here:
  virtual void scheduleReduceSystemVars(const GridP& grid,
                                        const PatchSet* perProcPatchSet,
                                        SchedulerP& scheduler);

  virtual void reduceSystemVars( const ProcessorGroup *,
                                 const PatchSubset    * patches,
                                 const MaterialSubset * /*matls*/,
                                       DataWarehouse  * /*old_dw*/,
                                       DataWarehouse  * new_dw );
  
  //////////
  // Insert Documentation Here:
  virtual void scheduleInitializeSystemVars(const GridP& grid,
                                            const PatchSet* perProcPatchSet,
                                            SchedulerP& scheduler);

  virtual void initializeSystemVars( const ProcessorGroup *,
                                     const PatchSubset    * patches,
                                     const MaterialSubset * /*matls*/,
                                           DataWarehouse  * /*old_dw*/,
                                           DataWarehouse  * new_dw );
      
  //////////
  // Insert Documentation Here:
  virtual void scheduleUpdateSystemVars(const GridP& grid,
                                        const PatchSet* perProcPatchSet,
                                        SchedulerP& scheduler);

  virtual void updateSystemVars( const ProcessorGroup *,
                                 const PatchSubset    * patches,
                                 const MaterialSubset * /*matls*/,
                                       DataWarehouse  * /*old_dw*/,
                                       DataWarehouse  * new_dw );
      
  // this is for wrapping up a time step when it can't be done in
  // scheduleTimeAdvance.
  virtual void scheduleFinalizeTimeStep(const LevelP& level, SchedulerP&) {}
  virtual void scheduleAnalysis(const LevelP& level, SchedulerP&) {}
     
  virtual void scheduleRefine( const PatchSet* patches, SchedulerP& scheduler );
  virtual void scheduleRefineInterface( const LevelP     & fineLevel, 
                                        SchedulerP & scheduler,
                                        bool         needCoarseOld,
                                        bool         needCoarseNew );
  virtual void scheduleCoarsen( const LevelP     & coarseLevel, 
                                SchedulerP & scheduler );

  /// Schedule to mark flags for AMR regridding
  virtual void scheduleErrorEstimate(const LevelP& coarseLevel,
                                     SchedulerP& sched);
  
  /// Schedule to mark initial flags for AMR regridding
  virtual void scheduleInitialErrorEstimate(const LevelP& coarseLevel,
                                            SchedulerP& sched);
  
  // Redo a time step if current time advance is not converging.
  // Returned time is the new dt to use.
  virtual void   recomputeTimeStep();
  virtual double recomputeTimeStep(double delT);
  virtual bool restartableTimeSteps();
  
  // use this to get the progress ratio of an AMR subcycle
  double getSubCycleProgress(DataWarehouse* fineNewDW);
  
  virtual void prepareForNextTimeStep( const GridP & grid );

  //////////
  // ask the application if it needs to be recompiled
  virtual bool needRecompile( double /*time*/,
                              double /*dt*/,
                              const GridP& /*grid*/)
  { return false; }
  
  virtual const VarLabel* getTimeStepLabel() const { return m_timeStepLabel; }
  virtual const VarLabel* getSimTimeLabel() const { return m_simulationTimeLabel; }
  virtual const VarLabel* getDelTLabel() const { return m_delTLabel; }

  //////////
  virtual void setModelMaker(bool val) { m_needModelMaker = val; }
  virtual bool needModelMaker() const { return m_needModelMaker; }
    
  virtual void setAMR(bool val) { m_AMR = val; }
  virtual bool isAMR() const { return m_AMR; }
  
  virtual void setLockstepAMR(bool val) { m_lockstepAMR = val; }
  virtual bool isLockstepAMR() const { return m_lockstepAMR; }

  virtual void setDynamicRegridding(bool val) {m_dynamicRegridding = val; }
  virtual bool isDynamicRegridding() const { return m_dynamicRegridding; }
  
  virtual void haveModifiedVars( bool val ) { m_haveModifiedVars = val; }
  virtual bool haveModifiedVars() const { return m_haveModifiedVars; }

  //////////
  virtual SimulationTime * getSimulationTime() const { return m_simulationTime; }
  virtual SimulationStateP getSimulationStateP() const { return m_sharedState; };
        
  virtual bool isRegridTimeStep() const { return m_isRegridTimeStep; }
  virtual void setRegridTimeStep(bool ans) { m_isRegridTimeStep = ans; }

  //////////
  virtual void adjustOutputInterval(bool ans) { m_adjustOutputInterval = ans; }
  virtual bool adjustOutputInterval() const { return m_adjustOutputInterval; }
  
  //////////
  virtual void adjustCheckpointInterval(bool ans) { m_adjustCheckpointInterval = ans; }
  virtual bool adjustCheckpointInterval() const { return m_adjustCheckpointInterval; }
    
  //////////
  virtual void mayEndSimulation(bool ans) { m_mayEndSimulation = ans; }
  virtual bool mayEndSimulation() const { return m_mayEndSimulation; }

  //////////
  // ask the application which primary task graph it wishes to
  // execute this time step, this will be an index into the
  // scheduler's vector of task-graphs.
  virtual int computeTaskGraphIndex() { return 0; }


  virtual void scheduleSwitchTest(const LevelP& /*level*/,
                                  SchedulerP& /*sched*/) {};
private:
    // The classes are private because only the top level application
    // should be changing them. This only really matter when there are
    // application built upon multiple application. The children
    // applications will not have valid values. They should ALWAYS get
    // the values via the data warehouse.
    
  //////////
  virtual   void setDelT( double val );
  virtual double getDelT() const { return m_delT; }
  virtual   void setDelTForAllLevels( SchedulerP& scheduler,
                                      const GridP & grid,
                                      const int totalFine );

  virtual   void setNextDelT( double val );
  virtual double getNextDelT() const { return m_nextDelT; }
  virtual   void validateNextDelT( DataWarehouse  * new_dw );

  //////////
  virtual   void setSimTime( double val );
  virtual double getSimTime() const { return m_simTime; };

  virtual   void setSimTimeStart( double val )
  {
    m_simTimeStart = val;
    setSimTime(val);
  }
    
  virtual double getSimTimeStart() const { return m_simTimeStart; }
    
  // Returns the integer time step index of the simulation.  All
  // simulations start with a time step number of 0.  This value is
  // incremented by one before a time step is processed.  The 'set'
  // function should only be called by the SimulationController at the
  // beginning of a simulation.  The 'increment' function is called by
  // the SimulationController at the beginning of each time step.
  virtual void setTimeStep( int timeStep );
  virtual void incrementTimeStep( const GridP & grid );
  virtual int  getTimeStep() const { return m_timeStep; }

  virtual bool isLastTimeStep( double walltime ) const;
  virtual bool maybeLastTimeStep( double walltime ) const;

protected:
  Scheduler*       m_scheduler{nullptr};
  ModelMaker*      m_modelMaker{nullptr};
  SolverInterface* m_solver{nullptr};
  Regridder*       m_regridder{nullptr};
  Output*          m_output{nullptr};

private:
  bool m_needModelMaker {false};
  bool m_AMR {false};
  bool m_lockstepAMR {false};

  bool m_dynamicRegridding {false};
  
  bool m_isRegridTimeStep {false};

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
  
  // Debug streams that can be turned on or off.
  virtual std::vector< DebugStream * > & getDebugStreams() { return m_debugStreams; }
  virtual std::vector< Dout * > & getDouts() { return m_douts; }
  
  virtual void setVisIt( unsigned int val ) { m_doVisIt = val; }
  virtual unsigned int  getVisIt() { return m_doVisIt; }

protected:
  // Reduction analysis variables for on the fly analysis
  std::vector< analysisVar > m_analysisVars;
  
  // Interactive variables from the UPS problem spec.
  std::vector< interactiveVar > m_UPSVars;

  // Interactive state variables from the application.
  std::vector< interactiveVar > m_stateVars;

  // Debug streams that can be turned on or off.
  std::vector< DebugStream * > m_debugStreams;
  std::vector< Dout * > m_douts;

  unsigned int m_doVisIt{false};
#endif
};

} // End namespace Uintah
   
#endif
