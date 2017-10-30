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

#include <CCA/Ports/ApplicationInterface.h>
#include <CCA/Ports/SchedulerP.h>

#include <Core/Parallel/UintahParallelPort.h>
#include <Core/Grid/Variables/ComputeSet.h>
#include <Core/Grid/GridP.h>
#include <Core/Grid/LevelP.h>
#include <Core/Grid/SimulationStateP.h>
#include <Core/Grid/SimulationTime.h>
#include <Core/OS/Dir.h>
#include <Core/Util/Handle.h>
#include <Core/ProblemSpec/ProblemSpecP.h>

#include <sci_defs/visit_defs.h>

namespace Uintah {

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
  
class ApplicationCommon : public ApplicationInterface {

public:
  ApplicationCommon();

  virtual ~ApplicationCommon();
  
  //////////
  // Insert Documentation Here:
  virtual void problemSetup( const ProblemSpecP &prob_spec );
  
  virtual void problemSetup( const ProblemSpecP & prob_spec,
			     const ProblemSpecP & restart_prob_spec,
                                          GridP & grid,
			       SimulationStateP & state ) = 0;
  
  virtual void preGridProblemSetup( const ProblemSpecP     & params, 
				    GridP            & grid,
				    SimulationStateP & state ) {};
  
  virtual void outputProblemSpec( ProblemSpecP & ps ) {};
  
  //////////
  // Insert Documentation Here:
  virtual void scheduleInitialize( const LevelP     & level,
				   SchedulerP & sched ) = 0;
  
  // on a restart schedule an initialization task
  virtual void scheduleRestartInitialize( const LevelP     & level,
					  SchedulerP & sched )  = 0;
  
  //////////
  // restartInitialize() is called once and only once if and when a
  // simulation is restarted.  This allows the simulation component
  // to handle initializations that are necessary when a simulation
  // is restarted.
     // 
  virtual void restartInitialize() {}

  virtual void switchInitialize( const LevelP & level, SchedulerP & sched ) {}
  
  //////////
  // Insert Documentation Here:
  virtual void scheduleComputeStableTimestep( const LevelP     & level,
                                                       SchedulerP & sched ) = 0;
  
  //////////
  // Insert Documentation Here:
  virtual void scheduleTimeAdvance(const LevelP& level, SchedulerP&);
  
  // this is for wrapping up a timestep when it can't be done in
  // scheduleTimeAdvance.
  virtual void scheduleFinalizeTimestep(const LevelP& level, SchedulerP&) {}
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
  
  // Redo a timestep if current time advance is not converging.
  // Returned time is the new dt to use.
  virtual double recomputeTimestep(double delt);
  virtual bool restartableTimesteps();
  
  // use this to get the progress ratio of an AMR subcycle
  double getSubCycleProgress(DataWarehouse* fineNewDW);
  

  //////////
  // ask the component if it needs to be recompiled
  virtual bool needRecompile( double /*time*/,
			      double /*dt*/,
			      const GridP& /*grid*/)
  { return false; }
  
  //////////
  virtual void setAMR(bool val) {m_AMR = val; }
  virtual bool isAMR() const { return m_AMR; }
  
  virtual void setLockstepAMR(bool val) {m_lockstepAMR = val; }
  virtual bool isLockstepAMR() const { return m_lockstepAMR; }

  virtual void setDynamicRegridding(bool val) {m_dynamicRegridding = val; }
  virtual bool isDynamicRegridding() const { return m_dynamicRegridding; }
  
  virtual void haveModifiedVars( bool val ) { m_haveModifiedVars = val; }
  virtual bool haveModifiedVars() const { return m_haveModifiedVars; }

  virtual void setSimulationTime( SimulationTime * st ) { m_simulationTime = st; }
  virtual SimulationTime * getSimulationTime() const { return m_simulationTime; }

  //////////
  virtual const VarLabel* getOutputIntervalLabel() const {
    return m_outputIntervalLabel;
  }
  virtual const VarLabel* getOutputTimestepIntervalLabel() const {
    return m_outputTimestepIntervalLabel;
  }

  virtual const VarLabel* getCheckpointIntervalLabel() const {
    return m_checkpointIntervalLabel;
  }
  virtual const VarLabel* getCheckpointTimestepIntervalLabel() const {
    return m_checkpointTimestepIntervalLabel;
  }

  virtual void updateOutputInterval(bool ans) { m_updateOutputInterval = ans; }
  virtual bool updateOutputInterval() const { return m_updateOutputInterval; }
  
  virtual void updateCheckpointInterval(bool ans) { m_updateCheckpointInterval = ans; }
  virtual bool updateCheckpointInterval() const { return m_updateCheckpointInterval; }

  //////////
  // ask the component which primary task graph it wishes to
  // execute this timestep, this will be an index into the
  // scheduler's vector of task-graphs.
  virtual int computeTaskGraphIndex() { return 0; }


  virtual void scheduleSwitchTest(const LevelP& /*level*/,
				  SchedulerP& /*sched*/) {};
 
protected:
  bool m_AMR {false};
  bool m_lockstepAMR {false};

  bool m_dynamicRegridding {false};
  
  bool m_haveModifiedVars {false};

  bool m_updateCheckpointInterval {false};
  bool m_updateOutputInterval {false};

  const VarLabel* m_outputIntervalLabel;
  const VarLabel* m_outputTimestepIntervalLabel;
  const VarLabel* m_checkpointIntervalLabel;
  const VarLabel* m_checkpointTimestepIntervalLabel;

  SimulationTime* m_simulationTime {nullptr};
  
private:
  ApplicationCommon(const ApplicationCommon&);
  ApplicationCommon& operator=(const ApplicationCommon&);

#ifdef HAVE_VISIT
public:
  // Reduction analysis variables for on the fly analysis
  virtual std::vector< analysisVar > & getAnalysisVars() { return m_analysisVars; }
  
  // Interactive variables from the UPS problem spec.
  virtual std::vector< interactiveVar > & getUPSVars() { return m_UPSVars; }
  
  // Interactive state variables from components.
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

  // Interactive state variables from components.
  std::vector< interactiveVar > m_stateVars;

  // Debug streams that can be turned on or off.
  std::vector< DebugStream * > m_debugStreams;
  std::vector< Dout * > m_douts;

  unsigned int m_doVisIt;
#endif
};

} // End namespace Uintah
   
#endif
