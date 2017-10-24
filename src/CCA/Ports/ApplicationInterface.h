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

#include <CCA/Ports/SchedulerP.h>

#include <Core/Parallel/UintahParallelPort.h>
#include <Core/Grid/Variables/ComputeSet.h>
#include <Core/Grid/GridP.h>
#include <Core/Grid/LevelP.h>
#include <Core/Grid/SimulationStateP.h>
#include <Core/Grid/SimulationTime.h>
#include <Core/Grid/Variables/VarLabel.h>
#include <Core/Grid/Variables/VarTypes.h>
#include <Core/OS/Dir.h>
#include <Core/ProblemSpec/ProblemSpecP.h>
#include <Core/Util/Handle.h>
#include <Core/Util/DebugStream.h>
#include <Core/Util/DOUT.hpp>

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

  class DataWarehouse;
  class SimulationTime;

  class ApplicationInterface : public UintahParallelPort {
  public:
    ApplicationInterface();
     virtual ~ApplicationInterface();
      
     //////////
     // Insert Documentation Here:
     virtual void problemSetup( const ProblemSpecP &prob_spec ) = 0;
  
     virtual void problemSetup( const ProblemSpecP     & params,
                                const ProblemSpecP     & restart_prob_spec,
                                      GridP            & grid,
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
                                                   SchedulerP & sched ) = 0;

     //////////
     // restartInitialize() is called once and only once if and when a simulation is restarted.
     // This allows the simulation component to handle initializations that are necessary when
     // a simulation is restarted.
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
     virtual void setAMR(bool val) = 0;
     virtual bool isAMR() const = 0;
  
     virtual void setLockstepAMR(bool val) = 0;
     virtual bool isLockstepAMR() const = 0;

     virtual void setDynamicRegridding(bool val) = 0;
     virtual bool isDynamicRegridding() const = 0;
  
     virtual void haveModifiedVars( bool val ) = 0;
     virtual bool haveModifiedVars() const = 0;
     
     virtual void setSimulationTime( SimulationTime * st ) = 0;
     virtual SimulationTime * getSimulationTime() const = 0;

     //////////
     virtual const VarLabel* getOutputIntervalLabel() const = 0;
     virtual const VarLabel* getOutputTimestepIntervalLabel() const = 0;
     
     virtual const VarLabel* getCheckpointIntervalLabel() const = 0;
     virtual const VarLabel* getCheckpointTimestepIntervalLabel() const = 0;
     
     virtual void updateOutputInterval(bool ans) = 0;
     virtual bool updateOutputInterval() const = 0;
     
     virtual void updateCheckpointInterval(bool ans) = 0;
     virtual bool updateCheckpointInterval() const = 0;

     //////////
     // ask the component which primary task graph it wishes to
     // execute this timestep, this will be an index into the
     // scheduler's vector of task-graphs.
     virtual int computeTaskGraphIndex()
     { return 0; }

     virtual void scheduleSwitchTest(const LevelP& /*level*/,
				     SchedulerP& /*sched*/)
     {};
 
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
