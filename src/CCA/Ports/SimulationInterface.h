/*

The MIT License

Copyright (c) 1997-2011 Center for the Simulation of Accidental Fires and 
Explosions (CSAFE), and  Scientific Computing and Imaging Institute (SCI), 
University of Utah.

License for the specific language governing rights and limitations under
Permission is hereby granted, free of charge, to any person obtaining a 
copy of this software and associated documentation files (the "Software"),
to deal in the Software without restriction, including without limitation 
the rights to use, copy, modify, merge, publish, distribute, sublicense, 
and/or sell copies of the Software, and to permit persons to whom the 
Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included 
in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS 
OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, 
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL 
THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER 
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING 
FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER 
DEALINGS IN THE SOFTWARE.

*/


#ifndef UINTAH_HOMEBREW_SimulationInterface_H
#define UINTAH_HOMEBREW_SimulationInterface_H

#include <Core/Parallel/UintahParallelPort.h>
#include <Core/Grid/Variables/ComputeSet.h>
#include <Core/Grid/GridP.h>
#include <Core/Grid/LevelP.h>
#include <Core/Grid/SimulationStateP.h>
#include <Core/Util/Handle.h>
#include <Core/ProblemSpec/ProblemSpecP.h>
#include <CCA/Ports/SchedulerP.h>
#include <Core/OS/Dir.h>

namespace Uintah {

  using SCIRun::Dir;
/**************************************

CLASS
   SimulationInterface
   
   Short description...

GENERAL INFORMATION

   SimulationInterface.h

   Steven G. Parker
   Department of Computer Science
   University of Utah

   Center for the Simulation of Accidental Fires and Explosions (C-SAFE)
  
   Copyright (C) 2000 SCI Group

KEYWORDS
   Simulation_Interface

DESCRIPTION
   Long description...
  
WARNING
  
****************************************/

  class DataWarehouse;
   class SimulationInterface : public UintahParallelPort {
   public:
     SimulationInterface();
     virtual ~SimulationInterface();
      
     //////////
     // Insert Documentation Here:
     virtual void problemSetup(const ProblemSpecP& params, 
                               const ProblemSpecP& restart_prob_spec,
                               GridP& grid, SimulationStateP& state) = 0;

     virtual void outputProblemSpec(ProblemSpecP& ps) {}
     virtual void outputPS(Dir& dir) {}
      
     //////////
     // Insert Documentation Here:
     virtual void scheduleInitialize(const LevelP& level,
				     SchedulerP&) = 0;
     //////////
     virtual void scheduleInitializeAddedMaterial(const LevelP& level,
                                                  SchedulerP&);
     //////////
     // restartInitialize() is called once and only once if and when a simulation is restarted.
     // This allows the simulation component to handle initializations that are necessary when
     // a simulation is restarted.
     // 
     virtual void restartInitialize() {}

     virtual void switchInitialize(const LevelP& level,SchedulerP&) {}
      
     //////////
     // Insert Documentation Here:
     virtual void scheduleComputeStableTimestep(const LevelP& level,
						SchedulerP&) = 0;
      
     //////////
     // Insert Documentation Here:
     virtual void scheduleTimeAdvance(const LevelP& level, SchedulerP&);

     // this is for wrapping up a timestep when it can't be done in scheduleTimeAdvance.
     virtual void scheduleFinalizeTimestep(const LevelP& level, SchedulerP&) {}
     virtual void scheduleRefine(const PatchSet* patches, 
				 SchedulerP& scheduler);
     virtual void scheduleRefineInterface(const LevelP& fineLevel, 
				          SchedulerP& scheduler,
					  bool needCoarseOld, bool needCoarseNew);
     virtual void scheduleCoarsen(const LevelP& coarseLevel, 
				  SchedulerP& scheduler);

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
     virtual bool needRecompile(double /*time*/, double /*dt*/,
				const GridP& /*grid*/) {return false;}

     // direct component to add a new material
     virtual void addMaterial(const ProblemSpecP& params, GridP& grid,
                              SimulationStateP& state);

     virtual void scheduleSwitchTest(const LevelP& /*level*/, SchedulerP& /*sched*/)
       {};
 
   private:
     SimulationInterface(const SimulationInterface&);
     SimulationInterface& operator=(const SimulationInterface&);
   };
} // End namespace Uintah
   


#endif
