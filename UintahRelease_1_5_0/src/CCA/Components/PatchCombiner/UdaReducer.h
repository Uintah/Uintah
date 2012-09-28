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


#ifndef UINTAH_HOMEBREW_Component_UdaReducer_H
#define UINTAH_HOMEBREW_Component_UdaReducer_H

#include <Core/Parallel/UintahParallelComponent.h>
#include <CCA/Ports/SimulationInterface.h>
#include <Core/DataArchive/DataArchive.h>
#include <Core/Grid/Grid.h>
#include <Core/Grid/Variables/ComputeSet.h>
#include <Core/Grid/Variables/VarLabel.h>
#include <Core/Grid/SimulationStateP.h>


#include <vector>

namespace Uintah {
  class LoadBalancer;

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

   class UdaReducer : public SimulationInterface, public UintahParallelComponent {
   public:
     UdaReducer(const ProcessorGroup* myworld, string udaDir);
     
     virtual ~UdaReducer();
     
     //////////
     // Insert Documentation Here:
     virtual void problemSetup(const ProblemSpecP& params, 
                               const ProblemSpecP& restart_prob_spec, 
                               GridP& grid, SimulationStateP& state);
     
     //////////
     // Insert Documentation Here:
     virtual void scheduleInitialize(const LevelP& level,
				     SchedulerP&);
      //////////
     // Insert Documentation Here:
     virtual void restartInitialize() {}
     
     //////////
     // Insert Documentation Here:
     virtual void scheduleComputeStableTimestep(const LevelP&,
						SchedulerP&) {}
     
     //////////
     // Insert Documentation Here:
     virtual void scheduleTimeAdvance( const LevelP& level, SchedulerP&);
     

     virtual bool needRecompile(double time, double dt,
                                const GridP& grid);
     double getMaxTime();

     GridP getGrid();
   private:
     UdaReducer(const UdaReducer&);
     UdaReducer& operator=(const UdaReducer&);

     void initialize(const ProcessorGroup*,
			 const PatchSubset* patches,
			 const MaterialSubset* matls,
			 DataWarehouse* /*old_dw*/,
			 DataWarehouse* new_dw);
       
     void readAndSetDelT(const ProcessorGroup*,
			 const PatchSubset* patches,
			 const MaterialSubset* matls,
			 DataWarehouse* /*old_dw*/,
			 DataWarehouse* new_dw);

     void readAndSetVars(const ProcessorGroup*,
			 const PatchSubset* patches,
			 const MaterialSubset* matls,
			 DataWarehouse* /*old_dw*/,
			 DataWarehouse* new_dw);

     string udaDir_;
     DataArchive* dataArchive_;
     std::vector<int> timesteps_;
     std::vector<double> times_;
     int timeIndex_;
     std::vector<int> numMaterials_;
     GridP oldGrid_;
     bool gridChanged;
     LoadBalancer* lb;
     VarLabel* delt_label;
     std::vector<VarLabel*> labels_;
     SimulationStateP d_sharedState;
   };
} // End namespace Uintah
   


#endif
