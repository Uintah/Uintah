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


#ifndef Packages_Uintah_CCA_Components_Switching_SteadyState_h
#define Packages_Uintah_CCA_Components_Switching_SteadyState_h

#include <Core/ProblemSpec/ProblemSpecP.h>
#include <CCA/Ports/SwitchingCriteria.h>
#include <Core/Grid/Variables/ComputeSet.h>
#include <Core/Grid/SimulationState.h>


namespace Uintah {

  class ProcessorGroup;
  class DataWarehouse;
  class VarLabel;

  class SteadyState : public SwitchingCriteria
    {
    public:
      // this function has a switch for all known SwitchingCriteria
    
      SteadyState(ProblemSpecP& ps);
      virtual ~SteadyState();
      
      virtual void problemSetup(const ProblemSpecP& ps, 
                                const ProblemSpecP& restart_prob_spec, 
                                SimulationStateP& sharedState);

      virtual void scheduleInitialize(const LevelP& level, SchedulerP& sched);
      virtual void scheduleSwitchTest(const LevelP& level, SchedulerP& sched);
      virtual void scheduleDummy(const LevelP& level, SchedulerP& sched);

      void switchTest(const ProcessorGroup*, const PatchSubset* patches,
                      const MaterialSubset* matls, DataWarehouse*,
                      DataWarehouse*);

      void initialize(const ProcessorGroup*, const PatchSubset* patches,
                      const MaterialSubset* matls, DataWarehouse* old_dw,
                      DataWarehouse* new_dw);

      void dummy(const ProcessorGroup*, const PatchSubset* patches,
                      const MaterialSubset* matls, DataWarehouse* old_dw,
                      DataWarehouse* new_dw);



    private:
      unsigned int d_material;
      unsigned int d_numSteps;
      const VarLabel* heatRate_CCLabel;
      const VarLabel* heatFluxSumLabel;
      const VarLabel* heatFluxSumTimeDerivativeLabel;
      SimulationStateP d_sharedState; 
    };
} // End namespace Uintah

#endif 
