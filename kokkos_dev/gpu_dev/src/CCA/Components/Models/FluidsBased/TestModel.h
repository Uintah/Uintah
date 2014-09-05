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



#ifndef Packages_Uintah_CCA_Components_Examples_TestModel_h
#define Packages_Uintah_CCA_Components_Examples_TestModel_h

#include <CCA/Ports/ModelInterface.h>
#include <Core/Grid/Variables/ComputeSet.h>

namespace Uintah {
  class MPMICELabel;

/**************************************

CLASS
   TestModel
   
   TestModel simulation

GENERAL INFORMATION

   TestModel.h

   Steven G. Parker
   Department of Computer Science
   University of Utah

   Center for the Simulation of Accidental Fires and Explosions (C-SAFE)
  
   Copyright (C) 2000 SCI Group

KEYWORDS
   TestModel

DESCRIPTION
   Long description...
  
WARNING
  
****************************************/

  class TestModel : public ModelInterface {
  public:
    TestModel(const ProcessorGroup* myworld, ProblemSpecP& params);
    virtual ~TestModel();

    virtual void outputProblemSpec(ProblemSpecP& ps);

    virtual void problemSetup(GridP& grid, SimulationStateP& sharedState,
                              ModelSetup* setup);
      
    virtual void scheduleInitialize(SchedulerP&,
                                        const LevelP& level,
                                        const ModelInfo*);

    virtual void restartInitialize() {}
      
    virtual void scheduleComputeStableTimestep(SchedulerP&,
                                                   const LevelP& level,
                                                   const ModelInfo*);
      
    virtual void scheduleComputeModelSources(SchedulerP&,
                                                const LevelP& level,
                                                const ModelInfo*);
                                             
    virtual void scheduleModifyThermoTransportProperties(SchedulerP&,
                                                         const LevelP&,
                                                         const MaterialSet*);
                                               
    virtual void computeSpecificHeat(CCVariable<double>&,
                                    const Patch*,
                                    DataWarehouse*,
                                    const int);
                                    
   virtual void scheduleErrorEstimate(const LevelP& coarseLevel,
                                      SchedulerP& sched);
                                      
   virtual void scheduleTestConservation(SchedulerP&,
                                         const PatchSet* patches,
                                         const ModelInfo* mi);

  private:    
    void computeModelSources(const ProcessorGroup*, 
                             const PatchSubset* patches,
                               const MaterialSubset* matls, 
                             DataWarehouse*, 
                               DataWarehouse* new_dw, 
                             const ModelInfo*);

    TestModel(const TestModel&);
    TestModel& operator=(const TestModel&);

    ProblemSpecP params;
    const Material* matl0;
    const Material* matl1;
    MPMICELabel* MIlb;
    MaterialSet* mymatls;
    Material* d_matl;
    double d_rate;
    double d_startTime;   // time to start converting m0->m1
    bool d_is_mpm_matl;  // Is matl 0 a mpm_matl?
    
    const VarLabel* totalMassXLabel;
    const VarLabel* totalIntEngXLabel;
    SimulationStateP d_sharedState;
  };
}

#endif
