
#ifndef Packages_Uintah_CCA_Components_Examples_TestModel_h
#define Packages_Uintah_CCA_Components_Examples_TestModel_h

#include <Packages/Uintah/CCA/Ports/ModelInterface.h>
#include <Packages/Uintah/Core/Grid/Variables/ComputeSet.h>

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
    double rate;
    bool d_is_mpm_matl;  // Is matl 0 a mpm_matl?
  };
}

#endif
