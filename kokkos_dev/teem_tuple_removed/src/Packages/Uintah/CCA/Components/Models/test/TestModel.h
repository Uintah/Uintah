
#ifndef Packages_Uintah_CCA_Components_Examples_TestModel_h
#define Packages_Uintah_CCA_Components_Examples_TestModel_h

#include <Packages/Uintah/CCA/Ports/ModelInterface.h>
#include <Packages/Uintah/Core/Grid/ComputeSet.h>
#include <Packages/Uintah/CCA/Components/MPMICE/MPMICELabel.h>
namespace Uintah {

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

    //////////
    // Insert Documentation Here:
    virtual void problemSetup(GridP& grid, SimulationStateP& sharedState,
			      ModelSetup* setup);
      
    //////////
    // Insert Documentation Here:
    virtual void scheduleInitialize(SchedulerP&,
				    const LevelP& level,
				    const ModelInfo*);

    //////////
    // Insert Documentation Here:
    virtual void restartInitialize() {}
      
    //////////
    // Insert Documentation Here:
    virtual void scheduleComputeStableTimestep(SchedulerP&,
					       const LevelP& level,
					       const ModelInfo*);
      
    //////////
    // Insert Documentation Here:
    virtual void scheduleMassExchange(SchedulerP&,
				      const LevelP& level,
				      const ModelInfo*);
    virtual void scheduleMomentumAndEnergyExchange(SchedulerP&,
						   const LevelP& level,
						   const ModelInfo*);
                                             
    virtual void scheduleModifyThermoTransportProperties(SchedulerP&,
                                               const LevelP&,
                                               const MaterialSet*);
                                               
    virtual void computeSpecificHeat(CCVariable<double>&,
                                 const Patch*,
                                 DataWarehouse*,
                                 const int);

  private:    
    void massExchange(const ProcessorGroup*, const PatchSubset* patches,
		      const MaterialSubset* matls, DataWarehouse*, 
		      DataWarehouse* new_dw, const ModelInfo*);

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
