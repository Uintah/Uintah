
#ifndef Packages_Uintah_CCA_Components_Examples_Simple_Burn_h
#define Packages_Uintah_CCA_Components_Examples_Simple_Burn_h

#include <Packages/Uintah/CCA/Ports/ModelInterface.h>
#include <Packages/Uintah/Core/Grid/Variables/ComputeSet.h>

namespace Uintah {
  class ICELabel;
  class MPMLabel;
  class MPMICELabel;
/**************************************

CLASS
   Simple_Burn
  

GENERAL INFORMATION

   Simple_Burn.h

   Steven G. Parker
   Department of Computer Science
   University of Utah

   Center for the Simulation of Accidental Fires and Explosions (C-SAFE)
  
   Copyright (C) 2000 SCI Group

KEYWORDS
   Simple_Burn

DESCRIPTION
   Long description...
  
WARNING
  
****************************************/

  class Simple_Burn : public ModelInterface {
  public:
    Simple_Burn(const ProcessorGroup* myworld, ProblemSpecP& params);
    virtual ~Simple_Burn();

    //////////
    // Insert Documentation Here:
    virtual void problemSetup(GridP& grid, SimulationStateP& sharedState,
			      ModelSetup* setup);
      
    virtual void activateModel(GridP& grid, SimulationStateP& sharedState,
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

   virtual void scheduleCheckNeedAddMaterial(SchedulerP&,
                                             const LevelP& level,
                                             const ModelInfo*);
                                             
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

    void checkNeedAddMaterial(const ProcessorGroup*, 
                              const PatchSubset* patches,
                              const MaterialSubset* matls,
                              DataWarehouse*,
                              DataWarehouse* new_dw,
                              const ModelInfo*);

    Simple_Burn(const Simple_Burn&);
    Simple_Burn& operator=(const Simple_Burn&);

    const VarLabel* onSurfaceLabel;   // diagnostic labels
    const VarLabel* surfaceTempLabel;

    ProblemSpecP params;
    const Material* matl0;
    const Material* matl1;
    SimulationStateP d_sharedState;   
    
    MPMICELabel* MIlb;
    ICELabel* Ilb;
    MPMLabel* Mlb;
    MaterialSet* mymatls;
    
    bool d_active;
    double d_thresholdTemp;
    double d_thresholdPress;
    double d_Enthalpy;
    double d_BurnCoeff;
    double d_refPress;
          
    bool d_is_mpm_matl;  // Is matl 0 a mpm_matl?
    double d_cv_0;      //specific heat
    #define d_SMALL_NUM 1e-100
    #define d_TINY_RHO 1e-12
  };
}

#endif
