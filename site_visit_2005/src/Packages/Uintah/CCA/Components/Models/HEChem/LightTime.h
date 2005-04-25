
#ifndef Packages_Uintah_CCA_Components_Models_LightTime_h
#define Packages_Uintah_CCA_Components_Models_LightTime_h

#include <Packages/Uintah/CCA/Ports/ModelInterface.h>
#include <Packages/Uintah/Core/Grid/Variables/ComputeSet.h>

namespace Uintah {
  class ICELabel;
/**************************************

CLASS
   LightTime
  

GENERAL INFORMATION

   LightTime.h

   Jim Guilkey
   Department of Mechanical Engineering
   University of Utah

   Center for the Simulation of Accidental Fires and Explosions (C-SAFE)
  
   Copyright (C) 2000 SCI Group

KEYWORDS
   "Program Burn" "Lighting Time Model"

DESCRIPTION

WARNING

****************************************/

  class LightTime : public ModelInterface {
  public:
    LightTime(const ProcessorGroup* myworld, ProblemSpecP& params);
    virtual ~LightTime();

    virtual void problemSetup(GridP& grid, SimulationStateP& sharedState,
			      ModelSetup* setup);
      
    virtual void scheduleInitialize(SchedulerP&,
				    const LevelP& level,
				    const ModelInfo*);

    void initialize(const ProcessorGroup*,
                    const PatchSubset* patches,
                    const MaterialSubset* matls,
                    DataWarehouse*,
                    DataWarehouse*);

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

    LightTime(const LightTime&);
    LightTime& operator=(const LightTime&);

    const VarLabel* reactedFractionLabel;   // diagnostic labels
    const VarLabel* delFLabel;   // diagnostic labels

    ProblemSpecP params;
    const Material* matl0;
    const Material* matl1;
    SimulationStateP d_sharedState;   

    ICELabel* Ilb;
    MaterialSet* mymatls;
    
    double d_P_cj;
    double d_D;
    double d_E0;
    double d_rho0;
    Point  d_start_place;
    Vector d_direction;
    bool   d_react_mixed_cells;

    #define d_SMALL_NUM 1e-100
    #define d_TINY_RHO 1e-12
  };
}

#endif
