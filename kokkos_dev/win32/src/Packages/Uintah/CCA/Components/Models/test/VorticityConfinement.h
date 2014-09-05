
#ifndef Packages_Uintah_CCA_Components_Examples_VorticityConfinement_h
#define Packages_Uintah_CCA_Components_Examples_VorticityConfinement_h
#include <Packages/Uintah/CCA/Ports/ModelInterface.h>

#include <Packages/Uintah/Core/Grid/Variables/ComputeSet.h>
#include <Packages/Uintah/Core/Grid/Variables/CCVariable.h>
#include <Packages/Uintah/Core/Grid/Variables/VarTypes.h>
#include <vector>

namespace Uintah {
  class ICELabel;

/**************************************

CLASS
   VorticityConfinement
   
   VorticityConfinement simulation

GENERAL INFORMATION

   VorticityConfinement.h

   Todd Harman
   Department of Computer Science
   University of Utah

   Center for the Simulation of Accidental Fires and Explosions (C-SAFE)
  
   Copyright (C) 2000 SCI Group

KEYWORDS
   VorticityConfinement

DESCRIPTION
   Long description...
  
WARNING
  
****************************************/

  class GeometryPiece;
  class VorticityConfinement :public ModelInterface {
  public:
    VorticityConfinement(const ProcessorGroup* myworld, ProblemSpecP& params);
    virtual ~VorticityConfinement();
    
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
    ICELabel* lb;
                                                
    void computeModelSources(const ProcessorGroup*, 
                             const PatchSubset* patches,
                             const MaterialSubset*,
                             DataWarehouse* old_dw,
                             DataWarehouse* new_dw,
                             const ModelInfo* mi);

    VorticityConfinement(const VorticityConfinement&);
    VorticityConfinement& operator=(const VorticityConfinement&);

    ProblemSpecP params;
    SimulationStateP sharedState;
    const Material* d_matl;
    MaterialSet* d_matl_set;
    double scale;
  };
}

#endif
