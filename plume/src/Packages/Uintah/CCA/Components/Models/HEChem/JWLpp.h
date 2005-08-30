
#ifndef Packages_Uintah_CCA_Components_Models_JWLpp_h
#define Packages_Uintah_CCA_Components_Models_JWLpp_h

#include <Packages/Uintah/CCA/Ports/ModelInterface.h>
#include <Packages/Uintah/Core/Grid/Variables/ComputeSet.h>

namespace Uintah {
  class ICELabel;
/**************************************

CLASS
   JWLpp
  

GENERAL INFORMATION

   JWLpp.h

   Jim Guilkey
   Department of Mechanical Engineering
   University of Utah

   Center for the Simulation of Accidental Fires and Explosions (C-SAFE)
  
   Copyright (C) 2000 SCI Group

KEYWORDS
   JWL++ "Reactive Flow Model"

DESCRIPTION
   Model for detonation of HE based on "JWL++:  A Simple Reactive
   Flow Code Package for Detonation", P.Clark Souers, Steve Anderson,
   James Mercer, Estella McGuire and Peter Vitello, Propellants,
   Explosives, Pyrotechnics, 25, 54-58, 2000.
  
WARNING

****************************************/

  class JWLpp : public ModelInterface {
  public:
    JWLpp(const ProcessorGroup* myworld, ProblemSpecP& params);
    virtual ~JWLpp();

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

    JWLpp(const JWLpp&);
    JWLpp& operator=(const JWLpp&);

    const VarLabel* reactedFractionLabel;   // diagnostic labels
    const VarLabel* delFLabel;   // diagnostic labels

    ProblemSpecP params;
    const Material* matl0;
    const Material* matl1;
    SimulationStateP d_sharedState;   

    ICELabel* Ilb;
    MaterialSet* mymatls;
    
    bool d_active;
    double d_G;
    double d_b;
    double d_E0;
    double d_rho0;
    double d_threshold_pressure;

    #define d_SMALL_NUM 1e-100
    #define d_TINY_RHO 1e-12
  };
}

#endif
