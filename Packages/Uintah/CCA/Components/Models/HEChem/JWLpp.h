
#ifndef Packages_Uintah_CCA_Components_Models_JWLpp_h
#define Packages_Uintah_CCA_Components_Models_JWLpp_h

#include <Packages/Uintah/CCA/Ports/ModelInterface.h>
#include <Packages/Uintah/Core/Grid/ComputeSet.h>

#include <Packages/Uintah/CCA/Components/ICE/ICELabel.h>
#include <Packages/Uintah/CCA/Components/MPM/MPMLabel.h>
#include <Packages/Uintah/CCA/Components/MPMICE/MPMICELabel.h>
namespace Uintah {

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
   Ignition and Growth

DESCRIPTION
   Model for detontation of HE based on "Sideways Plate Push Test for
   Detonating Explosives", C.M Tarver, W.C. Tao, Chet. G. Lee, Propellant,
   Explosives, Pyrotechnics, 21, 238-246, 1996.
  
WARNING

****************************************/

  class JWLpp : public ModelInterface {
  public:
    JWLpp(const ProcessorGroup* myworld, ProblemSpecP& params);
    virtual ~JWLpp();

    virtual void problemSetup(GridP& grid, SimulationStateP& sharedState,
			      ModelSetup* setup);
      
    virtual void scheduleInitialize(SchedulerP&,
				    const LevelP& level,
				    const ModelInfo*);

    virtual void restartInitialize() {}
      
    virtual void scheduleComputeStableTimestep(SchedulerP&,
					       const LevelP& level,
					       const ModelInfo*);
      
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
