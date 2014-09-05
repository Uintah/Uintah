
#ifndef Packages_Uintah_CCA_Components_Models_IandG_h
#define Packages_Uintah_CCA_Components_Models_IandG_h

#include <Packages/Uintah/CCA/Ports/ModelInterface.h>
#include <Packages/Uintah/Core/Grid/ComputeSet.h>

namespace Uintah {
  class ICELabel;
/**************************************

CLASS
   IandG
  

GENERAL INFORMATION

   IandG.h

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

  class IandG : public ModelInterface {
  public:
    IandG(const ProcessorGroup* myworld, ProblemSpecP& params);
    virtual ~IandG();

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

    IandG(const IandG&);
    IandG& operator=(const IandG&);

    const VarLabel* reactedFractionLabel;   // diagnostic labels
    const VarLabel* IandGterm1Label;   // diagnostic labels
    const VarLabel* IandGterm2Label;   // diagnostic labels
    const VarLabel* IandGterm3Label;   // diagnostic labels

    ProblemSpecP params;
    const Material* matl0;
    const Material* matl1;
    SimulationStateP d_sharedState;   

    ICELabel* Ilb;
    MaterialSet* mymatls;
    
    double d_I;
    double d_G1;
    double d_G2;
    double d_a;
    double d_b;
    double d_c;
    double d_d;
    double d_e;
    double d_g;
    double d_x;
    double d_y;
    double d_z;
    double d_Figmax;
    double d_FG1max;
    double d_FG2min;
    double d_rho0;
    double d_E0;
    double d_threshold_pressure;

    #define d_SMALL_NUM 1e-100
    #define d_TINY_RHO 1e-12
  };
}

#endif
