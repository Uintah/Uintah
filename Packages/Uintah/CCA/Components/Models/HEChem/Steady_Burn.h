
#ifndef Packages_Uintah_CCA_Components_Examples_Steady_Burn_h
#define Packages_Uintah_CCA_Components_Examples_Steady_Burn_h

#include <Packages/Uintah/CCA/Ports/ModelInterface.h>
#include <Packages/Uintah/Core/Grid/ComputeSet.h>

#include <Packages/Uintah/Core/Grid/NCVariable.h>

namespace Uintah {
  class ICELabel;
  class MPMLabel;
  class MPMICELabel;
/**************************************

CLASS
   Steady_Burn
  

GENERAL INFORMATION

   Steady_Burn.h

   Steven G. Parker
   Department of Computer Science
   University of Utah

   Center for the Simulation of Accidental Fires and Explosions (C-SAFE)
  
   Copyright (C) 2000 SCI Group

KEYWORDS
   Steady_Burn

DESCRIPTION
   This is one of the general class of reaction models that converts solid 
   material into gas (combustion of high energy material).  The fomalism is
   based on the Ward, Son, Brewster (WSB) algebraic model of HE combustion.
   At each time step, the burn rate (mass transfer rate) of the solid 
   material is computed by calculating the surface temperature expected for
   a steady burn (based on the bulk solid temperature, gas pressure or 
   density, thermal conductivity and heat capacity of the solid.  The main
   assumption is that the temperature distributions in both the solid and
   gas are at equilibrium at each time step in the simulation.
  
WARNING
  
****************************************/

  class Steady_Burn : public ModelInterface {
  public:
    Steady_Burn(const ProcessorGroup* myworld, ProblemSpecP& params);
    virtual ~Steady_Burn();

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
    virtual void initialize(const ProcessorGroup*,
                            const PatchSubset*,
                            const MaterialSubset*,
                            DataWarehouse*,
                            DataWarehouse*);
    
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


  private:    
    void computeModelSources(const ProcessorGroup*, const PatchSubset*,
			     const MaterialSubset*, DataWarehouse*, 
			     DataWarehouse*, const ModelInfo*);

    double computeSurfaceArea(Vector &rhoGradVector, Vector &dx);


    Vector computeDensityGradientVector(IntVector *nodeIdx, 
                                        constNCVariable<double> &NCsolidMass,
                                        constNCVariable<double> &NC_CCweight,
                                        Vector &dx);

    double computeBurnedMass(double& Ts, double To, double Kc, double Kg, 
			     double Vc, double Vg, double Cp, double surfArea, 
			     double delT, double solidMass);

    double computeWSBSurfaceTemp(double Ts, double To, double Kc, double Kg,
				 double Vc, double Vg, double Cp);

    double computeMassTransferRate(double Ts, double To, double Kc, double Vc, double Cp);

	 
    Steady_Burn(const Steady_Burn&);
    Steady_Burn& operator=(const Steady_Burn&);

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

    double Ac;
    double Ec;
    double Bg;
    double Qc;
    double Qg;
    double MW;
    double ignitionTemp;
  
    #define d_SMALL_NUM 1e-100
    #define d_TINY_RHO 1e-12
  };
}

#endif
