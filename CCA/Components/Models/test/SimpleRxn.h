
#ifndef Packages_Uintah_CCA_Components_Examples_SimpleRxn_h
#define Packages_Uintah_CCA_Components_Examples_SimpleRxn_h

#include <Packages/Uintah/CCA/Ports/ModelInterface.h>
#include <Packages/Uintah/Core/Grid/ComputeSet.h>

namespace Uintah {

/**************************************

CLASS
   SimpleRxn
   
   SimpleRxn simulation

GENERAL INFORMATION

   SimpleRxn.h

   Steven G. Parker
   Department of Computer Science
   University of Utah

   Center for the Simulation of Accidental Fires and Explosions (C-SAFE)
  
   Copyright (C) 2000 SCI Group

KEYWORDS
   SimpleRxn

DESCRIPTION
   Long description...
  
WARNING
  
****************************************/

  class SimpleRxn : public ModelInterface {
  public:
    SimpleRxn(const ProcessorGroup* myworld, ProblemSpecP& params);
    virtual ~SimpleRxn();

    //////////
    // Insert Documentation Here:
    virtual void problemSetup(GridP& grid, SimulationStateP& sharedState,
			      ModelSetup& setup);
      
    //////////
    // Insert Documentation Here:
    virtual void scheduleInitialize(const LevelP& level,
				    SchedulerP&);
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

  private:
    void massExchange(const ProcessorGroup*, const PatchSubset* patches,
		      const MaterialSubset* matls, DataWarehouse*, 
		      DataWarehouse* new_dw, const ModelInfo*);

    SimpleRxn(const SimpleRxn&);
    SimpleRxn& operator=(const SimpleRxn&);

    ProblemSpecP params;
    const Material* matl;

    MaterialSet* mymatls;
    double rate;
    double d_cv_0;  //specific heat

    VarLabel* massFraction;
  };
}

#endif
