
#ifndef Packages_Uintah_CCA_Components_Examples_Burger_h
#define Packages_Uintah_CCA_Components_Examples_Burger_h

#include <Packages/Uintah/Core/Parallel/UintahParallelComponent.h>
#include <Packages/Uintah/CCA/Ports/SimulationInterface.h>
#include <Packages/Uintah/Core/Grid/ComputeSet.h>

namespace Uintah {
  class SimpleMaterial;
  class ExamplesLabel;

/**************************************

CLASS
   Burger
   
   Burger simulation

GENERAL INFORMATION

   Burger.h

   Nathan Lovell and Steven Parker
   Department of Computer Science
   University of Utah

   Center for the Simulation of Accidental Fires and Explosions (C-SAFE)
  
   Copyright (C) 2002 SCI Group

KEYWORDS
   Burger

DESCRIPTION
   2D implementation of Burger's Algorithm using Euler's method.
   Independent study for CS3200
  
WARNING
  
****************************************/


  class Burger : public UintahParallelComponent, public SimulationInterface {
  public:
    Burger(const ProcessorGroup* myworld);
    virtual ~Burger();

    virtual void problemSetup(const ProblemSpecP& params, GridP& grid,
			      SimulationStateP&);
    virtual void scheduleInitialize(const LevelP& level,
				    SchedulerP& sched);
    virtual void scheduleComputeStableTimestep(const LevelP& level,
					       SchedulerP&);
    virtual void scheduleTimeAdvance( const LevelP& level, 
				      SchedulerP&, int step, int nsteps );
  private:
    void initialize(const ProcessorGroup*,
		    const PatchSubset* patches, const MaterialSubset* matls,
		    DataWarehouse* old_dw, DataWarehouse* new_dw);
    void computeStableTimestep(const ProcessorGroup*,
			       const PatchSubset* patches,
			       const MaterialSubset* matls,
			       DataWarehouse* old_dw, DataWarehouse* new_dw);
    void timeAdvance(const ProcessorGroup*,
		     const PatchSubset* patches,
		     const MaterialSubset* matls,
		     DataWarehouse* old_dw, DataWarehouse* new_dw);
    ExamplesLabel* lb_;
    SimulationStateP sharedState_;
    double delt_;
    SimpleMaterial* mymat_;

    Burger(const Burger&);
    Burger& operator=(const Burger&);
	 
  };
}

#endif
