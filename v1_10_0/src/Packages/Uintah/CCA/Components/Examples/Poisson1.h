
#ifndef Packages_Uintah_CCA_Components_Examples_Poisson1_h
#define Packages_Uintah_CCA_Components_Examples_Poisson1_h

#include <Packages/Uintah/Core/Parallel/UintahParallelComponent.h>
#include <Packages/Uintah/CCA/Ports/SimulationInterface.h>
#include <Packages/Uintah/Core/Grid/ComputeSet.h>

namespace Uintah {
  class SimpleMaterial;
  class ExamplesLabel;

/**************************************

CLASS
   Poisson1
   
   Poisson1 simulation

GENERAL INFORMATION

   Poisson1.h

   Steven G. Parker
   Department of Computer Science
   University of Utah

   Center for the Simulation of Accidental Fires and Explosions (C-SAFE)
  
   Copyright (C) 2000 SCI Group

KEYWORDS
   Poisson1

DESCRIPTION
   Long description...
  
WARNING
  
****************************************/

  class Poisson1 : public UintahParallelComponent, public SimulationInterface {
  public:
    Poisson1(const ProcessorGroup* myworld);
    virtual ~Poisson1();

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

    Poisson1(const Poisson1&);
    Poisson1& operator=(const Poisson1&);
	 
  };
}

#endif
