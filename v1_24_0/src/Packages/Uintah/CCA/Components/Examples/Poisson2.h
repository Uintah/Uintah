
#ifndef Packages_Uintah_CCA_Components_Examples_Poisson2_h
#define Packages_Uintah_CCA_Components_Examples_Poisson2_h

#include <Packages/Uintah/Core/Parallel/UintahParallelComponent.h>
#include <Packages/Uintah/CCA/Ports/SimulationInterface.h>
#include <Packages/Uintah/Core/Grid/ComputeSet.h>

namespace Uintah {
  class SimpleMaterial;
  class ExamplesLabel;

/**************************************

CLASS
   Poisson2
   
   Poisson2 simulation

GENERAL INFORMATION

   Poisson2.h

   Steven G. Parker
   Department of Computer Science
   University of Utah

   Center for the Simulation of Accidental Fires and Explosions (C-SAFE)
  
   Copyright (C) 2000 SCI Group

KEYWORDS
   Poisson2

DESCRIPTION
   Long description...
  
WARNING
  
****************************************/

  class Poisson2 : public UintahParallelComponent, public SimulationInterface {
  public:
    Poisson2(const ProcessorGroup* myworld);
    virtual ~Poisson2();

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
		     DataWarehouse* old_dw, DataWarehouse* new_dw,
		     LevelP, Scheduler*);
    void iterate(const ProcessorGroup*,
		 const PatchSubset* patches,
		 const MaterialSubset* matls,
		 DataWarehouse* old_dw, DataWarehouse* new_dw);
    ExamplesLabel* lb_;
    SimulationStateP sharedState_;
    double delt_;
    double maxresidual_;
    SimpleMaterial* mymat_;

    Poisson2(const Poisson2&);
    Poisson2& operator=(const Poisson2&);
	 
  };
}

#endif
