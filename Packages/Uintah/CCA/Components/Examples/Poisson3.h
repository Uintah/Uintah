
#ifndef Packages_Uintah_CCA_Components_Examples_Poisson3_h
#define Packages_Uintah_CCA_Components_Examples_Poisson3_h

#include <Packages/Uintah/CCA/Components/Examples/Interpolator.h>
#include <Packages/Uintah/Core/Parallel/UintahParallelComponent.h>
#include <Packages/Uintah/CCA/Ports/SimulationInterface.h>
#include <Packages/Uintah/Core/Grid/ComputeSet.h>
#include <Packages/Uintah/Core/Grid/NCVariable.h>

namespace Uintah {
  class SimpleMaterial;
  class ExamplesLabel;

/**************************************

CLASS
   Poisson3
   
   Poisson3 simulation

GENERAL INFORMATION

   Poisson3.h

   Steven G. Parker
   Department of Computer Science
   University of Utah

   Center for the Simulation of Accidental Fires and Explosions (C-SAFE)
  
   Copyright (C) 2000 SCI Group

KEYWORDS
   Poisson3

DESCRIPTION
   Long description...
  
WARNING
  
****************************************/




  class Poisson3 : public UintahParallelComponent, public SimulationInterface {
  public:
    Poisson3(const ProcessorGroup* myworld);
    virtual ~Poisson3();

    virtual void problemSetup(const ProblemSpecP& params, GridP& grid,
			      SimulationStateP&);
    virtual void scheduleInitialize(const LevelP& level,
				    SchedulerP& sched);
    virtual void scheduleComputeStableTimestep(const LevelP& level,
					       SchedulerP&);
    virtual void scheduleTimeAdvance( const LevelP& level, 
				      SchedulerP&, int step, int nsteps );

    // New functions
    virtual void scheduleRefine(const LevelP& fineLevel, SchedulerP& sched);
    void refine(const ProcessorGroup* pg,
                const PatchSubset* finePatches, 
		const MaterialSubset* matls,
                DataWarehouse*, 
                DataWarehouse* newDW);

    virtual void scheduleRefineInterface(const LevelP& fineLevel,
					 SchedulerP& scheduler,
					 int step, int nsteps);
    void refineInterface(const ProcessorGroup*,
			 const PatchSubset* finePatches, 
			 const MaterialSubset* matls,
			 DataWarehouse* fineDW, 
			 DataWarehouse* coarseDW,
			 int step, int nsteps);

    virtual void scheduleCoarsen(const LevelP& coarseLevel, SchedulerP& sched);
    void coarsen(const ProcessorGroup* pg,
	         const PatchSubset* finePatches, 
		 const MaterialSubset* matls,
                 DataWarehouse* coarseDW, 
                 DataWarehouse* fineDW);

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
		     bool modify);
    ExamplesLabel* lb_;
    SimulationStateP sharedState_;
    double delt_;
    SimpleMaterial* mymat_;
    Interpolator interpolator_;
    int max_int_support_;

    Poisson3(const Poisson3&);
    Poisson3& operator=(const Poisson3&);
	 
  };
}



#endif
