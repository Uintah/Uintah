
#ifndef Packages_Uintah_CCA_Components_Examples_AMRWave_h
#define Packages_Uintah_CCA_Components_Examples_AMRWave_h

#include <Packages/Uintah/CCA/Components/Examples/Wave.h>

namespace Uintah {
  class SimpleMaterial;
  class ExamplesLabel;

/**************************************

CLASS
   AMRWave
   
   AMRWave simulation

GENERAL INFORMATION

   AMRWave.h

   Steven Parker
   Department of Computer Science
   University of Utah

   Center for the Simulation of Accidental Fires and Explosions (C-SAFE)
  
   Copyright (C) 2002 SCI Group

KEYWORDS
   AMRWave

DESCRIPTION
  
WARNING
  
****************************************/

  class VarLabel;
  class AMRWave : public Wave {
  public:
    AMRWave(const ProcessorGroup* myworld);
    virtual ~AMRWave();

    virtual void problemSetup(const ProblemSpecP& params, GridP& grid,
		      SimulationStateP& sharedState);
    virtual void scheduleRefineInterface(const LevelP& fineLevel,
					 SchedulerP& scheduler,
					 int step, int nsteps);
    virtual void scheduleCoarsen(const LevelP& coarseLevel, SchedulerP& sched);
    virtual void scheduleRefine (const LevelP& fineLevel, SchedulerP& sched);

    virtual void scheduleErrorEstimate(const LevelP& coarseLevel,
				       SchedulerP& sched);
    virtual void scheduleInitialErrorEstimate(const LevelP& coarseLevel,
                                              SchedulerP& sched);
    virtual void scheduleTimeAdvance( const LevelP& level, 
				      SchedulerP&, int step, int nsteps );
  private:
    void errorEstimate(const ProcessorGroup*,
		       const PatchSubset* patches,
		       const MaterialSubset* matls,
		       DataWarehouse*, DataWarehouse* new_dw);
    void refine(const ProcessorGroup*,
                const PatchSubset* patches,
                const MaterialSubset* matls,
                DataWarehouse*, DataWarehouse* new_dw);
    void coarsen(const ProcessorGroup*,
                 const PatchSubset* patches,
                 const MaterialSubset* matls,
                 DataWarehouse*, DataWarehouse* new_dw);

    AMRWave(const AMRWave&);
    AMRWave& operator=(const AMRWave&);
	 
    double refine_threshold;
  };
}

#endif
