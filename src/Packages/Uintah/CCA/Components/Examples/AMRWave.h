
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
    virtual void scheduleRefine (const PatchSet* patches, SchedulerP& sched);

    virtual void scheduleErrorEstimate(const LevelP& coarseLevel,
				       SchedulerP& sched);
    virtual void scheduleInitialErrorEstimate(const LevelP& coarseLevel,
                                              SchedulerP& sched);
    virtual void scheduleTimeAdvance( const LevelP& level, 
				      SchedulerP&, int step, int nsteps );
  protected:
    virtual void addRefineDependencies( Task* /*task*/, const VarLabel* /*label*/,
                                        int /*step*/, int /*nsteps*/ );
    virtual void refineFaces(const Patch* finePatch, const Level* fineLevel, const Level* coarseLevel, 
                      CCVariable<double>& finevar, const VarLabel* label, int step, int nsteps,
                      int matl, DataWarehouse* coarse_old_dw, DataWarehouse* coarse_new_dw);
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

    void refineCell(CCVariable<double>& finevar, constCCVariable<double>& coarsevar, IntVector fineIndex,
                    const Level* fineLevel, const Level* coarseLevel); 
    void coarsenCell(CCVariable<double>& coarsevar, constCCVariable<double>& finevar, IntVector coarseIndex,
                    const Level* fineLevel, const Level* coarseLevel); 

    AMRWave(const AMRWave&);
    AMRWave& operator=(const AMRWave&);
	 
    double refine_threshold;
    bool do_refineFaces;
    bool do_refine;
    bool do_coarsen;
  };
}

#endif
