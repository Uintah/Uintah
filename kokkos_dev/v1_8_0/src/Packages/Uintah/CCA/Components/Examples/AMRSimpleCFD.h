
#ifndef Packages_Uintah_CCA_Components_Examples_AMRSimpleCFD_h
#define Packages_Uintah_CCA_Components_Examples_AMRSimpleCFD_h

#include <Packages/Uintah/CCA/Components/Examples/SimpleCFD.h>

namespace Uintah {
  class AMRSimpleCFD : public SimpleCFD {
  public:
    AMRSimpleCFD(const ProcessorGroup* myworld);
    virtual ~AMRSimpleCFD();
 
    virtual void problemSetup(const ProblemSpecP& params, GridP& grid,
		      SimulationStateP& sharedState);
    virtual void scheduleInitialize(const LevelP& level,
				    SchedulerP& sched);
    virtual void scheduleRefineInterface(const LevelP& fineLevel,
					 SchedulerP& scheduler,
					 int step, int nsteps);
    virtual void scheduleCoarsen(const LevelP& coarseLevel, SchedulerP& sched);
    virtual void scheduleErrorEstimate(const LevelP& coarseLevel,
				       SchedulerP& sched);
  protected:
    virtual void refineBoundaries(const Patch* patch,
				  CCVariable<double>& val,
				  DataWarehouse* new_dw,
				  const VarLabel* label,
				  int matl, double factor);
    virtual void refineBoundaries(const Patch* patch,
				  SFCXVariable<double>& val,
				  DataWarehouse* new_dw,
				  const VarLabel* label,
				  int matl, double factor);
    virtual void refineBoundaries(const Patch* patch,
				  SFCYVariable<double>& val,
				  DataWarehouse* new_dw,
				  const VarLabel* label,
				  int matl, double factor);
    virtual void refineBoundaries(const Patch* patch,
				  SFCZVariable<double>& val,
				  DataWarehouse* new_dw,
				  const VarLabel* label,
				  int matl, double factor);
    virtual void addRefineDependencies(Task* task, const VarLabel* var,
			       int step, int nsteps);

  private:

    void initialize(const ProcessorGroup*,
		    const PatchSubset* patches, const MaterialSubset* matls,
		    DataWarehouse* old_dw, DataWarehouse* new_dw);
    void refineInterface(const ProcessorGroup*,
			 const PatchSubset* patches,
			 const MaterialSubset* matls,
			 DataWarehouse*, DataWarehouse* new_dw,
			 double factor);
    void coarsen(const ProcessorGroup*,
		 const PatchSubset* patches,
		 const MaterialSubset* matls,
		 DataWarehouse*, DataWarehouse* new_dw);
    void errorEstimate(const ProcessorGroup*,
		       const PatchSubset* patches,
		       const MaterialSubset* matls,
		       DataWarehouse*, DataWarehouse* new_dw);
    AMRSimpleCFD(const AMRSimpleCFD&);
    AMRSimpleCFD& operator=(const AMRSimpleCFD&);

    double err_density_grad;
    double err_temperature_grad;
    double err_pressure_grad;
    double err_vorticity_mag;
  };
}

#endif
