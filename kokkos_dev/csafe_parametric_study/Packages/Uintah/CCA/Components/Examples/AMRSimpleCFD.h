
#ifndef Packages_Uintah_CCA_Components_Examples_AMRSimpleCFD_h
#define Packages_Uintah_CCA_Components_Examples_AMRSimpleCFD_h

#include <Packages/Uintah/CCA/Components/Examples/SimpleCFD.h>
#include <Packages/Uintah/Core/Grid/Task.h>

#include <Packages/Uintah/CCA/Components/Examples/share.h>
namespace Uintah {
  class SCISHARE AMRSimpleCFD : public SimpleCFD {
  public:
    AMRSimpleCFD(const ProcessorGroup* myworld);
    virtual ~AMRSimpleCFD();
 
    virtual void problemSetup(const ProblemSpecP& params, 
                              const ProblemSpecP& restart_prob_spec, 
                              GridP& grid, SimulationStateP& sharedState);
    virtual void scheduleInitialize(const LevelP& level,
				    SchedulerP& sched);
    virtual void scheduleRefineInterface(const LevelP& fineLevel,
					 SchedulerP& scheduler,
					 bool needCoarseOld, bool needCoarseNew);
    virtual void scheduleCoarsen(const LevelP& coarseLevel, SchedulerP& sched);
    virtual void scheduleRefine (const PatchSet* patches, SchedulerP& sched);

    virtual void scheduleErrorEstimate(const LevelP& coarseLevel,
				       SchedulerP& sched);
    virtual void scheduleInitialErrorEstimate(const LevelP& coarseLevel,
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
                                       bool, bool);

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
    void refine(const ProcessorGroup*,
		 const PatchSubset* patches,
		 const MaterialSubset* matls,
		 DataWarehouse*, DataWarehouse* new_dw);
    void errorEstimate(const ProcessorGroup*,
		       const PatchSubset* patches,
		       const MaterialSubset* matls,
		       DataWarehouse*, DataWarehouse* new_dw, bool initial);
    AMRSimpleCFD(const AMRSimpleCFD&);
    AMRSimpleCFD& operator=(const AMRSimpleCFD&);

    double err_density_grad;
    double err_temperature_grad;
    double err_pressure_grad;
    double err_vorticity_mag;
  };
}

#endif
