#ifndef Packages_Uintah_CCA_Components_Examples_Test1_h
#define Packages_Uintah_CCA_Components_Examples_Test1_h

#include <Packages/Uintah/Core/Parallel/UintahParallelComponent.h>
#include <Packages/Uintah/CCA/Ports/SimulationInterface.h>
#include <Packages/Uintah/Core/Grid/Variables/VarLabel.h>

namespace Uintah {

  class SimpleMaterial;

  class Test1 : public UintahParallelComponent, public SimulationInterface {
  public:
    Test1(const ProcessorGroup* myworld);
    virtual ~Test1();

    virtual void problemSetup(const ProblemSpecP& params, GridP& grid,
			      SimulationStateP&);
    virtual void scheduleInitialize(const LevelP& level,
				    SchedulerP& sched);
    virtual void scheduleComputeStableTimestep(const LevelP& level,
					       SchedulerP&);
    virtual void scheduleTimeAdvance(const LevelP& level, 
				      SchedulerP&, int step, int nsteps );

    virtual void scheduleSwitchTest(const LevelP&, SchedulerP& );

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

    void switchTest(const ProcessorGroup*,
                    const PatchSubset* patches,
                    const MaterialSubset* matls,
                    DataWarehouse* old_dw, DataWarehouse* new_dw);

    double delt_;
    SimpleMaterial* matl;
    SimulationStateP sharedState_;

    const VarLabel* SVariableLabel;
    const VarLabel* delt_label;
    const VarLabel* switchLabel;

    Test1(const Test1&);
    Test1& operator=(const Test1&);
	 
  };


}





#endif
