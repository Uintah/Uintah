#ifndef Packages_Uintah_CCA_Components_Examples_Test_h
#define Packages_Uintah_CCA_Components_Examples_Test_h

#include <Packages/Uintah/Core/Parallel/UintahParallelComponent.h>
#include <Packages/Uintah/CCA/Ports/SimulationInterface.h>
#include <Packages/Uintah/Core/Grid/Variables/VarLabel.h>

namespace Uintah {

  class SimpleMaterial;

  class Test : public UintahParallelComponent, public SimulationInterface {
  public:
    Test(const ProcessorGroup* myworld);
    virtual ~Test();

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

    double delt_;
    SimpleMaterial* matl;
    SimulationStateP sharedState_;

    const VarLabel* SVariableLabel;
    const VarLabel* delt_label;

    Test(const Test&);
    Test& operator=(const Test&);
	 
  };


}





#endif
