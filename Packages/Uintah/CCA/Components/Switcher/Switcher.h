#ifndef Packages_Uintah_CCA_Components_Switcher_h
#define Packages_Uintah_CCA_Components_Switcher_h

#include <Packages/Uintah/Core/Parallel/UintahParallelComponent.h>
#include <Packages/Uintah/CCA/Ports/SimulationInterface.h>
#include <Packages/Uintah/Core/Grid/Variables/VarLabel.h>

namespace Uintah {

  class SimpleMaterial;

  class Switcher : public UintahParallelComponent, public SimulationInterface {
  public:
    Switcher(const ProcessorGroup* myworld);
    virtual ~Switcher();

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

    SimulationStateP sharedState_;

    Switcher(const Switcher&);
    Switcher& operator=(const Switcher&);
	 
  };


}





#endif
