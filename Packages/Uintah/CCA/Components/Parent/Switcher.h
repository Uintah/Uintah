#ifndef Packages_Uintah_CCA_Components_Switcher_h
#define Packages_Uintah_CCA_Components_Switcher_h

#include <Packages/Uintah/Core/Parallel/UintahParallelComponent.h>
#include <Packages/Uintah/CCA/Ports/SimulationInterface.h>
#include <Packages/Uintah/Core/Grid/Variables/VarLabel.h>

namespace Uintah {


  class Switcher : public UintahParallelComponent, public SimulationInterface {
  public:
    enum switchState {idle, switching, post_switch};

    Switcher(const ProcessorGroup* myworld, ProblemSpecP& ups, bool doAMR);
    virtual ~Switcher();

    virtual void problemSetup(const ProblemSpecP& params, GridP& grid,
			      SimulationStateP&);
    virtual void scheduleInitialize(const LevelP& level,
				    SchedulerP& sched);
    virtual void scheduleComputeStableTimestep(const LevelP& level,
					       SchedulerP&);
    virtual void scheduleTimeAdvance( const LevelP& level, 
				      SchedulerP&, int step, int nsteps );

    virtual void scheduleSwitchTest(const LevelP& level, SchedulerP& sched);
    virtual void scheduleInitNewVars(const LevelP& level, SchedulerP& sched);
    virtual void scheduleCarryOverVars(const LevelP& level, SchedulerP& sched);

    virtual bool needRecompile(double time, double delt, const GridP& grid);

    virtual void addToTimestepXML(ProblemSpecP&);
    virtual void readFromTimestepXML(const ProblemSpecP&);
  private:
    void switchTest(const ProcessorGroup*,
                    const PatchSubset* patches,
                    const MaterialSubset* matls,
                    DataWarehouse* old_dw, DataWarehouse* new_dw);

    void initNewVars(const ProcessorGroup*,
                    const PatchSubset* patches,
                    const MaterialSubset* matls,
                    DataWarehouse* old_dw, DataWarehouse* new_dw);

    void carryOverVars(const ProcessorGroup*,
                    const PatchSubset* patches,
                    const MaterialSubset* matls,
                    DataWarehouse* old_dw, DataWarehouse* new_dw);

    SimulationInterface* d_sim;

    SimulationStateP d_sharedState;
    unsigned int d_numComponents;
    unsigned int d_componentIndex;
    switchState d_switchState;
    
    const VarLabel* switchLabel;

    Switcher(const Switcher&);
    Switcher& operator=(const Switcher&);
	 
  };


}





#endif
