#ifndef Packages_Uintah_CCA_Components_Switcher_h
#define Packages_Uintah_CCA_Components_Switcher_h

#include <Packages/Uintah/Core/Parallel/UintahParallelComponent.h>
#include <Packages/Uintah/CCA/Ports/SimulationInterface.h>
#include <Packages/Uintah/Core/Grid/Variables/ComputeSet.h>

#include <map>
using std::map;

namespace Uintah {

  class VarLabel;
  class Switcher : public UintahParallelComponent, public SimulationInterface {
  public:
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
    virtual void restartInitialize();

    virtual bool restartableTimesteps();

    virtual double recomputeTimestep(double);


    // direct component to add a new material
    virtual void addMaterial(const ProblemSpecP& params, GridP& grid,
                             SimulationStateP& state);

    virtual void scheduleInitializeAddedMaterial(const LevelP& level,
                                                 SchedulerP&);

    virtual void addToTimestepXML(ProblemSpecP&);
    virtual void readFromTimestepXML(const ProblemSpecP&);

    enum switchState { idle, switching };
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


    switchState d_switchState;
    // used to sync other switch tasks
    //VarLabel* d_switchLabel;
    SimulationInterface* d_sim;

    SimulationStateP d_sharedState;
    unsigned int d_numComponents;
    unsigned int d_componentIndex;
    
    vector<vector<string> > d_initVars;
    vector<vector<string> > d_initMatls;
    vector<vector<VarLabel*> > d_initVarLabels;

    vector<string> d_carryOverVars;
    vector<VarLabel*> d_carryOverVarLabels;
    vector<MaterialSubset*> d_carryOverVarMatls;

    typedef map<VarLabel*, MaterialSubset*> matlVarsType;
    vector<matlVarsType> d_matlVarsDB; // size to numlevels

    Switcher(const Switcher&);
    Switcher& operator=(const Switcher&);
	 
  };

}

#endif
