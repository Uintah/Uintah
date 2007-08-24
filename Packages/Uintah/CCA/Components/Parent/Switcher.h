#ifndef Packages_Uintah_CCA_Components_Switcher_h
#define Packages_Uintah_CCA_Components_Switcher_h

#include <Packages/Uintah/Core/Parallel/UintahParallelComponent.h>
#include <Packages/Uintah/CCA/Ports/SimulationInterface.h>
#include <Packages/Uintah/Core/Grid/Variables/ComputeSet.h>
#include <Packages/Uintah/Core/Grid/Variables/VarLabel.h>

#include <map>
#include <set>
using std::map;
using std::set;

namespace Uintah {
  class Switcher : public UintahParallelComponent, public SimulationInterface {
  public:
    Switcher(const ProcessorGroup* myworld, ProblemSpecP& ups, bool doAMR);
    virtual ~Switcher();

    virtual void problemSetup(const ProblemSpecP& params, 
                              const ProblemSpecP& restart_prob_spec, GridP& grid,
			      SimulationStateP&);

    virtual void outputProblemSpec(ProblemSpecP& ps);
    virtual void outputPS(Dir& dir);
    virtual void scheduleInitialize(const LevelP& level,
				    SchedulerP& sched);
    virtual void scheduleComputeStableTimestep(const LevelP& level,
					       SchedulerP&);
    virtual void scheduleTimeAdvance( const LevelP& level, 
				      SchedulerP&);

    virtual void scheduleSwitchTest(const LevelP& level, SchedulerP& sched);
    virtual void scheduleInitNewVars(const LevelP& level, SchedulerP& sched);
    virtual void scheduleCarryOverVars(const LevelP& level, SchedulerP& sched);
    virtual void scheduleSwitchInitialization(const LevelP& level, 
                                              SchedulerP& sched);
    virtual void scheduleFinalizeTimestep(const LevelP& level, SchedulerP& sched);

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


    // AMR
    virtual void scheduleRefineInterface(const LevelP& fineLevel,
                                         SchedulerP& scheduler,
                                         bool needCoarseOld, 
                                         bool needCoarseNew);
                                         
    virtual void scheduleRefine (const PatchSet* patches, 
                                 SchedulerP& sched); 
    
    virtual void scheduleCoarsen(const LevelP& coarseLevel, 
                                 SchedulerP& sched);


    virtual void scheduleInitialErrorEstimate(const LevelP& coarseLevel,
                                              SchedulerP& sched);
                                               
    virtual void scheduleErrorEstimate(const LevelP& coarseLevel,
                                       SchedulerP& sched);

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

    // since tasks are scheduled per-level, we can't turn the switch flag off
    // until they all are done, and since we need to turn it off during compilation,
    // we need to keep track of which levels we've switched
    vector<bool> d_doSwitching;

    bool d_restarting;

    // used to sync other switch tasks
    //VarLabel* d_switchLabel;
    SimulationInterface* d_sim;

    SimulationStateP d_sharedState;
    unsigned int d_numComponents;
    unsigned int d_componentIndex;
    
    set<const VarLabel*, VarLabel::Compare> d_computedVars;
    vector<vector<string> > d_initVars;
    vector<vector<string> > d_initMatls;
    vector<vector<int> > d_initLevels;
    vector<vector<VarLabel*> > d_initVarLabels;

    vector<string> d_carryOverVars;
    vector<VarLabel*> d_carryOverVarLabels;
    vector<MaterialSubset*> d_carryOverVarMatls;
    vector<bool> d_carryOverFinestLevelOnly; // either all levels or finest only
    vector<vector<bool> > d_doCarryOverVarPerLevel; // size to numlevels

    Switcher(const Switcher&);
    Switcher& operator=(const Switcher&);
	 
  };

}

#endif
