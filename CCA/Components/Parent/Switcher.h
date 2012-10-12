/*
 * Copyright (c) 1997-2012 The University of Utah
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the \"Software\"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and\/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED \"AS IS\", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 */

#ifndef Packages_Uintah_CCA_Components_Switcher_h
#define Packages_Uintah_CCA_Components_Switcher_h

#include <Core/Parallel/UintahParallelComponent.h>
#include <CCA/Ports/SimulationInterface.h>
#include <Core/Grid/Variables/ComputeSet.h>
#include <Core/Grid/Variables/VarLabel.h>

#include <map>
#include <set>
using std::map;
using std::set;

namespace Uintah {
  class Switcher : public UintahParallelComponent, public SimulationInterface {
  public:
    Switcher( const ProcessorGroup* myworld, ProblemSpecP& ups, bool doAMR, const string & uda );
    virtual ~Switcher();

    virtual void problemSetup(const ProblemSpecP& params, 
                              const ProblemSpecP& restart_prob_spec, 
                              GridP& grid,
                              SimulationStateP&);

    virtual void outputProblemSpec(ProblemSpecP& ps);
    virtual void outputPS(Dir& dir);
    virtual void scheduleInitialize(            const LevelP& level, SchedulerP& sched);
    virtual void scheduleComputeStableTimestep( const LevelP& level, SchedulerP& sched);
    virtual void scheduleTimeAdvance(           const LevelP& level, SchedulerP& sched);

    virtual void scheduleSwitchTest(            const LevelP& level, SchedulerP& sched);
    virtual void scheduleInitNewVars(           const LevelP& level, SchedulerP& sched);
    virtual void scheduleCarryOverVars(         const LevelP& level, SchedulerP& sched);
    virtual void scheduleSwitchInitialization(  const LevelP& level, SchedulerP& sched);
    virtual void scheduleFinalizeTimestep(      const LevelP& level, SchedulerP& sched);

    virtual bool needRecompile(double time, double delt, const GridP& grid);
    virtual void restartInitialize();

    virtual bool restartableTimesteps();

    virtual double recomputeTimestep(double);


    // direct component to add a new material
    virtual void addMaterial( const ProblemSpecP& params, 
                              GridP& grid,
                              SimulationStateP& state );

    virtual void scheduleInitializeAddedMaterial( const LevelP & level,
                                                  SchedulerP   & scheduler );




    // AMR
    virtual void scheduleRefineInterface(const LevelP& fineLevel,
                                         SchedulerP& scheduler,
                                         bool needCoarseOld, 
                                         bool needCoarseNew);
                                         
    virtual void scheduleRefine (const PatchSet* patches,  SchedulerP& sched); 
    
    virtual void scheduleCoarsen(const LevelP& coarseLevel, SchedulerP& sched);

    virtual void scheduleInitialErrorEstimate(const LevelP& coarseLevel,SchedulerP& sched);
                                               
    virtual void scheduleErrorEstimate(       const LevelP& coarseLevel,SchedulerP& sched);


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
                    
    void readSwitcherState( const ProblemSpecP&, 
                            SimulationStateP& state );

    ProblemSpecP d_master_ups;

    switchState d_switchState;

    // since tasks are scheduled per-level, we can't turn the switch flag off
    // until they all are done, and since we need to turn it off during compilation,
    // we need to keep track of which levels we've switched
    vector<bool> d_doSwitching;

    bool d_restarting;

    SimulationInterface* d_sim;

    SimulationStateP d_sharedState;
    unsigned int d_numComponents;
    unsigned int d_componentIndex;
    
    struct initVars{
      vector<string>            varNames;
      vector<string>            matlSetNames;
      vector<const MaterialSet*> matls;
      vector<int>               levels;
      vector<VarLabel*>         varLabels;
    };
    
    map<int, initVars*> d_initVars;
    
    set<const VarLabel*, VarLabel::Compare> d_computedVars;
    

    vector<string>          d_in_file;                  // contains the name of all the subcomponent inputfiles
    vector<string>          d_carryOverVars;
    vector<VarLabel*>       d_carryOverVarLabels;
    vector<MaterialSubset*> d_carryOverVarMatls;
    vector<bool>            d_carryOverFinestLevelOnly; // either all levels or finest only
    vector<vector<bool> >   d_doCarryOverVarPerLevel;   // size to numlevels

    Switcher(const Switcher&);
    Switcher& operator=(const Switcher&);
	 
  };

}

#endif
