
#ifndef UINTAH_HOMEBREW_SCHEDULER_H
#define UINTAH_HOMEBREW_SCHEDULER_H

#include <Packages/Uintah/Core/Parallel/UintahParallelPort.h>
#include <Packages/Uintah/Core/ProblemSpec/ProblemSpecP.h>
#include <Packages/Uintah/Core/Grid/ComputeSet.h>
#include <Packages/Uintah/Core/Grid/LevelP.h>
#include <Packages/Uintah/Core/Grid/Task.h>
#include <map>
#include <list>
#include <string>
#include <set>
#include <vector>

namespace Uintah {
  using namespace std;
  class LoadBalancer;
  class ProcessorGroup;
  class Task;
/**************************************

CLASS
   Scheduler
   
   Short description...

GENERAL INFORMATION

   Scheduler.h

   Steven G. Parker
   Department of Computer Science
   University of Utah

   Center for the Simulation of Accidental Fires and Explosions (C-SAFE)
  
   Copyright (C) 2000 SCI Group

KEYWORDS
   Scheduler

DESCRIPTION
   Long description...
  
WARNING
  
****************************************/

  class Scheduler : public UintahParallelPort {
  public:
    Scheduler();
    virtual ~Scheduler();
    
    virtual void problemSetup(const ProblemSpecP& prob_spec) = 0;
    
    //////////
    // Insert Documentation Here:
    virtual void initialize() = 0;

    virtual void doEmitTaskGraphDocs() = 0;
    
    //////////
    // Insert Documentation Here:
    virtual void compile(const ProcessorGroup * pc, bool init_timestep ) = 0;
    virtual void execute(const ProcessorGroup * pc ) = 0;
       
    //////////
    // Insert Documentation Here:
    virtual void addTask(Task* t, const PatchSet*, const MaterialSet*) = 0;
    
    virtual const vector<const Task::Dependency*>& getInitialRequires() = 0;
    
    virtual LoadBalancer* getLoadBalancer() = 0;
    virtual void releaseLoadBalancer() = 0;
    
    virtual DataWarehouse* get_old_dw() = 0;
    virtual DataWarehouse* get_new_dw() = 0;

    virtual void logMemoryUse() = 0;
      
    //////////
    // Insert Documentation Here:
    virtual void advanceDataWarehouse(const GridP& grid) = 0;
    //    protected:

    //////////
    // Insert Documentation Here:
    virtual void scheduleParticleRelocation(const LevelP& level,
					    const VarLabel* posLabel,
					    const vector<vector<const VarLabel*> >& labels,
					    const VarLabel* new_posLabel,
					    const vector<vector<const VarLabel*> >& new_labels,
					    const MaterialSet* matls) = 0;

    // Makes and returns a map that maps strings to VarLabels of
    // that name and a list of material indices for which that
    // variable is valid (at least according to d_allcomps).
    typedef map< string, list<int> > VarLabelMaterialMap;
    virtual VarLabelMaterialMap* makeVarLabelMaterialMap() = 0;
  private:
    Scheduler(const Scheduler&);
    Scheduler& operator=(const Scheduler&);
  };
} // End namespace Uintah

#endif
