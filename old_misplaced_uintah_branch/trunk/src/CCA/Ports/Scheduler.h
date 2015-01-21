
#ifndef UINTAH_HOMEBREW_SCHEDULER_H
#define UINTAH_HOMEBREW_SCHEDULER_H

#include <Core/Parallel/UintahParallelPort.h>
#include <Core/ProblemSpec/ProblemSpecP.h>
#include <Core/Grid/Variables/ComputeSet.h>
#include <Core/Grid/SimulationStateP.h>
#include <Core/Grid/GridP.h>
#include <Core/Grid/LevelP.h>
#include <Core/Grid/Task.h>
#include <sgi_stl_warnings_off.h>
#include <map>
#include <list>
#include <string>
#include <set>
#include <vector>
#include <sgi_stl_warnings_on.h>

#include <CCA/Ports/uintahshare.h>

namespace Uintah {
  using namespace std;
  class LoadBalancer;
  class Task;
  class SimulationInterface;
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

  class UINTAHSHARE Scheduler : public UintahParallelPort {
  public:
    Scheduler();
    virtual ~Scheduler();
    
    // Only called by the SimulationController, and only once, and only
    // if the simulation has been "restarted."
    virtual void setGeneration( int id ) = 0;

    virtual void problemSetup(const ProblemSpecP& prob_spec, 
                              SimulationStateP& state) = 0;
    
    //////////
    // Insert Documentation Here:
    virtual void initialize(int numOldDW = 1, int numNewDW = 1) = 0;

    virtual void setParentDWs(DataWarehouse* parent_old_dw, 
                              DataWarehouse* parent_new_dw) = 0;

    virtual void clearMappings() = 0;
    virtual void mapDataWarehouse(Task::WhichDW, int dwTag) = 0;

    virtual void doEmitTaskGraphDocs() = 0;
    
    //////////
    // Insert Documentation Here:
    virtual void compile() = 0;
    virtual void execute(int tgnum = 0, int iteration = 0) = 0;

    virtual SchedulerP createSubScheduler() = 0;
       
    //////////
    // Insert Documentation Here:

    enum tgType { NormalTaskGraph, IntermediateTaskGraph };

    virtual void addTaskGraph(tgType type) = 0;
    virtual int getNumTaskGraphs() = 0;
    virtual bool useSmallMessages() = 0;
    
    virtual void addTask(Task* t, const PatchSet*, const MaterialSet*) = 0;
    
    virtual const vector<const Task::Dependency*>& getInitialRequires() = 0;
    virtual const set<const VarLabel*, VarLabel::Compare>& getInitialRequiredVars() const = 0;
    virtual const set<const VarLabel*, VarLabel::Compare>& getComputedVars() const = 0;

    virtual LoadBalancer* getLoadBalancer() = 0;
    virtual void releaseLoadBalancer() = 0;

    virtual DataWarehouse* get_dw(int idx) = 0;
    virtual DataWarehouse* getLastDW(void) = 0;

    virtual bool isOldDW(int idx) const = 0;
    virtual bool isNewDW(int idx) const = 0;

    virtual void logMemoryUse() = 0;
      
    //////////
    // Insert Documentation Here:
    virtual void advanceDataWarehouse(const GridP& grid, bool initialization=false) = 0;
    virtual void fillDataWarehouses(const GridP& grid) = 0;
    virtual void replaceDataWarehouse(int index, const GridP& grid, bool initialization=false) = 0;
    virtual void setRestartable(bool restartable) = 0;

    //    protected:

    //////////
    // Insert Documentation Here:
    virtual void setPositionVar(const VarLabel* posLabel) = 0;
    
    virtual void scheduleParticleRelocation(const LevelP& level,
					    const VarLabel* posLabel,
					    const vector<vector<const VarLabel*> >& labels,
					    const VarLabel* new_posLabel,
					    const vector<vector<const VarLabel*> >& new_labels,
					    const VarLabel* particleIDLabel,
					    const MaterialSet* matls) = 0;

    //! Schedule copying data to new grid after regridding
    virtual void scheduleAndDoDataCopy(const GridP& grid, 
                                       SimulationInterface* sim) = 0;


    virtual void overrideVariableBehavior(string var, bool treatAsOld, 
                                          bool copyData, bool noScrub) = 0;

    // Get the SuperPatch (set of connected patches making a larger rectangle)
    // for the given label and patch and find the largest extents encompassing
    // the expected ghost cells (requiredLow, requiredHigh) and the requested
    // ghost cells as well (requestedLow, requestedHigh) for each of the
    // patches.  Required and requested will besame if requestedNumGCells = 0.
    virtual const vector<const Patch*>*
    getSuperPatchExtents(const VarLabel* label, int matlIndex,
			 const Patch* patch, Ghost::GhostType requestedGType,
			 int requestedNumGCells, IntVector& requiredLow,
			 IntVector& requiredHigh, IntVector& requestedLow,
			 IntVector& requestedHigh) const = 0;
    
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
