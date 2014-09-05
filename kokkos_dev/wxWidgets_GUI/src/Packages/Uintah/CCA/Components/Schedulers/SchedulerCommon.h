
#ifndef UINTAH_HOMEBREW_SCHEDULERCOMMON_H
#define UINTAH_HOMEBREW_SCHEDULERCOMMON_H

#include <Packages/Uintah/Core/Parallel/UintahParallelComponent.h>
#include <Packages/Uintah/Core/Grid/Variables/ComputeSet.h>
#include <Packages/Uintah/CCA/Ports/Scheduler.h>
#include <Packages/Uintah/CCA/Components/Schedulers/TaskGraph.h>
#include <Packages/Uintah/CCA/Components/Schedulers/GhostOffsetVarMap.h>
#include <Packages/Uintah/CCA/Components/Schedulers/OnDemandDataWarehouseP.h>
#include <Packages/Uintah/Core/Grid/SimulationState.h>
#include <Packages/Uintah/Core/Grid/SimulationStateP.h>
#include <sgi_stl_warnings_off.h>
#include <iosfwd>
#include <sgi_stl_warnings_on.h>

namespace Uintah {
  using namespace std;
  class Output;
  class DetailedTask;
  class DetailedTasks;
  class LocallyComputedPatchVarMap;
/**************************************

CLASS
   SchedulerCommon
   
   Short description...

GENERAL INFORMATION

   SchedulerCommon.h

   Steven G. Parker
   Department of Computer Science
   University of Utah

   Center for the Simulation of Accidental Fires and Explosions (C-SAFE)
  
   Copyright (C) 2000 SCI Group

KEYWORDS
   SchedulerCommon

DESCRIPTION
   Long description...
  
WARNING
  
****************************************/

  class SchedulerCommon : public Scheduler, public UintahParallelComponent {
  public:
    SchedulerCommon(const ProcessorGroup* myworld, Output* oport);
    virtual ~SchedulerCommon();

    virtual void problemSetup(const ProblemSpecP& prob_spec,
                              SimulationStateP& state);

    virtual void doEmitTaskGraphDocs();

    //////////
    // Insert Documentation Here:
    virtual void initialize(int numOldDW = 1, int numNewDW = 1,
			    DataWarehouse* parent_old_dw = 0,
			    DataWarehouse* parent_new_dw = 0);

    virtual void clearMappings();
    virtual void mapDataWarehouse(Task::WhichDW, int dwTag);
    void compile();

    //////////
    // Insert Documentation Here:
    virtual void addTask(Task* t, const PatchSet*, const MaterialSet*);

    // get information about the taskgraph
    virtual int getNumTasks() const { return graph.getNumTasks();}
    virtual Task* getTask(int i) {return graph.getTask(i);}

    virtual const vector<const Task::Dependency*>& getInitialRequires();

    virtual LoadBalancer* getLoadBalancer();
    virtual void releaseLoadBalancer();
       
    virtual DataWarehouse* get_dw(int idx);
    virtual DataWarehouse* getLastDW(void);

    virtual void logMemoryUse();
      
    //////////
    // Insert Documentation Here:
    virtual void advanceDataWarehouse(const GridP& grid);
    virtual void fillDataWarehouses(const GridP& grid);
    virtual void replaceDataWarehouse(int index, const GridP& grid);
    virtual void setRestartable(bool restartable);

    // Get the expected extents that may be needed for a particular variable
    // on a particular patch (which should include expected ghost cells.
    //virtual void
    //getExpectedExtents(const VarLabel* label, const Patch* patch,
    //	       IntVector& lowIndex, IntVector& highIndex) const;

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
			 IntVector& requestedHigh) const;
    

    // Makes and returns a map that maps strings to VarLabels of
    // that name and a list of material indices for which that
    // variable is valid (at least according to d_allcomps).
    typedef map< string, list<int> > VarLabelMaterialMap;
    virtual VarLabelMaterialMap* makeVarLabelMaterialMap();

    virtual bool isOldDW(int idx) const;
    virtual bool isNewDW(int idx) const;

    // Only called by the SimulationController, and only once, and only
    // if the simulation has been "restarted."
    virtual void setGeneration( int id ) { d_generation = id; }
    virtual const MaterialSet* getMaterialSet() const {return 0;}

    // This function will copy the data from the old grid to the new grid.
    // The PatchSubset structure will contain a patch on the new grid.
    void copyDataToNewGrid(const ProcessorGroup*,
			   const PatchSubset* patches,
			   const MaterialSubset* ,
			   DataWarehouse* old_dw,
			   DataWarehouse* new_dw);

    virtual void scheduleAndDoDataCopy(const GridP& grid, SimulationInterface* sim);
    
  protected:
    void finalizeTimestep();
    virtual void actuallyCompile() = 0;
    
    void makeTaskGraphDoc(const DetailedTasks* dt,
			  int rank = 0);
    void emitNode(const DetailedTask* dt, double start, double duration,
		  double execution_duration, 
		  double execution_flops,
		  double communication_flops = 0);
    void finalizeNodes(int process=0);

    void printTrackedVars(DetailedTask* dt, bool before);
    
    virtual void verifyChecksum() = 0;
    virtual bool useInternalDeps();

    TaskGraph graph;
    int       d_generation;

    SimulationStateP d_sharedState;

    std::vector<OnDemandDataWarehouseP> dws;
    int numOldDWs;

    int dwmap[Task::TotalDWs];
    DetailedTasks         * dts_;

    Output* m_outPort;
    bool restartable;

    //! These are so we can track certain variables over the taskgraph's
    //! execution.
    vector<string> trackingVars_;
    vector<string> trackingTasks_;
    vector<Task::WhichDW> trackingDWs_;
    double trackingStartTime_;
    double trackingEndTime_;
    int trackingLevel_;
    IntVector trackingStartIndex_;
    IntVector trackingEndIndex_;

    // so we can manually copy vars between AMR levels
    vector<string> copyDataVars_;
  private:

    SchedulerCommon(const SchedulerCommon&);
    SchedulerCommon& operator=(const SchedulerCommon&);

    ProblemSpecP m_graphDoc;
    ProblemSpecP m_nodes;
    ofstream* memlogfile;
    bool emit_taskgraph;
    GhostOffsetVarMap m_ghostOffsetVarMap;
    LocallyComputedPatchVarMap* m_locallyComputedPatchVarMap;

    //! These are to store which vars we have to copy to the new grid
    //! in a copy data task.  Set in scheduleDataCopy and used in
    //! copyDataToNewGrid.
    typedef map<const VarLabel*, MaterialSubset*> label_matl_map;
    vector<label_matl_map> label_matls_;


  };
} // End namespace Uintah

#endif
