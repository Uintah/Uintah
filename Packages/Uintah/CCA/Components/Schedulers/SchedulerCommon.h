
#ifndef UINTAH_HOMEBREW_SCHEDULERCOMMON_H
#define UINTAH_HOMEBREW_SCHEDULERCOMMON_H

#include <Packages/Uintah/Core/Parallel/UintahParallelComponent.h>
#include <Packages/Uintah/CCA/Ports/Scheduler.h>
#include <Packages/Uintah/CCA/Components/Schedulers/TaskGraph.h>
#include <Packages/Uintah/CCA/Components/Schedulers/GhostOffsetVarMap.h>
#include <Packages/Uintah/CCA/Components/Schedulers/LocallyComputedPatchVarMap.h>
#include <Packages/Uintah/CCA/Components/Schedulers/OnDemandDataWarehouseP.h>
#include <iosfwd>

class DOM_Document;
class DOM_Element;

namespace Uintah {
  using namespace std;
  class Output;
  class DetailedTask;
  class DetailedTasks;

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

    virtual void problemSetup(const ProblemSpecP& prob_spec);

    virtual void doEmitTaskGraphDocs();

    //////////
    // Insert Documentation Here:
    virtual void initialize();

    void compile( const ProcessorGroup * pg,  bool scrub_new, bool scrub_old=true );

    //////////
    // Insert Documentation Here:
    virtual void addTask(Task* t, const PatchSet*, const MaterialSet*);

    virtual const vector<const Task::Dependency*>& getInitialRequires();

    virtual LoadBalancer* getLoadBalancer();
    virtual void releaseLoadBalancer();
       
    virtual DataWarehouse* get_old_dw();
    virtual DataWarehouse* get_new_dw();

    virtual void set_old_dw(DataWarehouse*);
    virtual void set_new_dw(DataWarehouse*);

    virtual void logMemoryUse();
      
    //////////
    // Insert Documentation Here:
    virtual void advanceDataWarehouse(const GridP& grid);

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
  protected:
    virtual void actuallyCompile( const ProcessorGroup * pc ) = 0;
    
    void makeTaskGraphDoc(const DetailedTasks* dt,
			  int rank = 0);
    void emitNode(const DetailedTask* dt, double start, double duration,
		  double execution_duration, 
		  long long execution_flops,
		  long long communication_flops = 0);
    void finalizeNodes(int process=0);
    void scrub(const DetailedTask* dt);
    
    virtual void verifyChecksum() = 0;
    virtual bool useInternalDeps();

    TaskGraph graph;
    int       d_generation;

    OnDemandDataWarehouseP dws_[2];
    DetailedTasks         * dts_;

    Output* m_outPort;
  private:

    SchedulerCommon(const SchedulerCommon&);
    SchedulerCommon& operator=(const SchedulerCommon&);

    DOM_Document* m_graphDoc;
    DOM_Element* m_nodes;
    ofstream* memlogfile;
    bool emit_taskgraph;
    GhostOffsetVarMap m_ghostOffsetVarMap;
    LocallyComputedPatchVarMap m_locallyComputedPatchVarMap;
  };
} // End namespace Uintah

#endif
